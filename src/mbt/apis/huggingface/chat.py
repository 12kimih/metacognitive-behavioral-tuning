import gc
import json
import logging
import math
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import torch
import torch.multiprocessing as mp
from datasets import Dataset, load_from_disk
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, GenerationConfig, set_seed
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

from mbt.registry import register_api

API_NAME = "huggingface.chat"
BATCH_SIZE = 1
DATA_PARALLEL_SIZE = 1
NUM_PROC = int(os.environ.get("OMP_NUM_THREADS", "8"))
LOG_INTERVAL = 0.01
CACHE_INTERVAL = 0.02
RETURN_TEXT = True
SEED = 42


@register_api(API_NAME)
class API:
    """
    Hugging Face Transformers models wrapper for offline inference with data parallelism.

    This class manages the initialization, data loading, and generation process using
    Hugging Face's AutoModel and AutoTokenizer. It supports multi-GPU processing
    via `torch.multiprocessing`.

    Args:
      api_config (dict): A configuration dictionary containing the following keys:
        - model_name (str): The identifier for the model, used for creating output directories.
        - model_kwargs (dict): Arguments passed to `from_pretrained` (e.g., `pretrained_model_name_or_path`, `device_map`).
        - tokenizer_kwargs (dict, optional): Arguments passed to `AutoTokenizer.from_pretrained`.
        - chat_template_kwargs (dict, optional): Arguments passed to `tokenizer.apply_chat_template`.
        - generation_kwargs (dict, optional): Configuration for generation (e.g., `max_new_tokens`, `temperature`).
        - batch_size (int, optional): The number of samples per batch for inference. Defaults to BATCH_SIZE.
        - data_parallel_size (int, optional): The number of parallel processes (GPUs) to use. Defaults to DATA_PARALLEL_SIZE.
        - num_shards (int, optional): The number of data shards to split the request queue into. Defaults to data_parallel_size.
        - num_proc (int, optional): Number of CPU workers for dataset mapping and dataloading. Defaults to NUM_PROC.
        - log_interval (float, optional): The frequency ratio (0.0 to 1.0) for logging progress. Defaults to LOG_INTERVAL.
        - cache_interval (float, optional): The frequency ratio (0.0 to 1.0) for saving intermediate results to disk. Defaults to CACHE_INTERVAL.
        - return_text (bool, optional): If True, replaces the response object with just the content string in the final output. Defaults to RETURN_TEXT.
        - seed (int, optional): The random seed for reproducibility. Defaults to SEED.
    """

    def __init__(self, api_config: dict) -> None:
        super().__init__()
        self.api_config: dict = api_config
        self.model_name: str = api_config["model_name"]
        self.model_kwargs: dict = api_config["model_kwargs"]
        self.tokenizer_kwargs: dict = api_config.get("tokenizer_kwargs", {})
        self.chat_template_kwargs: dict = api_config.get("chat_template_kwargs", {})
        self.generation_kwargs: dict = api_config.get("generation_kwargs", {})
        self.batch_size: int = api_config.get("batch_size", BATCH_SIZE)
        self.data_parallel_size: int = api_config.get("data_parallel_size", DATA_PARALLEL_SIZE)
        self.num_shards: int = api_config.get("num_shards", self.data_parallel_size)
        self.num_proc: int = api_config.get("num_proc", NUM_PROC)
        self.log_interval: float = api_config.get("log_interval", LOG_INTERVAL)
        self.cache_interval: float = api_config.get("cache_interval", CACHE_INTERVAL)
        self.return_text: bool = api_config.get("return_text", RETURN_TEXT)
        self.seed: int = api_config.get("seed", SEED)

        self.cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        self.gpu_ids = [int(x.strip()) for x in self.cuda_visible_devices.split(",")] if self.cuda_visible_devices else list(range(torch.cuda.device_count()))
        assert len(self.gpu_ids) % self.data_parallel_size == 0
        self.group_size = len(self.gpu_ids) // self.data_parallel_size
        self.gpu_mapping = {i: ",".join(map(str, self.gpu_ids[i * self.group_size : (i + 1) * self.group_size])) for i in range(self.data_parallel_size)}

    def process(self, task_dir: Path) -> Path:
        self.task_dir = task_dir
        self.api_dir = self.task_dir / self.model_name
        self.api_dir.mkdir(parents=True, exist_ok=True)
        with (self.api_dir / "api_config.json").open("w", encoding="utf-8") as f:
            json.dump(self.api_config, f, ensure_ascii=False, indent=4)

        (self.api_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.api_dir / "cache").mkdir(parents=True, exist_ok=True)

        requests: Dataset = load_from_disk(str(self.task_dir / "requests"))

        mp.set_start_method("spawn", force=True)
        with mp.Manager() as manager:
            queue = manager.Queue()
            for i in range(min(len(requests), self.num_shards)):
                queue.put(i)
            with ProcessPoolExecutor(max_workers=self.data_parallel_size) as executor:
                futures = [executor.submit(self.call, rank, queue) for rank in range(self.data_parallel_size)]
                _ = [f.result() for f in futures]

        cache_files = sorted([f for f in (self.api_dir / "cache").glob("response_*") if f.is_dir()], key=lambda f: int(f.name.split("_")[1]))
        # responses: Dataset = concatenate_datasets([load_from_disk(str(f)) for f in cache_files])
        cached_responses = [load_from_disk(str(f)) for f in cache_files]
        responses: Dataset = Dataset.from_list([item for response in cached_responses for item in response])
        if self.return_text:
            responses = responses.map(lambda example: {"response": [choice.message.content for choice in ChatCompletion.model_validate(example["response"]).choices]}, num_proc=self.num_proc)
        requests = requests.add_column("response", responses["response"])
        requests.save_to_disk(str(self.api_dir / "responses"))

        return self.api_dir

    def call(self, rank: int, queue: Queue) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_mapping[rank]
        set_seed(self.seed)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            handlers=[logging.FileHandler(str(self.api_dir / "logs" / datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S_{rank}.log")))],
            force=True,
        )
        logger = logging.getLogger(__name__)

        model = None
        tokenizer = None
        model_loaded = False

        requests: Dataset = load_from_disk(str(self.task_dir / "requests"))

        while True:
            try:
                shard_idx = queue.get_nowait()
                logger.info(f"[Process {rank}] Processing shard: {shard_idx}")
            except Empty:
                logger.info(f"[Process {rank}] Shard queue empty. Exiting.")
                if model_loaded:
                    logger.info(f"[Process {rank}] Releasing model and CUDA memory.")
                    if model is not None:
                        del model
                    if tokenizer is not None:
                        del tokenizer
                    gc.collect()
                    torch.cuda.empty_cache()
                    logger.info(f"[Process {rank}] Resources released. Terminating.")
                return

            shard: Dataset = requests.shard(self.num_shards, shard_idx, contiguous=True)
            cache_file = self.api_dir / "cache" / f"response_{shard_idx}"
            cached_response: Dataset = load_from_disk(str(cache_file)) if cache_file.exists() else Dataset.from_dict({"request_id": [], "response": []})

            if cached_response["request_id"] != shard["request_id"][: len(cached_response)]:
                logger.error(f"[Process {rank}] Mismatch between cached response IDs and request IDs in {cache_file}.")
                logger.error(f"Cached IDs: {list(cached_response['request_id'])}")
                logger.error(f"Request IDs: {shard['request_id'][: len(cached_response)]}")
                raise ValueError

            if len(cached_response) == len(shard):
                tokens = count_tokens(cached_response["response"])
                for k, v in tokens.items():
                    logger.info(f"[Process {rank}] Token stats for {k}: sum={v['sum']}, mean={round(v['mean'])}, min={v['min']}, max={v['max']}")
                continue

            if not model_loaded:
                model_config = AutoConfig.from_pretrained(self.model_kwargs["pretrained_model_name_or_path"])
                if model_config.architectures and any(arch in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values() for arch in model_config.architectures):
                    model = AutoModelForImageTextToText.from_pretrained(**self.model_kwargs, device_map="auto")
                else:
                    model = AutoModelForCausalLM.from_pretrained(**self.model_kwargs, device_map="auto")
                model.eval()

                tokenizer = AutoTokenizer.from_pretrained(model_config._name_or_path, padding_side="left", **self.tokenizer_kwargs)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                generation_config = GenerationConfig.from_pretrained(model_config._name_or_path)
                generation_config.update(pad_token_id=tokenizer.pad_token_id, **self.generation_kwargs)

            dataloader = DataLoader(shard.select(range(len(cached_response), len(shard))), batch_size=self.batch_size, shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer, chat_template_kwargs=self.chat_template_kwargs), num_workers=self.num_proc, pin_memory=True)
            log_interval = math.ceil(self.log_interval * math.ceil(len(shard) / self.batch_size))
            cache_interval = math.ceil(self.cache_interval * math.ceil(len(shard) / self.batch_size))

            response: dict = cached_response.to_dict()
            start_time = time.time()
            logger.info(f"[Process {rank}] Starting inference for {self.model_kwargs['model']}. Total steps: {len(dataloader)}.")

            for i, (request_id, inputs) in tqdm(enumerate(dataloader, start=1), total=len(dataloader)):
                inputs = inputs.to(model.device)
                prompt_tokens = inputs.input_ids.size(-1)

                with torch.no_grad():
                    outputs = model.generate(**inputs, generation_config=generation_config)

                decoded_outputs = tokenizer.batch_decode(outputs[:, prompt_tokens:], skip_special_tokens=True)
                batched_outputs = [decoded_outputs[i : i + generation_config.num_return_sequences] for i in range(0, len(decoded_outputs), generation_config.num_return_sequences)]

                total_tokens = outputs.size(-1)
                completion_tokens = total_tokens - prompt_tokens
                finish_reason = "length" if (completion_tokens == generation_config.max_new_tokens) or (total_tokens == generation_config.max_length) else "stop"

                response["request_id"].extend(request_id)
                response["response"].extend(
                    [
                        ChatCompletion(
                            id=f"chatcmpl-{uuid.uuid4()}",
                            object="chat.completion",
                            created=int(time.time()),
                            model=self.model_kwargs["pretrained_model_name_or_path"],
                            choices=[
                                Choice(
                                    index=i,
                                    message=ChatCompletionMessage(role="assistant", content=content),
                                    finish_reason=finish_reason,
                                )
                                for i, content in enumerate(contents)
                            ],
                            usage=CompletionUsage(
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens * generation_config.num_return_sequences,
                                total_tokens=prompt_tokens + completion_tokens * generation_config.num_return_sequences,
                            ),
                        )
                        for contents in batched_outputs
                    ]
                )

                if i % log_interval == 0 or i == len(dataloader):
                    elapsed_time = time.time() - start_time
                    total_time = elapsed_time * (len(dataloader) / i)
                    logger.info(f"[Process {rank}] Progress: {i}/{len(dataloader)} steps. Time elapsed/estimated: {str(timedelta(seconds=int(elapsed_time)))}/{str(timedelta(seconds=int(total_time)))}.")

                if i % cache_interval == 0 or i == len(dataloader):
                    Dataset.from_dict(response).save_to_disk(str(cache_file))

            tokens = count_tokens(response["response"])
            for k, v in tokens.items():
                logger.info(f"[Process {rank}] Token stats for {k}: sum={v['sum']}, mean={round(v['mean'])}, min={v['min']}, max={v['max']}")


def collate_fn(batch: list[dict], tokenizer, chat_template_kwargs: dict) -> tuple:
    request_id = [example["request_id"] for example in batch]
    prompt = [example["prompt"] for example in batch]
    inputs = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, padding=True, return_tensors="pt", return_dict=True, **chat_template_kwargs)
    return request_id, inputs


def count_tokens(responses: list[dict]) -> dict:
    def compute_stats(tokens: list[int]) -> dict:
        if not tokens:
            return {"sum": 0, "mean": 0, "min": 0, "max": 0}
        array = np.array(tokens)
        return {"sum": np.sum(array), "mean": np.mean(array), "min": np.min(array), "max": np.max(array)}

    return {
        "prompt_tokens": compute_stats([response["usage"]["prompt_tokens"] for response in responses]),
        "completion_tokens": compute_stats([response["usage"]["completion_tokens"] for response in responses]),
    }
