import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from queue import Empty, Queue

import numpy as np
from datasets import Dataset, concatenate_datasets, load_from_disk
from httpx import Timeout
from openai import OpenAI
from openai.types.chat import ChatCompletion
from torch.utils.data import DataLoader
from tqdm import tqdm

from mbt.registry import register_api

API_NAME = "openai.chat"
MULTITURN = False
NUM_THREADS = 1
NUM_PROC = int(os.environ.get("OMP_NUM_THREADS", "8"))
LOG_INTERVAL = 0.01
CACHE_INTERVAL = 0.02
RETURN_TEXT = True
APPLY_THINK_FORMAT = False
REQUEST_TIMEOUT = 3600
REQUEST_MAX_RETRIES = 10
VALIDATION_MAX_RETRIES = 10
STRICT_VALIDATION = True
ALLOW_NONE = False
BATCH = False
SLEEP = 300
CANCEL = False
RELOAD = False
SEED = 42

VALID_FINISH_REASONS = {"stop", "tool_calls", "function_call"}


@register_api(API_NAME)
class API:
    """
    OpenAI Chat Completion API wrapper for parallel, offline inference.

    This class handles concurrent API requests using `ThreadPoolExecutor`, managing
    request sharding, local caching, and result aggregation. It supports features like
    multi-turn conversations, output validation (checking finish reasons), and specific
    formatting for reasoning models (e.g., handling <think> tags).

    Args:
      api_config (dict): A configuration dictionary containing the following keys:
        - model_name (str): Identifier for the model, used for creating output directories.
        - request_kwargs (dict): Arguments passed to `client.chat.completions.create` (e.g., `model`, `temperature`, `top_p`).
        - multiturn (bool, optional): If True, treats inputs as multi-turn conversation history. Defaults to MULTITURN.
        - num_threads (int, optional): The number of concurrent threads for making API calls. Defaults to NUM_THREADS.
        - num_shards (int, optional): Number of data shards to split the request queue into. Defaults to num_threads.
        - num_proc (int, optional): Number of CPU workers for dataset mapping operations. Defaults to NUM_PROC.
        - log_interval (float, optional): The frequency ratio (0.0 to 1.0) for logging progress. Defaults to LOG_INTERVAL.
        - cache_interval (float, optional): The frequency ratio (0.0 to 1.0) for saving intermediate results to disk. Defaults to CACHE_INTERVAL.
        - return_text (bool, optional): If True, simplifies the output to text content; otherwise keeps the full response object. Defaults to RETURN_TEXT.
        - apply_think_format (bool, optional): If True, formats reasoning content with <think> tags (useful for R1-like models). Defaults to APPLY_THINK_FORMAT.
        - request_timeout (int, optional): Timeout in seconds for individual API requests. Defaults to REQUEST_TIMEOUT.
        - request_max_retries (int, optional): Max retries for network/connection errors. Defaults to REQUEST_MAX_RETRIES.
        - validation_max_retries (int, optional): Max retries when the model response is invalid (e.g., wrong finish_reason). Defaults to VALIDATION_MAX_RETRIES.
        - strict_validation (bool, optional): If True, raises an error if validation fails after retries. Defaults to STRICT_VALIDATION.
        - allow_none (bool, optional): If True, allows responses with None content. Defaults to ALLOW_NONE.
        - batch (bool, optional): Configuration for OpenAI Batch API (currently implemented but commented out/reserved). Defaults to BATCH.
        - sleep (int, optional): Sleep duration for batch polling. Defaults to SLEEP.
        - cancel (bool, optional): Whether to cancel existing batch jobs. Defaults to CANCEL.
        - reload (bool, optional): Whether to reload/resubmit batch jobs. Defaults to RELOAD.
        - seed (int, optional): Random seed for reproducibility. Defaults to SEED.
    """

    def __init__(self, api_config: dict) -> None:
        super().__init__()
        self.api_config: dict = api_config
        self.model_name: str = api_config["model_name"]
        self.request_kwargs: dict = api_config["request_kwargs"]
        self.multiturn: bool = api_config.get("multiturn", MULTITURN)
        self.num_threads: int = api_config.get("num_threads", NUM_THREADS)
        self.num_shards: int = api_config.get("num_shards", self.num_threads)
        self.num_proc: int = api_config.get("num_proc", NUM_PROC)
        self.log_interval: float = api_config.get("log_interval", LOG_INTERVAL)
        self.cache_interval: float = api_config.get("cache_interval", CACHE_INTERVAL)
        self.return_text: bool = api_config.get("return_text", RETURN_TEXT)
        self.apply_think_format: bool = api_config.get("apply_think_format", APPLY_THINK_FORMAT)
        self.request_timeout: int = api_config.get("request_timeout", REQUEST_TIMEOUT)
        self.request_max_retries: int = api_config.get("request_max_retries", REQUEST_MAX_RETRIES)
        self.validation_max_retries: int = api_config.get("validation_max_retries", VALIDATION_MAX_RETRIES)
        self.strict_validation: bool = api_config.get("strict_validation", STRICT_VALIDATION)
        self.allow_none: bool = api_config.get("allow_none", ALLOW_NONE)
        self.batch: bool = api_config.get("batch", BATCH)
        self.sleep: int = api_config.get("sleep", SLEEP)
        self.cancel: bool = api_config.get("cancel", CANCEL)
        self.reload: bool = api_config.get("reload", RELOAD)
        self.seed: int = api_config.get("seed", SEED)

        if self.multiturn:
            assert self.request_kwargs.get("n", 1) == 1

    def process(self, task_dir: Path) -> Path:
        self.task_dir = task_dir
        self.api_dir = self.task_dir / self.model_name
        self.api_dir.mkdir(parents=True, exist_ok=True)
        with (self.api_dir / "api_config.json").open("w", encoding="utf-8") as f:
            json.dump(self.api_config, f, ensure_ascii=False, indent=4)

        (self.api_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.api_dir / "cache").mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            handlers=[logging.FileHandler(str(self.api_dir / "logs" / f"{timestamp}.log"))],
            force=True,
        )
        self.logger = logging.getLogger(__name__)

        self.client = OpenAI(timeout=Timeout(timeout=self.request_timeout, connect=5.0), max_retries=self.request_max_retries)
        self.requests: Dataset = load_from_disk(str(self.task_dir / "requests"))

        queue = Queue()
        for i in range(min(len(self.requests), self.num_shards)):
            queue.put(i)
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self.call, rank, queue) for rank in range(self.num_threads)]
            _ = [f.result() for f in futures]

        cache_files = sorted([f for f in (self.api_dir / "cache").glob("response_*") if f.is_dir()], key=lambda f: int(f.name.split("_")[1]))
        # responses: Dataset = concatenate_datasets([load_from_disk(str(f)).select_columns("response") for f in cache_files])
        caches = [load_from_disk(str(f)).select_columns("response") for f in cache_files]
        responses: Dataset = Dataset.from_list([response for cache in caches for response in cache])
        stats = compute_stats([turn for turns in responses["response"] for turn in turns] if self.multiturn else responses["response"])
        with (self.api_dir / "stats.json").open("w", encoding="utf-8") as f:
            f.write(json.dumps(stats, ensure_ascii=False, indent=4))
        if self.return_text:
            if self.multiturn:
                responses = responses.map(lambda example: {"valid": [response["choices"][0]["finish_reason"] in VALID_FINISH_REASONS for response in example["response"]]}, num_proc=self.num_proc)
                if self.apply_think_format:
                    responses = responses.map(lambda example: {"response": [f"<think>\n{response['choices'][0]['message']['reasoning_content']}\n</think>\n\n{response['choices'][0]['message']['content']}" for response in example["response"]]}, num_proc=self.num_proc)
                else:
                    responses = responses.map(lambda example: {"response": [response["choices"][0]["message"]["content"] for response in example["response"]]}, num_proc=self.num_proc)
            else:
                responses = responses.map(lambda example: {"valid": [choice["finish_reason"] in VALID_FINISH_REASONS for choice in example["response"]["choices"]]}, num_proc=self.num_proc)
                if self.apply_think_format:
                    responses = responses.map(lambda example: {"response": [f"<think>\n{choice['message']['reasoning_content']}\n</think>\n\n{choice['message']['content']}" for choice in example["response"]["choices"]]}, num_proc=self.num_proc)
                else:
                    responses = responses.map(lambda example: {"response": [choice["message"]["content"] for choice in example["response"]["choices"]]}, num_proc=self.num_proc)
        self.requests = concatenate_datasets([self.requests, responses], axis=1)
        self.requests.save_to_disk(str(self.api_dir / "responses"))

        return self.api_dir

    def call(self, rank: int, queue: Queue) -> None:
        while True:
            try:
                shard_idx = queue.get_nowait()
                self.logger.info(f"[Thread {rank}] Processing shard: {shard_idx}")
            except Empty:
                self.logger.info(f"[Thread {rank}] Shard queue empty. Exiting.")
                return

            shard: Dataset = self.requests.shard(self.num_shards, shard_idx, contiguous=True)
            cache_file = self.api_dir / "cache" / f"response_{shard_idx}"
            cached_response: Dataset = load_from_disk(str(cache_file)) if cache_file.exists() else Dataset.from_dict({"request_id": [], "response": []})

            if cached_response["request_id"] != shard["request_id"][: len(cached_response)]:
                self.logger.error(f"[Thread {rank}] Mismatch between cached response IDs and request IDs in {cache_file}.")
                self.logger.error(f"Cached IDs: {list(cached_response['request_id'])}")
                self.logger.error(f"Request IDs: {shard['request_id'][: len(cached_response)]}")
                raise ValueError

            if len(cached_response) == len(shard):
                stats = compute_stats([turn for turns in cached_response["response"] for turn in turns] if self.multiturn else cached_response["response"])
                for key, value in stats.items():
                    self.logger.info(f"[Thread {rank}] {key}: {value}")
                continue

            dataloader = DataLoader(shard.select(range(len(cached_response), len(shard))), shuffle=False, collate_fn=collate_fn, num_workers=self.num_proc, pin_memory=True)
            log_interval = math.ceil(self.log_interval * math.ceil(len(shard)))
            cache_interval = math.ceil(self.cache_interval * math.ceil(len(shard)))

            response: dict = cached_response.to_dict()
            start_time = time.time()
            self.logger.info(f"[Thread {rank}] Starting inference for {self.request_kwargs['model']}. Total steps: {len(dataloader)}.")

            for i, (request_id, prompt) in tqdm(enumerate(dataloader, start=1), total=len(dataloader)):
                if self.multiturn:
                    history = []
                    outputs = []

                    for user in prompt:
                        history.extend(user)
                        output = self.request(rank, request_id, history)
                        history.append({"role": "assistant", "content": output.choices[0].message.content})
                        outputs.append(output.model_dump())

                    response["request_id"].append(request_id)
                    response["response"].append(outputs)

                else:
                    output = self.request(rank, request_id, prompt)
                    response["request_id"].append(request_id)
                    response["response"].append(output.model_dump())

                if i % log_interval == 0 or i == len(dataloader):
                    elapsed_time = time.time() - start_time
                    total_time = elapsed_time * (len(dataloader) / i)
                    self.logger.info(f"[Thread {rank}] Progress: {i}/{len(dataloader)} steps. Time elapsed/estimated: {str(timedelta(seconds=int(elapsed_time)))}/{str(timedelta(seconds=int(total_time)))}.")

                if i % cache_interval == 0 or i == len(dataloader):
                    Dataset.from_dict(response).save_to_disk(str(cache_file))

            stats = compute_stats([turn for turns in response["response"] for turn in turns] if self.multiturn else response["response"])
            for key, value in stats.items():
                self.logger.info(f"[Thread {rank}] {key}: {value}")

    def request(self, rank: int, request_id: int, prompt) -> ChatCompletion:
        output = self.client.chat.completions.create(messages=prompt, **self.request_kwargs)

        for retry_count in range(1, self.validation_max_retries + 1):
            if self.allow_none:
                valid_choices = [c for c in output.choices if c.finish_reason in VALID_FINISH_REASONS]
                invalid_choices = [c for c in output.choices if c.finish_reason not in VALID_FINISH_REASONS]
                invalid_details = [f"(finish_reason: {c.finish_reason}" for c in invalid_choices]
            else:
                valid_choices = [c for c in output.choices if c.finish_reason in VALID_FINISH_REASONS and c.message.content is not None]
                invalid_choices = [c for c in output.choices if c.finish_reason not in VALID_FINISH_REASONS or c.message.content is None]
                invalid_details = [f"(finish_reason: {c.finish_reason}, content_is_none: {c.message.content is None})" for c in invalid_choices]

            if not invalid_choices:
                for index, choice in enumerate(output.choices):
                    choice.index = index
                return output

            self.logger.warning(f"[Thread {rank}] Re-running inference for request {request_id} ({len(invalid_choices)} choices) due to invalid responses: {invalid_details} (Attempt {retry_count}/{self.validation_max_retries}).")
            _output = self.client.chat.completions.create(messages=prompt, **(self.request_kwargs | {"n": len(invalid_choices)}))
            output.choices = valid_choices + _output.choices

        if self.strict_validation:
            raise RuntimeError(f"Failed to get a valid response for request {request_id} after {self.validation_max_retries} retries.")
        else:
            return output

    # def call_batch(self, rank: int, queue: Queue) -> None:
    #     batch_info_path = self.api_dir / "cache" / f"batch_info_{rank}.json"
    #     request_path = self.api_dir / "cache" / f"request_{rank}.jsonl"
    #     response_path = self.api_dir / "cache" / f"response_{rank}.jsonl"

    #     if response_path.exists():
    #         return

    #     if self.cancel:
    #         if batch_info_path.exists():
    #             batch = load_batch_info(batch_info_path)
    #             self.client.batches.cancel(batch_id=batch.id)
    #         return

    #     if not batch_info_path.exists() or self.reload:
    #         dataset = self.requests.map(build_request, fn_kwargs={"request_kwargs": self.request_kwargs}, with_indices=True, remove_columns=self.requests.column_names, num_proc=self.num_proc)
    #         dataset.to_json(request_path, lines=True, force_ascii=False, num_proc=self.num_proc)
    #         input_file = self.client.files.create(file=open(request_path, "rb"), purpose="batch")
    #         batch = self.client.batches.create(input_file_id=input_file.id, endpoint="/v1/chat/completions", completion_window="24h")
    #         save_batch_info(batch, batch_info_path, "w")

    #     batch = load_batch_info(batch_info_path)
    #     batch = self.wait_batch(batch.id)
    #     outputs = self.client.files.content(batch.output_file_id)
    #     save_batch(outputs, response_path, "w")

    # def wait_batch(self, batch_id):
    #     i = 0
    #     while True:
    #         batch = self.client.batches.retrieve(batch_id)
    #         i += 1
    #         self.logger.info(f"Try {i}: {batch}")
    #         if batch.status == "completed":
    #             return batch
    #         elif batch.status == ["in_progress", "validating", "finalizing", "cancelling"]:
    #             time.sleep(self.sleep)
    #             continue
    #         elif batch.status in ["failed", "expired", "cancelled"]:
    #             self.logger.error(f"Batch failed: {batch.status}")
    #             raise ValueError(f"Batch failed: {batch.status}")
    #         self.logger.error(f"Unseen batch status: {batch.status}")
    #         raise NotImplementedError(f"Unseen batch status: {batch.status}")


def collate_fn(batch: list[dict]) -> tuple[int, list]:
    return batch[0]["request_id"], batch[0]["prompt"]


# def save(responses: List[IndexedResponse], path: Path, mode: str):
#     with open(path, mode, encoding="utf-8") as f:
#         for r in responses:
#             f.write(f"{r.model_dump_json()}\n")


# def load(path: Path) -> List[IndexedResponse]:
#     with open(path, "r", encoding="utf-8") as f:
#         responses = [IndexedResponse.model_validate_json(line) for line in f if line.strip()]
#     responses.sort(key=lambda r: r.index)
#     return responses


# def build_request(example, index, request_kwargs):
#     return {"custom_id": f"request-{index}", "method": "POST", "url": "/v1/chat/completions", "body": {"messages": example["prompt"]} | request_kwargs}


# def save_batch_info(batch: Batch, path: Path, mode: str):
#     with open(path, mode, encoding="utf-8") as f:
#         f.write(batch.model_dump_json())


# def load_batch_info(path: Path):
#     with open(path, "r", encoding="utf-8") as f:
#         batch = Batch.model_validate_json(f.read().strip())
#     return batch


# def save_batch(outputs, path: Path, mode: str):
#     with open(path, mode, encoding="utf-8") as f:
#         f.write(outputs.text)


# def load_batch(path: Path):
#     with open(path, "r", encoding="utf-8") as f:
#         outputs = [json.loads(line) for line in f]
#     responses = [IndexedResponse(index=int(o["custom_id"].split("-")[-1]), response=ChatCompletion.model_validate(o["response"]["body"])) for o in outputs]
#     responses.sort(key=lambda r: r.index)
#     return responses


# def gather(cache_dir: Path, is_batch_api: bool) -> List[IndexedResponse]:
#     cache_files = cache_dir.glob("response_*.jsonl")
#     responses = []
#     for path in cache_files:
#         responses.extend(load_batch(path) if is_batch_api else load(path))
#     responses.sort(key=lambda r: r.index)
#     return responses


def compute_stats(responses: list[dict]) -> dict:
    def count_tokens(tokens: list[int]) -> dict:
        if not tokens:
            return {"mean": 0, "median": 0, "min": 0, "max": 0, "sum": 0}
        array = np.array(tokens)
        return {"mean": round(np.mean(array)), "median": round(np.median(array)), "min": int(np.min(array)), "max": int(np.max(array)), "sum": int(np.sum(array))}

    invalid_reasons = [c["finish_reason"] for response in responses for c in response["choices"] if c["finish_reason"] not in VALID_FINISH_REASONS]
    invalid_counts = {}
    for reason in invalid_reasons:
        invalid_counts[reason] = invalid_counts.get(reason, 0) + 1

    return {
        "prompt_tokens": count_tokens([response["usage"]["prompt_tokens"] for response in responses]),
        "completion_tokens": count_tokens([response["usage"]["completion_tokens"] for response in responses]),
        "invalid_counts": invalid_counts,
    }
