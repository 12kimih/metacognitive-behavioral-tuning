import json
import logging
import math
import os
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import requests
from datasets import Dataset, concatenate_datasets, load_from_disk
from httpx import Timeout
from openai import OpenAI
from openai.types.chat import ChatCompletion
from torch.utils.data import DataLoader
from tqdm import tqdm

from mbt.registry import register_api

API_NAME = "vllm.chat"
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
DRY_RUN = False
SEED = 42

HEALTH_CHECK_TIMEOUT = 3600
HEALTH_CHECK_REQUEST_TIMEOUT = 30
HEALTH_CHECK_INTERVAL = 30
GRACEFUL_SHUTDOWN_TIMEOUT = 30
VALID_FINISH_REASONS = {"stop", "tool_calls", "function_call"}


@register_api(API_NAME)
class API:
    """
    vLLM Chat API wrapper for high-throughput offline inference.

    This class automates the deployment of a local vLLM server as a subprocess,
    manages its lifecycle (startup, health checks, graceful shutdown), and performs
    inference using the OpenAI-compatible API. It handles log redirection from the
    server and supports data parallelism via threading.

    Args:
      api_config (dict): A configuration dictionary containing the following keys:
        - model_name (str): Identifier for the model, used for creating output directories.
        - model_kwargs (dict): Arguments converted to command-line flags for `vllm serve` (e.g., `model`, `tensor_parallel_size`, `gpu_memory_utilization`).
        - request_kwargs (dict, optional): Arguments passed to `client.chat.completions.create` (e.g., `temperature`, `max_tokens`).
        - multiturn (bool, optional): If True, treats inputs as conversation history. Defaults to MULTITURN.
        - num_threads (int, optional): The number of concurrent threads for making API calls. Defaults to NUM_THREADS.
        - num_shards (int, optional): Number of data shards to split the request queue into. Defaults to num_threads.
        - num_proc (int, optional): Number of CPU workers for dataset processing. Defaults to NUM_PROC.
        - log_interval (float, optional): The frequency ratio (0.0 to 1.0) for logging progress. Defaults to LOG_INTERVAL.
        - cache_interval (float, optional): The frequency ratio (0.0 to 1.0) for saving intermediate results to disk. Defaults to CACHE_INTERVAL.
        - return_text (bool, optional): If True, simplifies the output to text content. Defaults to RETURN_TEXT.
        - apply_think_format (bool, optional): If True, formats reasoning content with <think> tags. Defaults to APPLY_THINK_FORMAT.
        - request_timeout (int, optional): Timeout in seconds for individual API requests. Defaults to REQUEST_TIMEOUT.
        - request_max_retries (int, optional): Max retries for network/connection errors. Defaults to REQUEST_MAX_RETRIES.
        - validation_max_retries (int, optional): Max retries when the model response is invalid. Defaults to VALIDATION_MAX_RETRIES.
        - strict_validation (bool, optional): If True, raises an error if validation fails after retries. Defaults to STRICT_VALIDATION.
        - allow_none (bool, optional): If True, allows responses with None content. Defaults to ALLOW_NONE.
        - dry_run (bool, optional): If True, skips starting the vLLM server for testing purposes. Defaults to DRY_RUN.
        - seed (int, optional): Random seed for reproducibility. Defaults to SEED.
    """

    def __init__(self, api_config: dict) -> None:
        super().__init__()
        self.api_config: dict = api_config
        self.model_name: str = api_config["model_name"]
        self.model_kwargs: dict = api_config["model_kwargs"]
        self.request_kwargs: dict = api_config.get("request_kwargs", {})
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
        self.dry_run: bool = api_config.get("dry_run", DRY_RUN)
        self.seed: int = api_config.get("seed", SEED)

        if self.multiturn:
            assert self.request_kwargs.get("n", 1) == 1

        if self.model_kwargs.get("host") is None:
            self.model_kwargs["host"] = "0.0.0.0"

        if self.model_kwargs.get("port") is None:
            self.model_kwargs["port"] = find_free_port()

        if self.request_kwargs.get("model") is None:
            self.request_kwargs["model"] = self.model_kwargs["model"]

        self.args = ["vllm", "serve", self.model_kwargs.pop("model")]
        for key, value in self.model_kwargs.items():
            if isinstance(value, bool) and value:
                self.args.append(f"--{key.replace('_', '-')}")
            else:
                self.args.append(f"--{key.replace('_', '-')}")
                self.args.append(str(value))

        self.health_check_url = f"http://{self.model_kwargs['host']}:{self.model_kwargs['port']}/health"

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
        self.server_logger = self.setup_logger("server", str(self.api_dir / "logs" / f"{timestamp}_server.log"), propagate=False)

        try:
            server = None
            if not self.dry_run:
                self.logger.info(f"Starting vLLM server with command: {' '.join(self.args)}")
                server = subprocess.Popen(self.args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                self.logger.info(f"vLLM server process started with PID: {server.pid}")

                stdout_thread = threading.Thread(target=self.log_server, args=(server.stdout, logging.INFO))
                stderr_thread = threading.Thread(target=self.log_server, args=(server.stderr, logging.ERROR))
                stdout_thread.daemon = True
                stderr_thread.daemon = True
                stdout_thread.start()
                stderr_thread.start()

                if not self.wait_for_server_ready():
                    stdout, stderr = server.communicate()
                    self.logger.error(f"vLLM server stdout: {stdout}")
                    self.logger.error(f"vLLM server stderr: {stderr}")
                    raise RuntimeError("Failed to start vLLM server.")

            self.client = OpenAI(api_key="EMPTY", base_url=f"http://{self.model_kwargs['host']}:{self.model_kwargs['port']}/v1", timeout=Timeout(timeout=self.request_timeout, connect=5.0), max_retries=self.request_max_retries)
            self.requests: Dataset = load_from_disk(str(self.task_dir / "requests"))

            queue = Queue()
            for i in range(min(len(self.requests), self.num_shards)):
                queue.put(i)
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = [executor.submit(self.call, rank, queue) for rank in range(self.num_threads)]
                _ = [f.result() for f in futures]

        finally:
            if server is not None:
                self.shutdown_server(server)

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

    def setup_logger(self, name: str, filename: str, propagate: bool = True) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(filename)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")
        handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(handler)
        logger.propagate = propagate
        return logger

    def log_server(self, stream, level):
        for line in iter(stream.readline, ""):
            self.server_logger.log(level, line.strip())

    def wait_for_server_ready(self) -> bool:
        start_time = time.time()
        while time.time() - start_time < HEALTH_CHECK_TIMEOUT:
            try:
                response = requests.get(self.health_check_url, timeout=HEALTH_CHECK_REQUEST_TIMEOUT)
                if response.status_code == 200:
                    self.logger.info(f"vLLM server is ready at {self.health_check_url}")
                    return True
            except Exception:
                self.logger.info(f"Waiting for vLLM server at {self.health_check_url}...")
            time.sleep(HEALTH_CHECK_INTERVAL)
        self.logger.error(f"vLLM server failed to start within {HEALTH_CHECK_TIMEOUT} seconds.")
        return False

    def shutdown_server(self, server: subprocess.Popen) -> None:
        if server.poll() is not None:
            self.logger.info(f"vLLM server process (PID: {server.pid}) has already terminated.")
            return
        self.logger.info(f"Attempting to gracefully terminate vLLM server process (PID: {server.pid}).")
        server.terminate()
        try:
            server.wait(timeout=GRACEFUL_SHUTDOWN_TIMEOUT)
            self.logger.info(f"vLLM server process (PID: {server.pid}) terminated gracefully.")
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Graceful shutdown timed out. Forcing termination of vLLM server process (PID: {server.pid}).")
            server.kill()
            self.logger.info(f"vLLM server process (PID: {server.pid}) was killed.")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
        return int(port)


def collate_fn(batch: list[dict]) -> tuple[int, list]:
    return batch[0]["request_id"], batch[0]["prompt"]


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
