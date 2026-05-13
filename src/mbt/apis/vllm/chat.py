import contextlib
import json
import logging
import math
import os
import random
import signal
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
from datasets import Dataset, load_from_disk
from httpx import Timeout
from omegaconf import OmegaConf
from openai import OpenAI
from tqdm import tqdm

from mbt.registry import register_api

API_NAME = "vllm.chat"

NUM_THREADS = 1
NUM_PROC = int(os.environ.get("OMP_NUM_THREADS", "8"))
LOG_RATIO = 0.01
CACHE_RATIO = 0.1
SAMPLE_RATIO = 0.1
CLIENT_TIMEOUT = 600
CLIENT_MAX_RETRIES = 20
MAX_RETRIES = 0
RETRY_ON = ["length", "content_filter"]
HEALTH_CHECK_TIMEOUT = 3600
HEALTH_CHECK_REQUEST_TIMEOUT = 5
HEALTH_CHECK_INTERVAL = 10
GRACEFUL_SHUTDOWN_TIMEOUT = 60
SIGKILL_WAIT_TIMEOUT = 10
GPU_QUIESCE_SECONDS = 5
DRY_RUN = False
SEED = 42

VALID_FINISH_REASONS = {"stop", "tool_calls", "function_call"}
RESERVED_COLUMNS = frozenset({"request_id", "response", "valid"})
# Vestigial columns from earlier schema iterations — stripped on resume so a
# previously-migrated responses/ doesn't introduce mixed-schema rows when
# concatenated with freshly-completed worker rows.
LEGACY_DROP_COLUMNS = frozenset({"generation_id", "turn_id", "continuation_id"})


@register_api(API_NAME)
class API:
    """vLLM Chat API wrapper.

    Loads a yaml-driven server config, spawns ``vllm serve`` as a process group
    leader, talks to it via the OpenAI client, writes a single shared
    ``responses/`` HF Dataset that workers update under a lock. Periodic
    progress / cache / sample logging is driven by ratios over total work.

    api_config keys:
      - model_name (str, required): output subdir name under task_dir.
      - model_kwargs (dict, required): vllm serve flags. Must contain
        ``config: <path-to-yaml>``; yaml fields seed the merged dict and
        explicit kwargs override on top.
      - request_kwargs (dict): forwarded to client.chat.completions.create.
      - num_threads (int): worker pool size. Default 1.
      - max_retries (int): retries when finish_reason ∈ retry_on. Default 0.
      - retry_on (list[str]): finish_reason values that trigger a retry.
        Default ["length", "content_filter"].
      - log_ratio / cache_ratio / sample_ratio (float): trigger frequencies as
        ratio of total work. Defaults 0.01 / 0.1 / 0.1.
      - client_timeout (int): OpenAI client request timeout. Default 300.
      - client_max_retries (int): OpenAI client network retries. Default 20.
      - health_check_timeout (int): vllm /health polling budget. Default 3600.
      - dry_run (bool): skip server boot and worker loop. Default False.
      - seed (int): forwarded into request_kwargs if not already set. Default 42.
    """

    def __init__(self, api_config: dict) -> None:
        super().__init__()
        self.api_config: dict = api_config
        self.model_name: str = api_config["model_name"]
        self.request_kwargs: dict = dict(api_config.get("request_kwargs", {}))
        self.num_threads: int = api_config.get("num_threads", NUM_THREADS)
        self.num_proc: int = api_config.get("num_proc", NUM_PROC)
        self.max_retries: int = api_config.get("max_retries", MAX_RETRIES)
        self.retry_on: set = set(api_config.get("retry_on", RETRY_ON))
        self.log_ratio: float = api_config.get("log_ratio", LOG_RATIO)
        self.cache_ratio: float = api_config.get("cache_ratio", CACHE_RATIO)
        self.sample_ratio: float = api_config.get("sample_ratio", SAMPLE_RATIO)
        self.client_timeout: int = api_config.get("client_timeout", CLIENT_TIMEOUT)
        self.client_max_retries: int = api_config.get("client_max_retries", CLIENT_MAX_RETRIES)
        self.health_check_timeout: int = api_config.get("health_check_timeout", HEALTH_CHECK_TIMEOUT)
        self.dry_run: bool = api_config.get("dry_run", DRY_RUN)
        self.seed: int = api_config.get("seed", SEED)

        server_kwargs = _normalize_server_kwargs(api_config["model_kwargs"])
        if server_kwargs.get("host") is None:
            server_kwargs["host"] = "0.0.0.0"
        if server_kwargs.get("port") is None:
            server_kwargs["port"] = _find_free_port()
        self.default_model: str = server_kwargs["model"]
        self._host: str = server_kwargs["host"]
        self._port: int = server_kwargs["port"]
        self._serve_args = _build_vllm_serve_args(server_kwargs)
        self._health_check_url = f"http://{self._host}:{self._port}/health"
        self.request_kwargs.setdefault("model", self.default_model)
        self.request_kwargs.setdefault("seed", self.seed)

    # ------------------------------------------------------------ orchestrator

    def process(self, task_dir: Path) -> Path:
        self.task_dir = task_dir
        self.api_dir = task_dir / self.model_name
        self.api_dir.mkdir(parents=True, exist_ok=True)
        (self.api_dir / "logs").mkdir(parents=True, exist_ok=True)

        with (self.api_dir / "api_config.json").open("w", encoding="utf-8") as f:
            json.dump(self.api_config, f, ensure_ascii=False, indent=4)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            handlers=[logging.FileHandler(str(self.api_dir / "logs" / f"{timestamp}.log")), logging.StreamHandler()],
            force=True,
        )
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self._logger = logging.getLogger(__name__)

        server = None
        pgid: int | None = None
        server_log_fp = None
        skip_boot = self.dry_run or self._all_cached()
        try:
            if not skip_boot:
                server, pgid, server_log_fp = self._start_server()
                self._client = OpenAI(
                    api_key="EMPTY",
                    base_url=f"http://{self._host}:{self._port}/v1",
                    timeout=Timeout(timeout=self.client_timeout, connect=5.0),
                    max_retries=self.client_max_retries,
                )
            else:
                self._logger.info("[skip] all responses cached or dry_run; skipping vLLM server boot.")
            self._run()
        finally:
            if server is not None:
                _shutdown_server(server, pgid if pgid is not None else server.pid, server_log_fp, self._logger)

        if hasattr(self, "_results"):
            stats = self._compute_stats(self._results)
            with (self.api_dir / "stats.json").open("w", encoding="utf-8") as f:
                f.write(json.dumps(stats, ensure_ascii=False, indent=4))
        return self.api_dir

    # ----------------------------------------------------------- server boot

    def _start_server(self) -> tuple[subprocess.Popen, int, object]:
        server_log_path = self.api_dir / "logs" / "server.log"
        if server_log_path.exists():
            server_log_path.rename(server_log_path.with_suffix(server_log_path.suffix + f".{int(time.time())}.bak"))
        server_log_fp = open(server_log_path, "ab", buffering=0)  # noqa: SIM115

        self._logger.info(f"Starting vLLM server with command: {' '.join(self._serve_args)}")
        server = subprocess.Popen(
            self._serve_args,
            stdout=server_log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            pgid = os.getpgid(server.pid)
        except ProcessLookupError:
            pgid = server.pid
        self._logger.info(f"vLLM server process started with PID: {server.pid} (pgid={pgid})")

        if not self._wait_for_server_ready(server):
            _shutdown_server(server, pgid, server_log_fp, self._logger)
            raise RuntimeError("Failed to start vLLM server.")
        return server, pgid, server_log_fp

    def _wait_for_server_ready(self, server: subprocess.Popen) -> bool:
        start = time.time()
        while time.time() - start < self.health_check_timeout:
            if server.poll() is not None:
                self._logger.error(f"vLLM server (PID: {server.pid}) exited with code {server.returncode} before becoming ready.")
                return False
            try:
                response = requests.get(self._health_check_url, timeout=HEALTH_CHECK_REQUEST_TIMEOUT)
                if response.status_code == 200:
                    self._logger.info(f"vLLM server is ready at {self._health_check_url}")
                    return True
            except Exception:
                self._logger.info(f"Waiting for vLLM server at {self._health_check_url}...")
            time.sleep(HEALTH_CHECK_INTERVAL)
        self._logger.error(f"vLLM server failed to start within {self.health_check_timeout} seconds.")
        return False

    # ------------------------------------------------------------- realtime

    def _all_cached(self) -> bool:
        responses_dir = self.api_dir / "responses"
        if not responses_dir.exists():
            return False
        requests_ds = load_from_disk(str(self.task_dir / "requests"))
        cached = load_from_disk(str(responses_dir))
        request_ids = {row["request_id"] for row in requests_ds}
        cached_ids = {row["request_id"] for row in cached}
        return request_ids.issubset(cached_ids)

    def _run(self) -> None:
        self._requests: Dataset = load_from_disk(str(self.task_dir / "requests"))
        self._responses_dir = self.api_dir / "responses"

        if self._responses_dir.exists():
            cached = load_from_disk(str(self._responses_dir))
            self._results: dict[int, dict] = {
                row["request_id"]: {k: v for k, v in dict(row).items() if k not in LEGACY_DROP_COLUMNS}
                for row in cached
            }
            completed = set(self._results)
        else:
            self._results = {}
            completed = set()

        self._work_queue: Queue = Queue()
        for row_idx in range(len(self._requests)):
            request = self._requests[row_idx]
            if request["request_id"] in completed:
                continue
            self._work_queue.put((row_idx, request))

        total = len(self._requests)
        pending = self._work_queue.qsize()
        self._completed_count = total - pending
        self._initial_completed = self._completed_count
        self._total = total
        self._log_trigger = max(1, math.ceil(self.log_ratio * total))
        self._cache_trigger = max(1, math.ceil(self.cache_ratio * total))
        self._sample_trigger = max(1, math.ceil(self.sample_ratio * total))
        self._results_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._start_time = time.time()
        self._pbar = tqdm(total=total, initial=self._completed_count, desc="Inference")

        self._log_start_banner()

        if self.dry_run or self._work_queue.qsize() == 0:
            self._logger.info("[skip] dry_run or all responses cached; skipping worker loop.")
            self._pbar.close()
            self._save_responses()
            return

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self._worker, rank) for rank in range(self.num_threads)]
            for f in futures:
                f.result()

        self._pbar.close()
        self._save_responses()

    def _worker(self, rank: int) -> None:
        self._logger.info(f"[Thread {rank}] Worker started.")
        while True:
            try:
                row_idx, request = self._work_queue.get_nowait()
            except Empty:
                self._logger.info(f"[Thread {rank}] Queue empty. Exiting.")
                return

            request_id = request["request_id"]
            messages = request["prompt"]
            response_dump = self._call_api(rank, request_id, messages)
            valid = [c["finish_reason"] in VALID_FINISH_REASONS for c in response_dump.get("choices", [])]
            base_fields = {k: v for k, v in request.items() if k not in RESERVED_COLUMNS}

            with self._results_lock:
                self._results[request_id] = {
                    **base_fields,
                    "request_id": request_id,
                    "response": response_dump,
                    "valid": valid,
                }
                self._completed_count += 1
                count = self._completed_count

            self._pbar.update(1)

            if count % self._log_trigger == 0:
                self._log_progress(count)
            if count == self._initial_completed + 1 or count % self._sample_trigger == 0:
                self._log_sample(request_id, messages, response_dump)
            if count % self._cache_trigger == 0:
                self._save_responses()

    def _call_api(self, rank: int, request_id: int, messages: list) -> dict:
        output = self._client.chat.completions.create(messages=messages, **self.request_kwargs)
        for retry in range(1, self.max_retries + 1):
            choice = output.choices[0]
            if choice.finish_reason in VALID_FINISH_REASONS:
                return output.model_dump()
            if choice.finish_reason not in self.retry_on:
                break
            self._logger.warning(f"[Thread {rank}] Retrying request {request_id}: finish_reason={choice.finish_reason} (Attempt {retry}/{self.max_retries})")
            output = self._client.chat.completions.create(messages=messages, **self.request_kwargs)
        return output.model_dump()

    # ------------------------------------------------------------ persistence

    def _save_responses(self) -> None:
        with self._cache_lock:
            with self._results_lock:
                snapshot = dict(self._results)
            if not snapshot:
                return
            sorted_rows = [snapshot[k] for k in sorted(snapshot)]
            Dataset.from_list(sorted_rows).save_to_disk(str(self._responses_dir))
            _cleanup_stale_shards(self._responses_dir)
            self._logger.info(f"Responses saved: {len(sorted_rows)} rows.")

    # ----------------------------------------------------------------- stats

    def _compute_stats(self, results: dict) -> dict:
        def count_tokens(tokens: list[int]) -> dict:
            if not tokens:
                return {"mean": 0, "median": 0, "min": 0, "max": 0, "sum": 0}
            arr = np.array(tokens)
            return {"mean": round(np.mean(arr)), "median": round(np.median(arr)), "min": int(np.min(arr)), "max": int(np.max(arr)), "sum": int(np.sum(arr))}

        responses = [row["response"] for row in results.values()]
        prompt_tokens = [r.get("usage", {}).get("prompt_tokens", 0) or 0 for r in responses]
        completion_tokens = [r.get("usage", {}).get("completion_tokens", 0) or 0 for r in responses]
        failures: dict[str, int] = {}
        for r in responses:
            for c in r.get("choices", []):
                fr = c.get("finish_reason")
                if fr not in VALID_FINISH_REASONS:
                    failures[fr] = failures.get(fr, 0) + 1
        return {
            "prompt_tokens": count_tokens(prompt_tokens),
            "completion_tokens": count_tokens(completion_tokens),
            "failures": failures,
        }

    # ----------------------------------------------------------------- logs

    def _log_start_banner(self) -> None:
        pending = self._work_queue.qsize()
        sample_line = f"  Sample every : {self._sample_trigger} steps ({100.0 * self._sample_trigger / max(1, self._total):.1f}%)"
        self._logger.info("\n" + "\n".join([
            "══════════════════════════════════════════════════════════════════════",
            "  Inference Start",
            "══════════════════════════════════════════════════════════════════════",
            f"  Model        : {self.default_model}",
            f"  Total        : {self._total} items",
            f"  Cached       : {self._completed_count} items",
            f"  Pending      : {pending} items",
            f"  Threads      : {self.num_threads}",
            f"  Log every    : {self._log_trigger} steps",
            f"  Cache every  : {self._cache_trigger} steps",
            sample_line,
            "══════════════════════════════════════════════════════════════════════",
        ]))

    def _log_progress(self, count: int) -> None:
        elapsed = time.time() - self._start_time
        speed = (count - self._initial_completed) / elapsed if elapsed > 0 else 0
        eta = (self._total - count) / speed if speed > 0 else 0
        pct = 100.0 * count / max(1, self._total)
        remaining = self._total - count
        self._logger.info("\n" + "\n".join([
            "══════════════════════════════════════════════════════════════════════",
            f"  Progress  : {count} / {self._total} ({pct:.1f}%)",
            f"  Speed     : {speed:.1f} items/sec",
            f"  Elapsed   : {timedelta(seconds=int(elapsed))}",
            f"  ETA       : {timedelta(seconds=int(eta))}",
            f"  Remaining : {remaining} items",
            "══════════════════════════════════════════════════════════════════════",
        ]))

    def _log_sample(self, request_id: int, messages: list, response: dict) -> None:
        choice = response.get("choices", [{}])[0]
        usage = response.get("usage", {}) or {}
        msg = choice.get("message", {}) or {}
        lines = [
            "──────────────────────────────────────────────────────────────────────",
            f"  Sample (request_id: {request_id})",
            "──────────────────────────────────────────────────────────────────────",
            "  [Messages]",
        ]
        for m in messages:
            lines.append(f"  {json.dumps(m, ensure_ascii=False)}")
        lines.append("")
        lines.append(f"  [Response] (finish_reason: {choice.get('finish_reason')}, prompt_tokens: {usage.get('prompt_tokens')}, completion_tokens: {usage.get('completion_tokens')})")
        lines.append(f"  {json.dumps(msg, ensure_ascii=False)}")
        lines.append("──────────────────────────────────────────────────────────────────────")
        self._logger.info("\n" + "\n".join(lines))


# --------------------------------------------------------------------- helpers


def _normalize_server_kwargs(server_kwargs: dict) -> dict:
    """Expand ``config:`` yaml into server_kwargs with yaml-as-default semantics.

    None-valued keys are dropped (lets task YAML use ``model: null`` /
    ``config: null`` as placeholders that fall through to yaml defaults). Keys
    are normalized to underscore form so yaml's ``tensor-parallel-size`` and
    CLI's ``tensor_parallel_size`` do not both end up in the merged dict.
    ``_build_vllm_serve_args`` re-emits with hyphens for the vllm CLI.
    """
    sk = {k.replace("-", "_"): v for k, v in server_kwargs.items() if v is not None}
    config_path = sk.pop("config", None)
    if config_path is None:
        return sk
    cfg = OmegaConf.to_container(OmegaConf.load(str(config_path)), resolve=True)
    yaml_dict = {k.replace("-", "_"): v for k, v in cfg.items() if v is not None}
    return {**yaml_dict, **sk}


def _build_vllm_serve_args(server_kwargs: dict) -> list[str]:
    """Convert server_kwargs into ``vllm serve <model> --flag value ...``.

    Forces ``override_generation_config.max_new_tokens=null`` unless the user
    supplied an explicit value, neutralizing the model repo's
    ``generation_config.json::max_new_tokens`` cap so vLLM's server-side
    fallback can return ``max_model_len - prompt_len`` when the client sends
    no ``max_completion_tokens``.
    """
    server_kwargs = dict(server_kwargs)
    user_ogc = dict(server_kwargs.get("override_generation_config") or {})
    user_ogc.setdefault("max_new_tokens", None)
    server_kwargs["override_generation_config"] = user_ogc

    args = ["vllm", "serve", server_kwargs.pop("model")]
    for key, value in server_kwargs.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            else:
                args.extend([flag, "False"])
        elif isinstance(value, (dict, list)):
            args.extend([flag, json.dumps(value)])
        else:
            args.extend([flag, str(value)])
    return args


def _shutdown_server(server: subprocess.Popen, pgid: int, server_log_fp: object, logger: logging.Logger) -> None:
    try:
        if server.poll() is not None:
            logger.info(f"vLLM server (PID: {server.pid}) has already terminated.")
            _reap_process_group(pgid, logger)
            time.sleep(GPU_QUIESCE_SECONDS)
            return

        logger.info(f"Sending SIGTERM to vLLM process group (PID: {server.pid}, pgid={pgid}).")
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGTERM)

        try:
            server.wait(timeout=GRACEFUL_SHUTDOWN_TIMEOUT)
            logger.info(f"vLLM server (PID: {server.pid}) terminated gracefully.")
        except subprocess.TimeoutExpired:
            logger.warning(f"Graceful shutdown timed out after {GRACEFUL_SHUTDOWN_TIMEOUT}s; sending SIGKILL to pgid={pgid}.")
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            try:
                server.wait(timeout=SIGKILL_WAIT_TIMEOUT)
                logger.info(f"vLLM server (PID: {server.pid}) was killed.")
            except subprocess.TimeoutExpired:
                logger.error(f"vLLM server (PID: {server.pid}) did not exit after SIGKILL within {SIGKILL_WAIT_TIMEOUT}s.")

        _reap_process_group(pgid, logger)
        time.sleep(GPU_QUIESCE_SECONDS)
    finally:
        if server_log_fp is not None:
            with contextlib.suppress(Exception):
                server_log_fp.close()


def _reap_process_group(pgid: int, logger: logging.Logger) -> None:
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return
    logger.warning(f"Process group pgid={pgid} still has live members; sending SIGKILL again.")
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)


def _cleanup_stale_shards(dataset_dir: Path) -> None:
    """Drop ``data-*.arrow`` files not listed in the dataset's state.json."""
    state_file = dataset_dir / "state.json"
    if not state_file.exists():
        return
    try:
        state = json.loads(state_file.read_text())
    except Exception:
        return
    current = {f["filename"] for f in state.get("_data_files", [])}
    for f in dataset_dir.glob("data-*.arrow"):
        if f.name not in current:
            with contextlib.suppress(Exception):
                f.unlink()


def _find_free_port(min_port: int = 20000, max_port: int = 29999, attempts: int = 50) -> int:
    for _ in range(attempts):
        port = random.randint(min_port, max_port)
        with socket.socket() as s:
            try:
                s.bind(("", port))
            except OSError:
                continue
        return port
    raise RuntimeError(f"No free port found in [{min_port}, {max_port}] after {attempts} attempts")
