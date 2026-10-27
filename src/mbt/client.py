"""One chat client for OpenAI-compatible and Anthropic-compatible endpoints.

Every model call in this repository goes through `Client.chat`. The client speaks two wire
formats:

* `openai`: `POST {base_url}/chat/completions`. OpenAI, DeepSeek, OpenRouter, Qwen Cloud
  (DashScope compatible mode), vLLM, SGLang and most other hosted or local servers.
* `anthropic`: `POST {base_url}/v1/messages`. Anthropic, and the Anthropic-compatible endpoints
  of DeepSeek, OpenRouter and Qwen Cloud.

Both are streamed, because a teacher trace can run to more than 100k output tokens and a
non-streamed request of that length is cut by most proxies. The reasoning text is collected from
whichever field the server uses (`reasoning_content`, `reasoning`, Anthropic `thinking` blocks,
or a `<think>` block at the start of the content).
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import httpx

from mbt.endpoints import Endpoint

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
_THINK = re.compile(r"^\s*<think>(.*?)</think>\s*", re.S)


class EndpointError(RuntimeError):
    """The endpoint answered with an error that retrying did not clear."""


@dataclass
class ChatResult:
    """What one call produced."""

    content: str
    reasoning: str
    finish_reason: str | None
    usage: dict[str, Any] = field(default_factory=dict)
    request: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def hit_length_limit(self) -> bool:
        return self.finish_reason in ("length", "max_tokens")


def _split_think(content: str, reasoning: str) -> tuple[str, str]:
    """A server without a reasoning parser returns `<think>...</think>` inside the content."""
    if not reasoning:
        match = _THINK.match(content)
        if match:
            return content[match.end():], match.group(1).strip("\n")
    return content, reasoning


def _sse(response: httpx.Response) -> Iterator[tuple[str | None, dict]]:
    """`(event name, data)` for each server-sent event of a streamed response."""
    event: str | None = None
    for line in response.iter_lines():
        if not line:
            event = None
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
            continue
        if line.startswith("data:"):
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                yield event, json.loads(data)
            except json.JSONDecodeError:
                continue


class Client:
    """A client for one `Endpoint`. Thread-safe; one instance can serve many workers."""

    def __init__(self, endpoint: Endpoint, *, retries: int = 10, timeout: float = 3600.0):
        self.endpoint = endpoint
        self.retries = retries
        key = os.environ.get(endpoint.api_key_env, "") if endpoint.api_key_env else ""
        if endpoint.api_key_env and not key:
            raise EndpointError(f"the environment variable {endpoint.api_key_env} is not set")
        self._key = key
        self._http = httpx.Client(timeout=httpx.Timeout(timeout, connect=60.0))

    # ------------------------------------------------------------------ requests
    def _openai_request(self, messages: list[dict], max_tokens: int | None) -> dict:
        ep = self.endpoint
        body: dict[str, Any] = {"model": ep.model, "messages": messages, "stream": True,
                                "stream_options": {"include_usage": True}}
        if ep.temperature is not None:
            body["temperature"] = ep.temperature
        if ep.top_p is not None:
            body["top_p"] = ep.top_p
        tokens = max_tokens or ep.max_tokens
        if tokens:
            body[ep.max_tokens_field] = tokens
        if ep.reasoning_effort:
            body["reasoning_effort"] = ep.reasoning_effort
        body.update(ep.extra_body)
        return body

    def _anthropic_request(self, messages: list[dict], max_tokens: int | None) -> dict:
        ep = self.endpoint
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        body: dict[str, Any] = {
            "model": ep.model, "stream": True,
            "max_tokens": max_tokens or ep.max_tokens or 32000,
            "messages": [m for m in messages if m["role"] != "system"],
        }
        if system:
            body["system"] = system
        if ep.temperature is not None:
            body["temperature"] = ep.temperature
        if ep.top_p is not None:
            body["top_p"] = ep.top_p
        if ep.reasoning_effort:
            body.setdefault("output_config", {})["effort"] = ep.reasoning_effort
        body.update(ep.extra_body)
        return body

    def _headers(self) -> dict[str, str]:
        ep = self.endpoint
        headers = {"content-type": "application/json", **ep.headers}
        if ep.api == "anthropic":
            headers["anthropic-version"] = "2023-06-01"
        if self._key:
            if ep.auth == "x-api-key":
                headers["x-api-key"] = self._key
            else:
                headers["authorization"] = f"Bearer {self._key}"
        return headers

    def _url(self) -> str:
        base = self.endpoint.base_url.rstrip("/")
        if self.endpoint.api == "anthropic":
            return base + ("/messages" if base.endswith("/v1") else "/v1/messages")
        return base + "/chat/completions"

    # ------------------------------------------------------------------ reading
    @staticmethod
    def _read_openai(response: httpx.Response) -> tuple[str, str, str | None, dict]:
        content, reasoning, finish, usage = [], [], None, {}
        for _event, data in _sse(response):
            if data.get("usage"):
                usage = data["usage"]
            for choice in data.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                for key in ("reasoning_content", "reasoning"):
                    if isinstance(delta.get(key), str) and delta[key]:
                        reasoning.append(delta[key])
                        break
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
        return "".join(content), "".join(reasoning), finish, usage

    @staticmethod
    def _read_anthropic(response: httpx.Response) -> tuple[str, str, str | None, dict]:
        content, reasoning, finish, usage = [], [], None, {}
        for event, data in _sse(response):
            kind = data.get("type") or event
            if kind == "message_start":
                usage.update((data.get("message") or {}).get("usage") or {})
            elif kind == "content_block_delta":
                delta = data.get("delta") or {}
                if delta.get("type") == "text_delta":
                    content.append(delta.get("text", ""))
                elif delta.get("type") == "thinking_delta":
                    reasoning.append(delta.get("thinking", ""))
            elif kind == "message_delta":
                finish = (data.get("delta") or {}).get("stop_reason") or finish
                usage.update(data.get("usage") or {})
            elif kind == "error":
                raise EndpointError(json.dumps(data))
        return "".join(content), "".join(reasoning), finish, usage

    # ------------------------------------------------------------------ the call
    def chat(self, messages: list[dict], *, max_tokens: int | None = None) -> ChatResult:
        """One completion, retried on transport errors and retryable statuses."""
        anthropic = self.endpoint.api == "anthropic"
        body = (self._anthropic_request if anthropic else self._openai_request)(messages, max_tokens)
        reader = self._read_anthropic if anthropic else self._read_openai
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            started = time.time()
            try:
                with self._http.stream("POST", self._url(), headers=self._headers(),
                                       json=body) as response:
                    if response.status_code != 200:
                        text = response.read().decode("utf-8", "replace")
                        if self._key:
                            text = text.replace(self._key, "[REDACTED]")
                        text = text[:2000]
                        error = EndpointError(f"HTTP {response.status_code}: {text}")
                        if response.status_code not in RETRY_STATUS:
                            raise error
                        raise _Retry(error)
                    content, reasoning, finish, usage = reader(response)
                content, reasoning = _split_think(content, reasoning)
                return ChatResult(content=content, reasoning=reasoning, finish_reason=finish,
                                  usage=usage, request={k: v for k, v in body.items()
                                                        if k != "messages"},
                                  seconds=round(time.time() - started, 3))
            except _Retry as retry:
                last = retry.error
            except EndpointError as error:
                detail = str(error)
                if self._key:
                    detail = detail.replace(self._key, "[REDACTED]")
                raise EndpointError(detail[:2000]) from None
            except (httpx.TransportError, httpx.StreamError) as error:
                last = error
            if attempt < self.retries:
                time.sleep(min(120.0, 2.0 ** attempt) * (0.5 + random.random()))
        detail = str(last)
        if self._key:
            detail = detail.replace(self._key, "[REDACTED]")
        raise EndpointError(f"no answer after {self.retries + 1} attempts: {detail}")


class _Retry(Exception):
    def __init__(self, error: Exception):
        super().__init__(str(error))
        self.error = error
