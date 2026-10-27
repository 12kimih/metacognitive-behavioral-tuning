"""Where a model call goes, and the presets for common providers.

An `Endpoint` is everything the client needs: the wire format, the base URL, the model id, the
environment variable that holds the key, and the sampling settings. A preset fills in the
provider's base URL and key variable; the model id and every setting can be overridden on the
command line (`--model`, `--base-url`, `--api-key-env`, `--temperature`, ...).

The defaults reproduce the settings of the paper: DeepSeek-V4.1-Flash with reasoning effort
`max`, temperature 1.0 and top_p 0.95. Model ids change over time; check the provider's model
list when a preset's default id is refused.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Literal

Api = Literal["openai", "anthropic"]


@dataclass(frozen=True)
class Endpoint:
    api: Api
    base_url: str
    model: str
    api_key_env: str | None = None
    auth: Literal["bearer", "x-api-key"] = "bearer"
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_tokens_field: str = "max_tokens"
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)

    def replace(self, **changes: Any) -> "Endpoint":
        return dataclasses.replace(self, **{k: v for k, v in changes.items() if v is not None})


#: The paper's editor and teacher settings, applied on top of a preset by `resolve`.
PAPER_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "reasoning_effort": "max"}

PRESETS: dict[str, Endpoint] = {
    # DeepSeek official API (OpenAI format). `reasoning_effort` accepts low ... max.
    "deepseek": Endpoint("openai", "https://api.deepseek.com", "deepseek-flash",
                         api_key_env="DEEPSEEK_API_KEY"),
    # DeepSeek official API (Anthropic format), also what Claude Code talks to for MBT-A.
    "deepseek-anthropic": Endpoint("anthropic", "https://api.deepseek.com/anthropic",
                                   "deepseek-flash", api_key_env="DEEPSEEK_API_KEY"),
    "openai": Endpoint("openai", "https://api.openai.com/v1", "gpt-5.6-luna",
                       api_key_env="OPENAI_API_KEY", max_tokens_field="max_completion_tokens"),
    "anthropic": Endpoint("anthropic", "https://api.anthropic.com", "claude-sonnet-5",
                          api_key_env="ANTHROPIC_API_KEY", auth="x-api-key"),
    "openrouter": Endpoint("openai", "https://openrouter.ai/api/v1", "deepseek/deepseek-flash",
                           api_key_env="OPENROUTER_API_KEY"),
    "openrouter-anthropic": Endpoint("anthropic", "https://openrouter.ai/api",
                                     "deepseek/deepseek-flash", api_key_env="OPENROUTER_API_KEY"),
    # Qwen Cloud (Alibaba Cloud Model Studio, DashScope), international region. For the China
    # region use https://dashscope.aliyuncs.com/... instead.
    "qwen": Endpoint("openai", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                     "qwen-plus", api_key_env="DASHSCOPE_API_KEY",
                     extra_body={"enable_thinking": True}),
    "qwen-anthropic": Endpoint("anthropic", "https://dashscope-intl.aliyuncs.com/apps/anthropic",
                               "qwen-plus", api_key_env="DASHSCOPE_API_KEY"),
    # A local OpenAI-compatible server such as vLLM or SGLang serving DeepSeek-V4.1-Flash. The
    # chat template switch turns thinking on for that model under vLLM.
    "vllm": Endpoint("openai", "http://localhost:8000/v1", "deepseek-ai/DeepSeek-V4.1-Flash",
                     api_key_env=None,
                     extra_body={"top_k": -1,
                                 "chat_template_kwargs": {"thinking_mode": "thinking"}}),
}


def resolve(preset: str, *, model: str | None = None, base_url: str | None = None,
            api: str | None = None, api_key_env: str | None = None,
            temperature: float | None = None, top_p: float | None = None,
            max_tokens: int | None = None, reasoning_effort: str | None = None,
            extra_body: str | None = None, paper_sampling: bool = True) -> Endpoint:
    """A preset with the command-line overrides applied.

    `paper_sampling` puts the paper's temperature, top_p and reasoning effort in first, so an
    explicit override still wins. `extra_body` is a JSON object merged into every request body.
    """
    if preset not in PRESETS:
        raise SystemExit(f"unknown endpoint preset {preset!r}; one of {', '.join(PRESETS)}")
    ep = PRESETS[preset]
    if paper_sampling:
        ep = ep.replace(**PAPER_SAMPLING)
    ep = ep.replace(model=model, base_url=base_url, api=api, temperature=temperature,
                    top_p=top_p, max_tokens=max_tokens, reasoning_effort=reasoning_effort)
    if api_key_env is not None:
        ep = dataclasses.replace(ep, api_key_env=api_key_env or None)
    if extra_body:
        ep = dataclasses.replace(ep, extra_body={**ep.extra_body, **json.loads(extra_body)})
    return ep
