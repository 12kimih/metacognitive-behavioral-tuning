import json

import httpx

from mbt.client import Client
from mbt.endpoints import Endpoint


def sse(events, named=False):
    lines = []
    for ev in events:
        if named:
            lines.append(f"event: {ev['type']}")
        lines.append("data: " + json.dumps(ev))
        lines.append("")
    if not named:
        lines += ["data: [DONE]", ""]
    return "\n".join(lines).encode()


def client_with(endpoint, handler, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "k")
    c = Client(endpoint, retries=0)
    c._http = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def test_openai_stream_with_reasoning_content(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, content=sse([
            {"choices": [{"delta": {"reasoning_content": "think "}}]},
            {"choices": [{"delta": {"reasoning_content": "more"}}]},
            {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"completion_tokens": 3}},
        ]))

    ep = Endpoint("openai", "https://example.test/v1", "m", api_key_env="TEST_KEY",
                  temperature=1.0, reasoning_effort="max")
    r = client_with(ep, handler, monkeypatch).chat([{"role": "user", "content": "hi"}])
    assert (r.reasoning, r.content, r.finish_reason) == ("think more", "answer", "stop")
    assert seen["url"] == "https://example.test/v1/chat/completions"
    assert seen["body"]["reasoning_effort"] == "max" and seen["body"]["stream"] is True
    assert seen["auth"] == "Bearer k"


def test_a_think_block_in_the_content_is_split_off(monkeypatch):
    def handler(request):
        return httpx.Response(200, content=sse([
            {"choices": [{"delta": {"content": "<think>\nr\n</think>\n\nanswer"},
                          "finish_reason": "length"}]}]))

    ep = Endpoint("openai", "http://x/v1", "m", api_key_env="TEST_KEY")
    r = client_with(ep, handler, monkeypatch).chat([{"role": "user", "content": "hi"}])
    assert (r.reasoning, r.content, r.hit_length_limit) == ("r", "answer", True)


def test_anthropic_stream_with_thinking(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-api-key")
        return httpx.Response(200, content=sse([
            {"type": "message_start", "message": {"usage": {"input_tokens": 5}}},
            {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "t"}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "a"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
             "usage": {"output_tokens": 2}},
            {"type": "message_stop"},
        ], named=True))

    ep = Endpoint("anthropic", "https://example.test", "m", api_key_env="TEST_KEY",
                  auth="x-api-key")
    r = client_with(ep, handler, monkeypatch).chat([{"role": "user", "content": "hi"}])
    assert (r.reasoning, r.content, r.finish_reason) == ("t", "a", "end_turn")
    assert r.usage == {"input_tokens": 5, "output_tokens": 2}
    assert seen["url"] == "https://example.test/v1/messages" and seen["key"] == "k"


def test_a_server_error_cannot_reflect_the_api_key_into_a_record(monkeypatch):
    import pytest
    from mbt.client import EndpointError

    secret = 'test-only-secret-123456789'
    monkeypatch.setenv('TEST_KEY', secret)
    endpoint = Endpoint('openai', 'https://example.test/v1', 'm', api_key_env='TEST_KEY')
    client = Client(endpoint, retries=0)
    client._http = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(401, text='invalid token '+secret)))
    with pytest.raises(EndpointError) as error:
        client.chat([{'role': 'user', 'content': 'Q'}])
    assert secret not in str(error.value) and '[REDACTED]' in str(error.value)


def test_a_stream_error_redacts_a_key_before_truncating_the_message(monkeypatch):
    import pytest
    from mbt.client import EndpointError

    secret = "test-only-secret-for-streaming-123456789"
    monkeypatch.setenv("TEST_KEY", secret)
    data = {"type": "error", "error": {"message": "SENTINEL"}}
    prefix = json.dumps(data).index("SENTINEL")
    data["error"]["message"] = "x" * (1975 - prefix) + secret
    endpoint = Endpoint("anthropic", "https://example.test", "m", api_key_env="TEST_KEY")
    client = Client(endpoint, retries=0)
    client._http = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=sse([data], named=True))))
    with pytest.raises(EndpointError) as error:
        client.chat([{"role": "user", "content": "Q"}])
    assert secret[:12] not in str(error.value)
    assert "[REDACTED]" in str(error.value)
    assert len(str(error.value)) <= 2000
