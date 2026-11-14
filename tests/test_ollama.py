import json

import httpx
import pytest

from radiology_rag.config import Settings
from radiology_rag.generation.ollama import OllamaCloudClient, OllamaError


def test_local_chat_never_forwards_cloud_key_and_bounds_memory():
    def handler(request):
        assert request.url == "http://127.0.0.1:11434/api/chat"
        assert "Authorization" not in request.headers
        body = json.loads(request.content)
        assert body["keep_alive"] == 0
        assert body["options"]["num_ctx"] == 8192 and body["options"]["num_thread"] == 2
        assert body["options"]["num_batch"] == 128
        return httpx.Response(200, json={"done": True, "message": {"content": "OK"}})

    settings = Settings(base_url="http://127.0.0.1:11434", model="unit-local-model",
                        api_key="retained-cloud-key", no_cloud=True)
    with OllamaCloudClient(settings, transport=httpx.MockTransport(handler)) as client:
        assert client.chat([{"role": "user", "content": "test"}]).content == "OK"


def test_local_models_exclude_remote_entries():
    settings = Settings(base_url="http://127.0.0.1:11434", no_cloud=True)
    response = {"models": [{"name": "local-model"}, {"name": "remote:cloud"},
                           {"name": "remote", "remote_host": "https://ollama.com"}]}
    with OllamaCloudClient(settings, transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json=response)
    )) as client:
        assert client.list_models() == ["local-model"]


def configured(**overrides):
    return Settings(**{"api_key": "unit-test-secret", "model": "unit-test-model", **overrides})


def test_chat_contract_usage_and_private_metrics(caplog):
    def handler(request):
        assert request.url == "https://ollama.com/api/chat"
        assert request.headers["Authorization"] == "Bearer unit-test-secret"
        payload = json.loads(request.content)
        assert payload["stream"] is False
        assert payload["options"] == {"temperature": 0.1, "num_predict": 1024}
        return httpx.Response(200, json={
            "model": "unit-test-model", "done": True, "done_reason": "stop",
            "message": {"content": "private-answer"}, "prompt_eval_count": 8,
            "eval_count": 3, "total_duration": 123456,
        })

    with caplog.at_level("INFO"), OllamaCloudClient(
        configured(), transport=httpx.MockTransport(handler)
    ) as client:
        result = client.chat([{"role": "user", "content": "private-prompt"}])
    assert result.content == "private-answer"
    assert result.prompt_tokens == 8 and result.completion_tokens == 3
    assert result.server_total_duration_ns == 123456 and result.latency_seconds >= 0
    assert "content" not in result.metrics()
    for private in ("unit-test-secret", "private-answer", "private-prompt"):
        assert private not in caplog.text


def test_missing_usage_stays_none():
    with OllamaCloudClient(configured(), transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json={"done": True, "message": {"content": "OK"}})
    )) as client:
        result = client.chat([])
    assert result.prompt_tokens is None and result.completion_tokens is None


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_terminal_errors_do_not_retry_or_leak_body(code):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(code, json={"error": "sensitive-body"})

    with OllamaCloudClient(configured(), transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OllamaError) as error:
            client.chat([])
    assert len(calls) == 1
    assert "sensitive-body" not in str(error.value)


def test_rate_limit_then_success_respects_retry_after():
    calls, waits = [], []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(200, json={"done": True, "message": {"content": "OK"}})

    with OllamaCloudClient(
        configured(), transport=httpx.MockTransport(handler), sleep=waits.append
    ) as client:
        result = client.chat([])
    assert result.attempts == 2 and waits == [2]


def test_retry_after_above_cap_stops_without_early_retry():
    waits = []
    with OllamaCloudClient(configured(), sleep=waits.append, transport=httpx.MockTransport(
        lambda _: httpx.Response(429, headers={"Retry-After": "3600"})
    )) as client:
        with pytest.raises(OllamaError, match="Retry-After"):
            client.chat([])
    assert waits == []


def test_transport_timeout_has_bounded_retries():
    calls, waits = [], []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("private transport details")

    with OllamaCloudClient(
        configured(), transport=httpx.MockTransport(handler), sleep=waits.append
    ) as client:
        with pytest.raises(OllamaError) as error:
            client.chat([])
    assert len(calls) == 3 and len(waits) == 2
    assert "private transport details" not in str(error.value)


@pytest.mark.parametrize("payload", [
    {"error": "sensitive"}, {"done": False}, [], {"done": True, "message": None},
])
def test_api_error_or_incomplete_response_is_not_a_success(payload):
    with OllamaCloudClient(configured(), transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json=payload)
    )) as client:
        with pytest.raises(OllamaError):
            client.chat([])


def test_missing_key_fails_before_network():
    def handler(_):
        pytest.fail("Network should not be called")

    with OllamaCloudClient(
        Settings(model="test"), transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(ValueError, match="OLLAMA_API_KEY"):
            client.chat([])


def test_models_uses_public_tags_without_assuming_generation_model():
    def handler(request):
        assert request.method == "GET" and request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "listed-model"}]})

    with OllamaCloudClient(Settings(), transport=httpx.MockTransport(handler)) as client:
        assert client.list_models() == ["listed-model"]
