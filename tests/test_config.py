import pytest

from radiology_rag.config import Settings


def test_env_overrides_explicit_dotenv_without_exposing_secret(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("OLLAMA_MODEL=from-file\nOLLAMA_API_KEY=test-secret\n", encoding="utf-8")
    monkeypatch.setenv("OLLAMA_MODEL", "from-process")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    settings = Settings.from_env(path)
    assert settings.model == "from-process" and settings.api_key == "test-secret"
    assert "test-secret" not in repr(settings)


@pytest.mark.parametrize("url", [
    "http://ollama.com", "https://user:secret@ollama.com", "https://ollama.com/api",
    "https://ollama.com?key=secret", "https://ollama.com/#secret",
])
def test_cloud_origin_rejects_insecure_or_ambiguous_urls(url):
    with pytest.raises(ValueError):
        Settings(base_url=url)


@pytest.mark.parametrize("kwargs", [
    {"timeout_seconds": -1}, {"max_retries": 6}, {"temperature": float("nan")},
    {"num_predict": 0}, {"max_backoff_seconds": 90},
])
def test_configuration_rejects_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_local_origin_needs_no_key_and_no_cloud_rejects_remote_origins():
    local = Settings(base_url="http://127.0.0.1:11434", model="qwen3:4b", no_cloud=True)
    assert local.is_local and not local.api_key
    with pytest.raises(ValueError, match="bloquea"):
        Settings(no_cloud=True)
    with pytest.raises(ValueError, match="Cloud"):
        Settings(base_url="http://localhost:11434", model="model:cloud", no_cloud=True)
    with pytest.raises(ValueError):
        Settings(base_url="http://192.168.1.12:11434")
