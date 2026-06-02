"""Unit tests for LLM provider factory."""

import pytest

from src.finsight.llm.factory import get_llm
from src.finsight.llm.gemini import GeminiLLM
from src.finsight.llm.ollama import OllamaLLM


@pytest.fixture(autouse=True)
def _clear_llm_cache():
    get_llm.cache_clear()
    yield
    get_llm.cache_clear()


def test_get_llm_ollama(monkeypatch):
    class _Settings:
        llm_provider = "ollama"
        ollama_model = "llama3.1"
        ollama_base_url = "http://localhost:11434"

    monkeypatch.setattr("src.finsight.llm.factory.get_settings", lambda: _Settings())
    assert isinstance(get_llm(), OllamaLLM)


def test_get_llm_gemini(monkeypatch):
    monkeypatch.setattr("src.finsight.llm.gemini.genai.configure", lambda **_: None)

    class _Settings:
        llm_provider = "gemini"
        gemini_api_key = "key"
        llm_model = "gemini-2.0-flash"

    monkeypatch.setattr("src.finsight.llm.factory.get_settings", lambda: _Settings())
    assert isinstance(get_llm(), GeminiLLM)


def test_get_llm_unknown_raises(monkeypatch):
    class _Settings:
        llm_provider = "unknown"

    monkeypatch.setattr("src.finsight.llm.factory.get_settings", lambda: _Settings())
    with pytest.raises(ValueError):
        get_llm()
