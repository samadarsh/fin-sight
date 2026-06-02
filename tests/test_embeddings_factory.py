"""Unit tests for embedding provider factory."""

import pytest

from src.finsight.embeddings.factory import get_embedder
from src.finsight.embeddings.gemini import GeminiEmbeddings
from src.finsight.embeddings.local import LocalEmbeddings


@pytest.fixture(autouse=True)
def _clear_embedder_cache():
    get_embedder.cache_clear()
    yield
    get_embedder.cache_clear()


def test_get_embedder_local(monkeypatch):
    monkeypatch.setattr(
        "src.finsight.embeddings.local.SentenceTransformer",
        lambda name: object(),
    )

    class _Settings:
        embedding_provider = "local"
        local_embedding_model = "fake"
        local_embedding_batch_size = 32

    monkeypatch.setattr("src.finsight.embeddings.factory.get_settings", lambda: _Settings())
    assert isinstance(get_embedder(), LocalEmbeddings)


def test_get_embedder_gemini(monkeypatch):
    monkeypatch.setattr("src.finsight.embeddings.gemini.genai.configure", lambda **_: None)

    class _Settings:
        embedding_provider = "gemini"
        gemini_api_key = "key"
        embedding_model = "gemini-embedding-001"
        embedding_batch_size = 10
        embedding_batch_delay = 0.0

    monkeypatch.setattr("src.finsight.embeddings.factory.get_settings", lambda: _Settings())
    assert isinstance(get_embedder(), GeminiEmbeddings)


def test_get_embedder_unknown_raises(monkeypatch):
    class _Settings:
        embedding_provider = "unknown"

    monkeypatch.setattr("src.finsight.embeddings.factory.get_settings", lambda: _Settings())
    with pytest.raises(ValueError):
        get_embedder()
