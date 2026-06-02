"""Unit tests for local embeddings (mocked model, no download)."""

import numpy as np
import pytest

from src.finsight.embeddings.local import LocalEmbeddings


class _FakeModel:
    def encode(self, texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False):
        if isinstance(texts, str):
            return np.array([float(len(texts)), 0.5], dtype=float)
        return np.array([[float(len(t)), 0.5] for t in texts], dtype=float)


@pytest.fixture
def embedder(monkeypatch):
    monkeypatch.setattr(
        "src.finsight.embeddings.local.SentenceTransformer",
        lambda name: _FakeModel(),
    )
    return LocalEmbeddings(model_name="fake-model", batch_size=8)


def test_embed_documents_returns_vectors(embedder):
    vectors = embedder.embed_documents(["hello", "world"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 2


def test_embed_query_returns_vector(embedder):
    vector = embedder.embed_query("what are the risks?")
    assert len(vector) == 2


def test_bge_query_uses_prefix(monkeypatch):
    seen: list[str] = []

    class _CaptureModel:
        def encode(self, texts, **kwargs):
            seen.append(texts if isinstance(texts, str) else texts[0])
            return np.array([1.0, 0.0])

    monkeypatch.setattr(
        "src.finsight.embeddings.local.SentenceTransformer",
        lambda name: _CaptureModel(),
    )
    embedder = LocalEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    embedder.embed_query("TCS risks")
    assert seen[0].startswith("Represent this sentence")
