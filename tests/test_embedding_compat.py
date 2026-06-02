"""Tests for embedding fingerprinting and Chroma compatibility guards."""

import pytest

from src.finsight.embeddings.fingerprint import EmbeddingFingerprint, fingerprint_embedder
from src.finsight.errors import EmbeddingMismatchError
from src.finsight.models import Chunk, ChunkMetadata
from src.finsight.vectorstore.chroma_store import ChromaStore


class FakeEmbedderTwoD:
    """2-dimensional fake embedder for fast tests."""

    model_name = "fake-2d"

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeEmbedderThreeD:
    model_name = "fake-3d"

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


@pytest.fixture
def store(tmp_path):
    return ChromaStore(persist_dir=tmp_path, collection="embed_compat")


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(
        text="sample",
        metadata=ChunkMetadata(
            company="TCS",
            doc_type="annual_report",
            page=1,
            source_file="tcs.pdf",
            chunk_id=chunk_id,
        ),
    )


def test_fingerprint_embedder_dimension():
    fp = fingerprint_embedder(FakeEmbedderTwoD())
    assert fp.dimension == 2
    assert fp.provider  # defaults from settings


def test_empty_store_stamps_metadata(store):
    embedder = FakeEmbedderTwoD()
    store.ensure_embedding_compatible(embedder, write=True)
    stored = store.stored_embedding_fingerprint()
    assert stored is not None
    assert stored.dimension == 2


def test_mismatch_raises_when_dimensions_differ(store):
    store.add_chunks([_chunk("a")], [[1.0, 0.0]])
    with pytest.raises(EmbeddingMismatchError):
        store.ensure_embedding_compatible(FakeEmbedderThreeD())


def test_query_dimension_guard(store):
    store.add_chunks([_chunk("a")], [[1.0, 0.0]])
    store._stamp_embedding_metadata(
        EmbeddingFingerprint(provider="local", model="fake-2d", dimension=2)
    )
    with pytest.raises(EmbeddingMismatchError):
        store.query([1.0, 0.0, 0.0], k=1)
