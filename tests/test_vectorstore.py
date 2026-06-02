"""Unit tests for the ChromaDB store.

Uses a temporary directory and small hand-made vectors, so no network or API
key is required.
"""

import pytest

from src.finsight.models import Chunk, ChunkMetadata
from src.finsight.vectorstore.chroma_store import ChromaStore


def _chunk(text: str, chunk_id: str, page: int, company: str = "TCS") -> Chunk:
    return Chunk(
        text=text,
        metadata=ChunkMetadata(
            company=company,
            doc_type="annual_report",
            page=page,
            source_file=f"{company.lower()}.pdf",
            chunk_id=chunk_id,
            year=2026,
        ),
    )


@pytest.fixture
def store(tmp_path):
    return ChromaStore(persist_dir=tmp_path, collection="test_col")


def test_add_and_query_returns_nearest(store):
    chunks = [
        _chunk("revenue grew strongly", "id1", 1),
        _chunk("cyber security risk", "id2", 2),
    ]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    assert store.add_chunks(chunks, embeddings) == 2
    assert store.count() == 2

    results = store.query([0.0, 1.0], k=1)
    assert len(results) == 1
    assert results[0].text == "cyber security risk"
    assert results[0].page == 2
    assert 0.0 <= results[0].score <= 1.0


def test_length_mismatch_raises(store):
    with pytest.raises(ValueError):
        store.add_chunks([_chunk("x", "id1", 1)], [[1.0], [2.0]])


def test_metadata_filter(store):
    chunks = [
        _chunk("tcs content", "id1", 1, company="TCS"),
        _chunk("ioc content", "id2", 1, company="IOC"),
    ]
    store.add_chunks(chunks, [[1.0, 0.0], [0.0, 1.0]])
    results = store.query([1.0, 1.0], k=5, filters={"company": "IOC"})
    assert len(results) == 1
    assert results[0].company == "IOC"


def test_upsert_is_idempotent(store):
    chunk = _chunk("same chunk", "fixed-id", 1)
    store.add_chunks([chunk], [[1.0, 0.0]])
    store.add_chunks([chunk], [[1.0, 0.0]])
    assert store.count() == 1


def test_delete_by_source(store):
    store.add_chunks(
        [_chunk("a", "id1", 1, company="TCS"), _chunk("b", "id2", 1, company="IOC")],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    store.delete_by_source("ioc.pdf")
    assert store.count() == 1


def test_list_documents(store):
    store.add_chunks(
        [_chunk("a", "id1", 1), _chunk("b", "id2", 2)],
        [[1.0, 0.0], [0.5, 0.5]],
    )
    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0]["chunks"] == 2
    assert docs[0]["company"] == "TCS"


def test_existing_chunk_ids(store):
    store.add_chunks([_chunk("a", "id1", 1)], [[1.0, 0.0]])
    assert store.existing_chunk_ids("tcs.pdf") == {"id1"}
    assert store.existing_chunk_ids("other.pdf") == set()
