"""Unit tests for the query pipeline."""

import pytest

from src.finsight.models import Chunk, ChunkMetadata, QueryResponse
from src.finsight.pipeline.query_pipeline import answer_question
from src.finsight.vectorstore.chroma_store import ChromaStore


class FakeEmbedder:
    batch_size = 10
    batch_delay = 0.0

    def embed_query(self, text: str) -> list[float]:
        if "risk" in text.lower():
            return [0.0, 1.0]
        return [1.0, 0.0]


class FakeLLM:
    def generate(self, system: str, user: str) -> str:
        return "TCS cited cybersecurity risks [annual-report-2025-2026.pdf p.10]."


@pytest.fixture
def store(tmp_path):
    return ChromaStore(persist_dir=tmp_path, collection="query_test")


def _seed_store(store: ChromaStore) -> None:
    chunks = [
        Chunk(
            text="Revenue grew strongly in FY2026.",
            metadata=ChunkMetadata(
                company="TCS",
                doc_type="annual_report",
                page=5,
                source_file="annual-report-2025-2026.pdf",
                chunk_id="rev",
            ),
        ),
        Chunk(
            text="Cybersecurity is a key operational risk.",
            metadata=ChunkMetadata(
                company="TCS",
                doc_type="annual_report",
                page=10,
                source_file="annual-report-2025-2026.pdf",
                chunk_id="risk",
            ),
        ),
    ]
    store.add_chunks(chunks, [[1.0, 0.0], [0.0, 1.0]])


def test_answer_question_empty_store(store):
    result = answer_question(
        "What is revenue?",
        store=store,
        llm=FakeLLM(),
        embedder=FakeEmbedder(),
    )
    assert isinstance(result, QueryResponse)
    assert "don't have any relevant documents" in result.answer
    assert result.sources == []


def test_answer_question_empty_store_skips_llm_init(store, monkeypatch):
    def _boom() -> None:
        raise RuntimeError("LLM should not be initialized for an empty store")

    monkeypatch.setattr("src.finsight.pipeline.query_pipeline.get_llm", _boom)
    result = answer_question("What is revenue?", store=store, embedder=FakeEmbedder())
    assert "don't have any relevant documents" in result.answer
    assert result.sources == []


def test_answer_question_with_context(store):
    _seed_store(store)
    result = answer_question(
        "What risks did TCS mention?",
        store=store,
        llm=FakeLLM(),
        embedder=FakeEmbedder(),
        k=1,
    )
    assert "cybersecurity" in result.answer.lower()
    assert len(result.sources) == 1
    assert result.sources[0].page == 10


def test_answer_question_respects_company_filter(store):
    _seed_store(store)
    store.add_chunks(
        [
            Chunk(
                text="Pipeline expansion continues.",
                metadata=ChunkMetadata(
                    company="IOC",
                    doc_type="annual_report",
                    page=20,
                    source_file="ioc.pdf",
                    chunk_id="ioc1",
                ),
            )
        ],
        [[0.0, 1.0]],
    )
    result = answer_question(
        "What risks did TCS mention?",
        filters={"company": "TCS"},
        store=store,
        llm=FakeLLM(),
        embedder=FakeEmbedder(),
        k=2,
    )
    assert all(source.company == "TCS" for source in result.sources)


def test_answer_question_comparison_mode(store):
    store.add_chunks(
        [
            Chunk(
                text="TCS revenue grew 4.6% in FY 2026.",
                metadata=ChunkMetadata(
                    company="TCS",
                    doc_type="annual_report",
                    page=58,
                    source_file="tcs.pdf",
                    chunk_id="tcs-rev",
                ),
            ),
            Chunk(
                text="IOC revenue from operations was 845513 crore.",
                metadata=ChunkMetadata(
                    company="IOC",
                    doc_type="annual_report",
                    page=113,
                    source_file="ioc.pdf",
                    chunk_id="ioc-rev",
                ),
            ),
        ],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    class ComparisonLLM:
        def generate(self, system: str, user: str) -> str:
            assert "=== TCS ===" in user
            assert "=== IOC ===" in user
            return (
                "## TCS\n4.6% growth\n\n"
                "## IOC\n845513 crore\n\n"
                "## Comparison summary\nDifferent trends."
            )

    result = answer_question(
        "Compare revenue growth",
        compare=True,
        companies=["TCS", "IOC"],
        store=store,
        llm=ComparisonLLM(),
        embedder=FakeEmbedder(),
        k_per_company=1,
    )
    assert "Comparison summary" in result.answer
    assert {source.company for source in result.sources} == {"TCS", "IOC"}
