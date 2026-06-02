"""Tests for comparison query helpers and retrieval."""

import pytest

from src.finsight.models import Chunk, ChunkMetadata, Source
from src.finsight.prompts.templates import build_comparison_context, build_comparison_prompt
from src.finsight.retrieval.comparison import (
    detect_companies_in_question,
    is_comparison_query,
    resolve_comparison_companies,
)
from src.finsight.retrieval.retriever import retrieve_per_company
from src.finsight.vectorstore.chroma_store import ChromaStore


class FakeEmbedder:
    batch_size = 64
    batch_delay = 0.0

    def embed_query(self, text: str) -> list[float]:
        if "ioc" in text.lower():
            return [0.0, 1.0]
        return [1.0, 0.0]


@pytest.fixture
def store(tmp_path):
    return ChromaStore(persist_dir=tmp_path, collection="comparison_test")


def _chunk(text: str, chunk_id: str, company: str, page: int) -> Chunk:
    return Chunk(
        text=text,
        metadata=ChunkMetadata(
            company=company,
            doc_type="annual_report",
            page=page,
            source_file=f"{company.lower()}.pdf",
            chunk_id=chunk_id,
        ),
    )


def test_detect_companies_in_question():
    found = detect_companies_in_question("Compare TCS and IOC revenue", ["TCS", "IOC"])
    assert found == ["TCS", "IOC"]


def test_is_comparison_query_keywords():
    assert is_comparison_query("Compare revenue growth", []) is True
    assert is_comparison_query(
        "Compare revenue growth mentioned by TCS and IOC",
        ["TCS", "IOC"],
    ) is True
    assert is_comparison_query("What is revenue?", []) is False


def test_preview_query_mode_auto_detects(store):
    store.add_chunks(
        [
            _chunk("tcs revenue grew 4.6%", "t1", "TCS", 1),
            _chunk("ioc revenue declined", "i1", "IOC", 2),
        ],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    from src.finsight.pipeline.query_pipeline import preview_query_mode

    mode = preview_query_mode(
        "Compare revenue growth mentioned by TCS and IOC",
        store=store,
    )
    assert mode.startswith("comparison (")
    assert "TCS" in mode and "IOC" in mode


def test_resolve_comparison_companies_defaults_to_two_indexed(store):
    store.add_chunks(
        [
            _chunk("tcs revenue grew 4.6%", "t1", "TCS", 1),
            _chunk("ioc revenue declined", "i1", "IOC", 2),
        ],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    resolved = resolve_comparison_companies("Compare revenue growth", store)
    assert set(resolved) == {"TCS", "IOC"}


def test_fy_label():
    from src.finsight.retrieval.retriever import _fy_label

    assert _fy_label(2025) == "2024-25"
    assert _fy_label(2026) == "2025-26"


def test_retrieve_per_company_returns_both_companies(store):
    store.add_chunks(
        [
            _chunk("tcs revenue grew 4.6%", "t1", "TCS", 1),
            _chunk("tcs margins improved", "t2", "TCS", 2),
            _chunk("ioc revenue from operations 845513 crore declined", "i1", "IOC", 10),
            _chunk("ioc subsidiary list", "i2", "IOC", 11),
        ],
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
    )
    sources = retrieve_per_company(
        "Compare revenue growth",
        ["TCS", "IOC"],
        k_per_company=2,
        embedder=FakeEmbedder(),
        store=store,
    )
    companies = {source.company for source in sources}
    assert companies == {"TCS", "IOC"}
    assert len(sources) == 4
    ioc_pages = {source.page for source in sources if source.company == "IOC"}
    assert 10 in ioc_pages


def test_build_comparison_context_groups_by_company():
    sources = [
        Source(
            text="tcs",
            company="TCS",
            doc_type="annual_report",
            page=1,
            source_file="tcs.pdf",
            score=0.9,
        ),
        Source(
            text="ioc",
            company="IOC",
            doc_type="annual_report",
            page=2,
            source_file="ioc.pdf",
            score=0.8,
        ),
    ]
    context = build_comparison_context(sources, ["TCS", "IOC"])
    assert "=== TCS ===" in context
    assert "=== IOC ===" in context


def test_build_comparison_prompt_includes_companies():
    sources = [
        Source(
            text="tcs",
            company="TCS",
            doc_type="annual_report",
            page=1,
            source_file="tcs.pdf",
            score=0.9,
        ),
    ]
    system, user = build_comparison_prompt("Compare revenue", sources, ["TCS", "IOC"])
    assert "COMPARISON" in system
    assert "TCS, IOC" in user
