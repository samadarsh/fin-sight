"""Tests for citation formatting and post-processing."""

from src.finsight.models import Source
from src.finsight.prompts.citations import (
    format_citation,
    has_inline_citations,
    normalize_answer_citations,
)
from src.finsight.prompts.templates import build_context


def _source(page: int = 58) -> Source:
    return Source(
        text="Revenue grew 4.6%.",
        company="TCS",
        doc_type="annual_report",
        page=page,
        source_file="annual-report-2025-2026.pdf",
        score=0.9,
    )


def test_format_citation():
    assert format_citation("annual-report-2025-2026.pdf", 58) == (
        "[annual-report-2025-2026.pdf p.58]"
    )


def test_build_context_includes_citation_token():
    context = build_context([_source()])
    assert "Citation token" in context
    assert "[annual-report-2025-2026.pdf p.58]" in context


def test_normalize_replaces_source_labels():
    sources = [_source()]
    answer = normalize_answer_citations("Growth was 4.6% per Source 1.", sources)
    assert "[annual-report-2025-2026.pdf p.58]" in answer


def test_normalize_appends_citations_when_missing():
    sources = [_source(), _source(page=59)]
    answer = normalize_answer_citations("Revenue grew without inline cites.", sources)
    assert has_inline_citations(answer) or answer.strip().endswith(
        "[annual-report-2025-2026.pdf p.59]"
    )
    assert "Citations:" in answer


def test_normalize_preserves_existing_inline_citations():
    sources = [_source()]
    original = "Growth was 4.6% [annual-report-2025-2026.pdf p.58]."
    assert normalize_answer_citations(original, sources) == original
