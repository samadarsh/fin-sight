"""Unit tests for prompt templates."""

from src.finsight.models import Source
from src.finsight.prompts.templates import build_context, build_rag_prompt


def test_build_context_includes_source_metadata():
    sources = [
        Source(
            text="Revenue grew 6.3%.",
            company="TCS",
            doc_type="annual_report",
            page=52,
            source_file="annual-report-2025-2026.pdf",
            score=0.9,
        )
    ]
    context = build_context(sources)
    assert "TCS" in context
    assert "page 52" in context
    assert "Revenue grew 6.3%." in context
    assert "[annual-report-2025-2026.pdf p.52]" in context


def test_build_rag_prompt_returns_system_and_user():
    sources = [
        Source(
            text="Cybersecurity is a key risk.",
            company="TCS",
            doc_type="annual_report",
            page=10,
            source_file="annual-report-2025-2026.pdf",
            score=0.8,
        )
    ]
    system, user = build_rag_prompt("What risks did TCS mention?", sources)
    assert "FinSight" in system
    assert "Cybersecurity is a key risk." in user
    assert "What risks did TCS mention?" in user
