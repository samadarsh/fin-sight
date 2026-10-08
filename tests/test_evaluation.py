"""Unit tests for evaluation scoring."""

import json

import pytest

from src.finsight.evaluation import (
    EvalItem,
    citation_correct,
    cited_pages,
    facts_found,
    first_hit_rank,
    load_items,
    report_markdown,
    score,
    summarize,
)
from src.finsight.models import Source


def _src(page: int, file: str = "ar-2026.pdf") -> Source:
    return Source(
        text="...", company="TCS", doc_type="annual_report", page=page, source_file=file, score=0.9
    )


ITEM = EvalItem(
    id="rev",
    question="What was revenue?",
    source_file="ar-2026.pdf",
    pages=[58, 59],
    answer_contains=[["2,55,324", "255324"], "crore"],
)


def test_first_hit_rank_matches_file_and_page():
    sources = [_src(10), _src(58, "other.pdf"), _src(59)]
    assert first_hit_rank(ITEM, sources) == 3
    assert first_hit_rank(ITEM, [_src(10)]) is None


def test_file_match_ignores_folder_and_case():
    assert first_hit_rank(ITEM, [_src(58, "data/documents/AR-2026.PDF")]) == 1


def test_cited_pages_dedupes_in_order():
    answer = "A [ar-2026.pdf p.58]. B [x.pdf p.3]. C [ar-2026.pdf p.58]."
    assert cited_pages(answer) == [("ar-2026.pdf", 58), ("x.pdf", 3)]


def test_stray_double_bracket_still_counts():
    assert cited_pages("x [[ar-2026.pdf p.58].") == [("ar-2026.pdf", 58)]


def test_citation_correct_needs_an_expected_page():
    assert citation_correct(ITEM, "Revenue rose [ar-2026.pdf p.59].")
    assert not citation_correct(ITEM, "Revenue rose [ar-2026.pdf p.60].")
    assert not citation_correct(ITEM, "Revenue rose.")


def test_facts_ignore_case_commas_and_rupee_sign():
    assert facts_found(ITEM, "Revenue was ₹2,55,324 Crore.") == 2
    assert facts_found(ITEM, "Revenue was 255324 crore.") == 2
    assert facts_found(ITEM, "Revenue was 2,40,893 crore.") == 1


def test_score_retrieval_only_leaves_answer_metrics_empty():
    result = score(ITEM, [_src(58)], None, 0.42)
    assert result.retrieval_hit and result.first_hit_rank == 1
    assert result.citation_correct is None
    assert result.facts_ok is None


def test_summarize_and_report():
    results = [
        score(ITEM, [_src(58)], "₹2,55,324 crore [ar-2026.pdf p.58]", 1.0),
        score(ITEM, [_src(1), _src(59)], "about 2.5 lakh crore [ar-2026.pdf p.1]", 3.0),
        score(ITEM, [_src(1)], None, 0.5, error="TimeoutError: slow"),
    ]
    summary = summarize(results, k=5)
    assert summary["questions"] == 3
    assert summary["errors"] == 1
    assert summary["retrieval_hit_at_k"] == 100.0
    assert summary["mrr"] == 0.75  # ranks 1 and 2
    assert summary["citation_accuracy"] == 50.0
    assert summary["fact_match"] == 50.0
    table = report_markdown(summary, label="test run")
    assert "| Retrieval hit@5 | 100.0% |" in table
    assert "1 question(s) failed" in table


def test_load_items_skips_comments_and_reports_bad_lines(tmp_path):
    good = tmp_path / "q.jsonl"
    good.write_text(
        "# comment\n\n"
        + json.dumps({"question": "Q?", "source_file": "a.pdf", "pages": ["3"]})
        + "\n",
        encoding="utf-8",
    )
    items = load_items(good)
    assert items[0].id == "q3" and items[0].pages == [3]

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"question": "Q?"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        load_items(bad)


def test_example_file_parses():
    items = load_items(__import__("pathlib").Path("eval/questions.example.jsonl"))
    assert len(items) == 3
