"""Unit tests for the ingestion layer (cleaner + chunker)."""

from src.finsight.ingestion.chunker import chunk_pages
from src.finsight.ingestion.cleaner import clean_text, find_boilerplate_lines


def test_clean_text_strips_page_numbers_and_normalizes_whitespace():
    raw = "Statutory   Reports\n\t\n42\nReal content line here."
    cleaned = clean_text(raw)
    assert "42" not in cleaned.splitlines()
    assert "Statutory Reports" in cleaned
    assert "Real content line here." in cleaned


def test_clean_text_removes_boilerplate():
    boilerplate = {"Integrated Annual Report 2025-26"}
    raw = "Integrated Annual Report 2025-26\nActual paragraph text."
    cleaned = clean_text(raw, boilerplate)
    assert "Integrated Annual Report 2025-26" not in cleaned
    assert "Actual paragraph text." in cleaned


def test_find_boilerplate_detects_repeated_header():
    pages = [(i, f"Annual Report Header\nUnique body for page {i}.") for i in range(1, 11)]
    boilerplate = find_boilerplate_lines(pages, min_pages=5)
    assert "Annual Report Header" in boilerplate
    assert "Unique body for page 1." not in boilerplate


def test_chunk_pages_produces_chunks_with_metadata():
    pages = [(1, "First paragraph. " * 80), (2, "Second page content. " * 80)]
    chunks = chunk_pages(
        pages,
        company="TCS",
        doc_type="annual_report",
        source_file="tcs.pdf",
        year=2026,
        chunk_size=200,
        chunk_overlap=20,
    )
    assert len(chunks) > 2
    first = chunks[0]
    assert first.metadata.company == "TCS"
    assert first.metadata.source_file == "tcs.pdf"
    assert first.metadata.page in (1, 2)
    assert len(first.metadata.chunk_id) == 16


def test_chunk_ids_are_deterministic():
    pages = [(1, "Deterministic content. " * 50)]
    kwargs = dict(company="TCS", doc_type="annual_report", source_file="tcs.pdf")
    ids_a = [c.metadata.chunk_id for c in chunk_pages(pages, **kwargs)]
    ids_b = [c.metadata.chunk_id for c in chunk_pages(pages, **kwargs)]
    assert ids_a == ids_b
