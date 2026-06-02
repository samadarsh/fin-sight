"""Unit tests for the ingest pipeline."""

from pathlib import Path

import pytest

from src.finsight.pipeline.ingest_pipeline import ingest_document
from src.finsight.vectorstore.chroma_store import ChromaStore


class FakeEmbedder:
    batch_size = 50
    batch_delay = 0.0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 0.0]


class FlakyEmbedder(FakeEmbedder):
    """Fails after the first successful batch to simulate a rate limit."""

    batch_size = 5
    batch_delay = 0.0

    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("Gemini embedding failed after 3 attempts")
        return super().embed_documents(texts)


@pytest.fixture
def store(tmp_path):
    return ChromaStore(persist_dir=tmp_path, collection="ingest_test")


def test_ingest_document_end_to_end(store, tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")  # not a real PDF; loader will fail

    # Use a real PDF from the repo if present, otherwise skip.
    real_pdf = Path("data/documents/annual-report-2025-2026.pdf")
    if not real_pdf.exists():
        pytest.skip("sample PDF not available")

    result = ingest_document(
        real_pdf,
        company="TCS",
        doc_type="annual_report",
        year=2026,
        embedder=FakeEmbedder(),
        store=store,
    )
    assert result.source_file == "annual-report-2025-2026.pdf"
    assert result.company == "TCS"
    assert result.pages_processed > 0
    assert result.chunks_added > 0
    assert result.total_chunks_in_store == result.chunks_added

    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0]["company"] == "TCS"


def test_ingest_document_replace_existing(store, tmp_path):
    real_pdf = Path("data/documents/annual-report-2025-2026.pdf")
    if not real_pdf.exists():
        pytest.skip("sample PDF not available")

    kwargs = dict(
        company="TCS",
        doc_type="annual_report",
        year=2026,
        embedder=FakeEmbedder(),
        store=store,
        replace_existing=True,
    )
    first = ingest_document(real_pdf, **kwargs)
    second = ingest_document(real_pdf, **kwargs)

    assert second.chunks_added == first.chunks_added
    assert store.count() == first.chunks_added
    assert len(store.list_documents()) == 1


def test_ingest_document_missing_file(store):
    with pytest.raises(FileNotFoundError):
        ingest_document(
            "missing.pdf",
            company="TCS",
            doc_type="annual_report",
            embedder=FakeEmbedder(),
            store=store,
        )


def test_ingest_document_two_companies(store, tmp_path):
    tcs = Path("data/documents/annual-report-2025-2026.pdf")
    ioc = Path("data/documents/SingleAnnualReport202425.pdf")
    if not tcs.exists() or not ioc.exists():
        pytest.skip("sample PDFs not available")

    embedder = FakeEmbedder()
    ingest_document(
        tcs,
        company="TCS",
        doc_type="annual_report",
        year=2026,
        embedder=embedder,
        store=store,
    )
    ingest_document(
        ioc,
        company="IOC",
        doc_type="annual_report",
        year=2025,
        embedder=embedder,
        store=store,
    )

    assert store.count() > 0
    assert len(store.list_documents()) == 2

    tcs_hits = store.query([1000.0, 0.0], k=3, filters={"company": "TCS"})
    ioc_hits = store.query([1000.0, 0.0], k=3, filters={"company": "IOC"})
    assert all(hit.company == "TCS" for hit in tcs_hits)
    assert all(hit.company == "IOC" for hit in ioc_hits)


def test_ingest_document_resume_after_partial(store):
    real_pdf = Path("data/documents/annual-report-2025-2026.pdf")
    if not real_pdf.exists():
        pytest.skip("sample PDF not available")

    partial = ingest_document(
        real_pdf,
        company="TCS",
        doc_type="annual_report",
        year=2026,
        embedder=FlakyEmbedder(),
        store=store,
    )
    assert partial.partial is True
    assert partial.chunks_added == 5
    assert store.count() == 5

    completed = ingest_document(
        real_pdf,
        company="TCS",
        doc_type="annual_report",
        year=2026,
        embedder=FakeEmbedder(),
        store=store,
        resume=True,
    )
    assert completed.partial is False
    assert completed.chunks_skipped == 5
    assert completed.chunks_added > 0
    assert store.count() == partial.chunks_added + completed.chunks_added

