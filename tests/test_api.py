"""API integration tests."""

import io

import pytest
from fastapi.testclient import TestClient

from api.deps import get_documents_dir, get_store
from api.main import app
from config.settings import get_settings
from src.finsight.models import QueryResponse, Source
from src.finsight.vectorstore.chroma_store import ChromaStore


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    chroma_dir = tmp_path / "chroma"
    store = ChromaStore(persist_dir=chroma_dir, collection="api_test")

    monkeypatch.setenv("DOCUMENTS_DIR", str(docs_dir))
    monkeypatch.setenv("CHROMA_DIR", str(chroma_dir))
    get_settings.cache_clear()

    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_documents_dir] = lambda: docs_dir

    yield TestClient(app), store, docs_dir

    app.dependency_overrides.clear()
    get_settings.cache_clear()


def test_health(api_client):
    client, _, _ = api_client
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_pdf(api_client):
    client, _, docs_dir = api_client
    pdf_bytes = b"%PDF-1.4 minimal"
    response = client.post(
        "/upload",
        files={"file": ("report.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["source_file"] == "report.pdf"
    assert (docs_dir / "report.pdf").exists()


def test_upload_rejects_non_pdf(api_client):
    client, _, _ = api_client
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 400


def test_upload_rejects_invalid_pdf_magic(api_client):
    client, _, _ = api_client
    response = client.post(
        "/upload",
        files={"file": ("fake.pdf", io.BytesIO(b"NOTPDF-content"), "application/pdf")},
    )
    assert response.status_code == 400
    assert "magic" in response.json()["detail"].lower()


def test_list_documents_empty(api_client):
    client, _, _ = api_client
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json() == []


def test_ingest_not_found(api_client):
    client, _, _ = api_client
    response = client.post(
        "/ingest",
        json={
            "source_file": "missing.pdf",
            "company": "TCS",
            "doc_type": "annual_report",
        },
    )
    assert response.status_code == 404


def test_query_empty_store(api_client, monkeypatch):
    client, _, _ = api_client

    def fake_preview(*args, **kwargs):
        return "standard"

    monkeypatch.setattr("api.main.preview_query_mode", fake_preview)

    response = client.post("/query", json={"question": "What is revenue?"})
    assert response.status_code == 200
    data = response.json()
    assert "don't have any relevant documents" in data["answer"]
    assert data["mode"] == "standard"


def test_query_with_mocked_pipeline(api_client, monkeypatch):
    client, store, _ = api_client

    def fake_answer(*args, **kwargs):
        return QueryResponse(
            answer="TCS revenue grew 4.6%",
            sources=[
                Source(
                    text="revenue grew",
                    company="TCS",
                    doc_type="annual_report",
                    page=58,
                    source_file="tcs.pdf",
                    score=0.9,
                )
            ],
        )

    monkeypatch.setattr("api.main.answer_question", fake_answer)
    monkeypatch.setattr("api.main.preview_query_mode", lambda *a, **kw: "standard (filter: TCS)")

    response = client.post(
        "/query",
        json={"question": "Revenue growth?", "company": "TCS"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "4.6%" in data["answer"]
    assert len(data["sources"]) == 1
    assert data["mode"] == "standard (filter: TCS)"


def test_delete_document(api_client):
    client, store, _ = api_client
    from src.finsight.models import Chunk, ChunkMetadata

    store.add_chunks(
        [
            Chunk(
                text="sample",
                metadata=ChunkMetadata(
                    company="TCS",
                    doc_type="annual_report",
                    page=1,
                    source_file="tcs.pdf",
                    chunk_id="abc",
                ),
            )
        ],
        [[1.0, 0.0]],
    )

    response = client.delete("/documents/tcs.pdf")
    assert response.status_code == 200
    assert response.json()["deleted"] is True
    assert store.count() == 0


def test_delete_document_not_found(api_client):
    client, _, _ = api_client
    response = client.delete("/documents/nope.pdf")
    assert response.status_code == 404
