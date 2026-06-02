"""FinSight FastAPI application."""

from functools import partial
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from api.deps import get_documents_dir, get_store
from api.schemas import (
    DeleteDocumentResponse,
    DocumentSummary,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryAPIResponse,
    QueryRequest,
    UploadResponse,
)
from config.settings import get_settings
from src.finsight.errors import GeminiQuotaError
from src.finsight.pipeline.ingest_pipeline import ingest_document
from src.finsight.pipeline.query_pipeline import answer_question, preview_query_mode
from src.finsight.vectorstore.chroma_store import ChromaStore

app = FastAPI(
    title="FinSight API",
    description="RAG API for financial documents — upload, ingest, query.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    documents_dir: Path = Depends(get_documents_dir),
) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    documents_dir.mkdir(parents=True, exist_ok=True)
    dest = documents_dir / Path(file.filename).name
    size_bytes = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            out.write(chunk)
            size_bytes += len(chunk)

    return UploadResponse(
        source_file=dest.name,
        path=str(dest),
        size_bytes=size_bytes,
    )


def _run_ingest(req: IngestRequest, store: ChromaStore) -> IngestResponse:
    settings = get_settings()
    pdf_path = settings.documents_path / req.source_file
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {req.source_file}")

    try:
        result = ingest_document(
            pdf_path,
            company=req.company,
            doc_type=req.doc_type,
            year=req.year,
            quarter=req.quarter,
            replace_existing=req.replace_existing,
            resume=req.resume,
            store=store,
        )
    except GeminiQuotaError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IngestResponse(**result.model_dump())


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, store: ChromaStore = Depends(get_store)) -> IngestResponse:
    return await run_in_threadpool(partial(_run_ingest, req, store))


def _run_query(req: QueryRequest, store: ChromaStore) -> QueryAPIResponse:
    filters = {"company": req.company} if req.company else None
    compare = req.compare
    if req.companies and len(req.companies) >= 2:
        compare = True

    mode = preview_query_mode(
        req.question,
        compare=compare,
        companies=req.companies,
        filters=filters,
        store=store,
    )

    try:
        result = answer_question(
            req.question,
            k=req.k,
            k_per_company=req.k_per_company,
            filters=filters,
            compare=compare,
            companies=req.companies,
            store=store,
        )
    except GeminiQuotaError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return QueryAPIResponse(mode=mode, **result.model_dump())


@app.post("/query", response_model=QueryAPIResponse)
async def query(req: QueryRequest, store: ChromaStore = Depends(get_store)) -> QueryAPIResponse:
    return await run_in_threadpool(partial(_run_query, req, store))


@app.get("/documents", response_model=list[DocumentSummary])
async def list_documents(store: ChromaStore = Depends(get_store)) -> list[DocumentSummary]:
    docs = await run_in_threadpool(store.list_documents)
    return [DocumentSummary(**doc) for doc in docs]


@app.delete("/documents/{source_file}", response_model=DeleteDocumentResponse)
async def delete_document(
    source_file: str, store: ChromaStore = Depends(get_store)
) -> DeleteDocumentResponse:
    def _delete() -> DeleteDocumentResponse:
        existing = {doc["source_file"] for doc in store.list_documents()}
        if source_file not in existing:
            raise HTTPException(status_code=404, detail=f"Document not found: {source_file}")

        store.delete_by_source(source_file)
        return DeleteDocumentResponse(
            source_file=source_file,
            deleted=True,
            remaining_chunks=store.count(),
        )

    return await run_in_threadpool(_delete)
