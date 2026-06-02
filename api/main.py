"""FinSight FastAPI application."""

from functools import partial
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
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
from src.finsight.errors import EmbeddingMismatchError, GeminiQuotaError
from src.finsight.pipeline.ingest_pipeline import ingest_document
from src.finsight.pipeline.query_pipeline import answer_question, preview_query_mode
from src.finsight.vectorstore.chroma_store import ChromaStore

settings = get_settings()

app = FastAPI(
    title="FinSight API",
    description="RAG API for financial documents — upload, ingest, query.",
    version="0.2.0",
)

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

_PDF_MAGIC = b"%PDF"
_CHUNK_SIZE = 1024 * 1024


async def _save_upload_stream(
    file: UploadFile,
    dest: Path,
    *,
    max_bytes: int,
    initial: bytes,
) -> int:
    """Stream upload to disk with size limit; ``initial`` is the already-read header."""
    size = len(initial)
    with dest.open("wb") as out:
        out.write(initial)
        while chunk := await file.read(_CHUNK_SIZE):
            size += len(chunk)
            if size > max_bytes:
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds maximum upload size of {max_bytes // (1024 * 1024)} MB",
                )
            out.write(chunk)
    return size


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@app.post("/upload", response_model=UploadResponse)
@limiter.limit(settings.rate_limit)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    documents_dir: Path = Depends(get_documents_dir),
) -> UploadResponse:
    _ = request
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    header = await file.read(1024)
    if not header.startswith(_PDF_MAGIC):
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF: file must start with %PDF magic bytes",
        )

    documents_dir.mkdir(parents=True, exist_ok=True)
    dest = documents_dir / Path(file.filename).name
    max_bytes = get_settings().max_upload_bytes
    size_bytes = await _save_upload_stream(file, dest, max_bytes=max_bytes, initial=header)

    return UploadResponse(
        source_file=dest.name,
        path=str(dest),
        size_bytes=size_bytes,
    )


def _run_ingest(req: IngestRequest, store: ChromaStore) -> IngestResponse:
    cfg = get_settings()
    pdf_path = cfg.documents_path / req.source_file
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
    except EmbeddingMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except GeminiQuotaError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IngestResponse(**result.model_dump())


@app.post("/ingest", response_model=IngestResponse)
@limiter.limit(settings.rate_limit)
async def ingest(
    request: Request,
    req: IngestRequest,
    store: ChromaStore = Depends(get_store),
) -> IngestResponse:
    _ = request
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
    except EmbeddingMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except GeminiQuotaError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return QueryAPIResponse(mode=mode, **result.model_dump())


@app.post("/query", response_model=QueryAPIResponse)
@limiter.limit(settings.rate_limit)
async def query(
    request: Request,
    req: QueryRequest,
    store: ChromaStore = Depends(get_store),
) -> QueryAPIResponse:
    _ = request
    return await run_in_threadpool(partial(_run_query, req, store))


@app.get("/documents", response_model=list[DocumentSummary])
@limiter.limit(settings.rate_limit)
async def list_documents(
    request: Request,
    store: ChromaStore = Depends(get_store),
) -> list[DocumentSummary]:
    _ = request
    docs = await run_in_threadpool(store.list_documents)
    return [DocumentSummary(**doc) for doc in docs]


@app.delete("/documents/{source_file}", response_model=DeleteDocumentResponse)
@limiter.limit(settings.rate_limit)
async def delete_document(
    request: Request,
    source_file: str,
    store: ChromaStore = Depends(get_store),
) -> DeleteDocumentResponse:
    _ = request

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
