"""Request and response schemas for the FastAPI layer."""

from typing import Literal

from pydantic import BaseModel, Field

from src.finsight.models import IngestResult, QueryResponse

DocType = Literal["annual_report", "transcript", "presentation", "filing"]


class HealthResponse(BaseModel):
    status: str = "ok"


class UploadResponse(BaseModel):
    source_file: str
    path: str
    size_bytes: int


class IngestRequest(BaseModel):
    source_file: str = Field(..., description="PDF filename in the documents directory")
    company: str
    doc_type: DocType
    year: int | None = None
    quarter: str | None = None
    replace_existing: bool = False
    resume: bool = True


class IngestResponse(IngestResult):
    pass


class QueryRequest(BaseModel):
    question: str
    company: str | None = Field(None, description="Filter to one company (standard mode)")
    companies: list[str] | None = Field(None, description="Companies for comparison mode")
    compare: bool | None = Field(None, description="Force comparison on/off; null = auto-detect")
    k: int | None = None
    k_per_company: int | None = None


class QueryAPIResponse(QueryResponse):
    mode: str


class DocumentSummary(BaseModel):
    source_file: str
    company: str
    doc_type: str
    year: int | None = None
    chunks: int


class DeleteDocumentResponse(BaseModel):
    source_file: str
    deleted: bool
    remaining_chunks: int
