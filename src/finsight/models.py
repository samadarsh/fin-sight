"""Core data models shared across the RAG pipeline.

Note: ChromaDB metadata values must be primitives (str/int/float/bool) and
cannot be nested, so ``ChunkMetadata`` is intentionally flat.
"""


from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    """Metadata attached to every chunk and stored alongside its vector."""

    company: str
    doc_type: str  # "annual_report" | "transcript" | "presentation" | "filing"
    page: int
    source_file: str
    chunk_id: str
    year: int | None = None
    quarter: str | None = None  # e.g. "Q1", "Q2"

    def to_chroma(self) -> dict:
        """Flatten to a Chroma-safe dict (drop None values)."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class Chunk(BaseModel):
    """A piece of text plus its metadata, ready to embed and store."""

    text: str
    metadata: ChunkMetadata


class Source(BaseModel):
    """A retrieved chunk surfaced to the user as a citation."""

    text: str
    company: str
    doc_type: str
    page: int
    source_file: str
    score: float


class QueryResponse(BaseModel):
    """Final answer returned to the user, with supporting sources."""

    answer: str
    sources: list[Source]


class IngestResult(BaseModel):
    """Summary returned after a document is ingested into the vector store."""

    source_file: str
    company: str
    doc_type: str
    year: int | None = None
    pages_processed: int
    chunks_added: int
    chunks_skipped: int = 0
    total_chunks_in_store: int
    partial: bool = False
