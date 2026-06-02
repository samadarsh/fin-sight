"""Chunking: turn cleaned pages into embeddable ``Chunk`` objects.

Uses LangChain's ``RecursiveCharacterTextSplitter`` (the industry-standard
"Version 2" approach), which tries to split on paragraph -> line -> sentence
-> word boundaries before falling back to hard character cuts. This keeps
chunks semantically coherent compared to naive fixed-size splitting.

Each chunk carries flat metadata (company, doc type, page, source file, a
stable chunk id) so retrieved results can be cited precisely.
"""

import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import get_settings
from src.finsight.models import Chunk, ChunkMetadata


def _make_chunk_id(source_file: str, page: int, index: int, text: str) -> str:
    """Deterministic chunk id from source location + content hash.

    Including a content hash means re-ingesting an identical document produces
    identical ids (useful for idempotent upserts later).
    """
    payload = f"{source_file}|{page}|{index}|{text}".encode()
    return hashlib.sha1(payload).hexdigest()[:16]


def chunk_pages(
    pages: list[tuple[int, str]],
    *,
    company: str,
    doc_type: str,
    source_file: str,
    year: int | None = None,
    quarter: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split cleaned pages into overlapping chunks with metadata.

    Args:
        pages: ``(page_number, cleaned_text)`` tuples.
        company: Company name, e.g. "TCS".
        doc_type: One of "annual_report" | "transcript" | "presentation" | "filing".
        source_file: Original filename, used for citation + chunk ids.
        year: Reporting year, if known.
        quarter: Reporting quarter, e.g. "Q1", if applicable.
        chunk_size: Target chunk size in characters (defaults to settings).
        chunk_overlap: Overlap between chunks (defaults to settings).

    Returns:
        A list of ``Chunk`` objects ready for embedding.
    """
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[Chunk] = []
    for page_no, text in pages:
        for index, piece in enumerate(splitter.split_text(text)):
            piece = piece.strip()
            if not piece:
                continue
            metadata = ChunkMetadata(
                company=company,
                doc_type=doc_type,
                page=page_no,
                source_file=source_file,
                chunk_id=_make_chunk_id(source_file, page_no, index, piece),
                year=year,
                quarter=quarter,
            )
            chunks.append(Chunk(text=piece, metadata=metadata))
    return chunks
