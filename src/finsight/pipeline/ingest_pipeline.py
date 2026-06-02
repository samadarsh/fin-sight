"""Document ingestion pipeline.

Orchestrates the full path from a PDF on disk to searchable vectors:

    load → clean → chunk → embed (batch) → store (batch)

Embeddings are written to Chroma after each batch so a rate-limit error does
not discard progress. Re-run with ``resume=True`` to continue where you left off.
"""

import logging
import time
from pathlib import Path

from src.finsight.embeddings.base import EmbeddingProvider
from src.finsight.embeddings.factory import get_embedder
from src.finsight.errors import GeminiQuotaError
from src.finsight.ingestion.chunker import chunk_pages
from src.finsight.ingestion.cleaner import clean_pages
from src.finsight.ingestion.loader import load_pdf
from src.finsight.models import Chunk, IngestResult
from src.finsight.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


def _embedding_settings(embedder: EmbeddingProvider) -> tuple[int, float]:
    batch_size = getattr(embedder, "batch_size", 50)
    batch_delay = getattr(embedder, "batch_delay", 0.0)
    return batch_size, batch_delay


def _embed_and_store_batches(
    chunks: list[Chunk],
    *,
    embedder: EmbeddingProvider,
    store: ChromaStore,
    existing_ids: set[str],
) -> tuple[int, int, bool]:
    """Embed chunks in batches, persisting after each batch.

    Returns:
        ``(chunks_added, chunks_skipped, hit_rate_limit)``
    """
    batch_size, batch_delay = _embedding_settings(embedder)
    pending = [chunk for chunk in chunks if chunk.metadata.chunk_id not in existing_ids]
    skipped = len(chunks) - len(pending)
    written = 0
    total_batches = (len(pending) + batch_size - 1) // batch_size if pending else 0

    for batch_index, start in enumerate(range(0, len(pending), batch_size), start=1):
        batch = pending[start : start + batch_size]
        try:
            vectors = embedder.embed_documents([chunk.text for chunk in batch])
        except GeminiQuotaError as exc:
            if written > 0:
                logger.warning("%s Saved %d chunks — re-run with resume=True later.", exc, written)
                return written, skipped, True
            raise
        except RuntimeError as exc:
            if "Gemini embedding failed" in str(exc) and written > 0:
                logger.warning(
                    "Rate limit hit after %d chunks; re-run with resume=True to continue",
                    written,
                )
                return written, skipped, True
            raise

        written += store.add_chunks(batch, vectors)
        logger.info(
            "Stored batch %d/%d (%d chunks total for this run)",
            batch_index,
            total_batches,
            written,
        )
        if batch_delay and batch_index < total_batches:
            time.sleep(batch_delay)

    return written, skipped, False


def ingest_document(
    path: str | Path,
    *,
    company: str,
    doc_type: str,
    year: int | None = None,
    quarter: str | None = None,
    source_file: str | None = None,
    replace_existing: bool = False,
    resume: bool = True,
    embedder: EmbeddingProvider | None = None,
    store: ChromaStore | None = None,
) -> IngestResult:
    """Ingest a PDF into the vector store.

    Args:
        path: Path to the PDF file.
        company: Company ticker or name, e.g. ``"TCS"``.
        doc_type: One of ``annual_report``, ``transcript``, ``presentation``,
            ``filing``.
        year: Reporting year, if known.
        quarter: Reporting quarter, e.g. ``"Q1"``, if applicable.
        source_file: Filename stored in metadata and used for deletion /
            listing. Defaults to the PDF basename.
        replace_existing: If True, delete all existing chunks for this file
            before ingesting. Use this for a clean re-ingest.
        resume: If True, skip chunks whose ``chunk_id`` is already in the
            store. Combine with ``replace_existing=False`` after a partial run.
        embedder: Optional embedding provider (defaults to ``GeminiEmbeddings``).
        store: Optional vector store (defaults to persistent ``ChromaStore``).

    Returns:
        An ``IngestResult`` with counts and metadata. ``partial=True`` when
        ingestion stopped early due to a rate limit but some chunks were saved.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the PDF yields no embeddable text.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    source_file = source_file or path.name
    embedder = embedder or get_embedder()
    store = store or ChromaStore()
    store.ensure_embedding_compatible(embedder, write=True)

    logger.info("Loading %s", path)
    raw_pages = load_pdf(path)
    cleaned_pages = clean_pages(raw_pages)
    if not cleaned_pages:
        raise ValueError(f"No extractable text in {path}")

    logger.info("Chunking %s (%d pages)", source_file, len(cleaned_pages))
    chunks = chunk_pages(
        cleaned_pages,
        company=company,
        doc_type=doc_type,
        source_file=source_file,
        year=year,
        quarter=quarter,
    )
    if not chunks:
        raise ValueError(f"No chunks produced from {path}")

    if replace_existing:
        logger.info("Removing existing chunks for %s", source_file)
        store.delete_by_source(source_file)

    existing_ids = store.existing_chunk_ids(source_file) if resume else set()
    if existing_ids:
        logger.info("Resuming %s — %d chunks already stored", source_file, len(existing_ids))

    batch_size, _ = _embedding_settings(embedder)
    pending_count = sum(1 for c in chunks if c.metadata.chunk_id not in existing_ids)
    logger.info(
        "Embedding %d new chunks for %s (~%d API batches)",
        pending_count,
        source_file,
        (pending_count + batch_size - 1) // batch_size if pending_count else 0,
    )

    written, skipped, partial = _embed_and_store_batches(
        chunks,
        embedder=embedder,
        store=store,
        existing_ids=existing_ids,
    )

    return IngestResult(
        source_file=source_file,
        company=company,
        doc_type=doc_type,
        year=year,
        pages_processed=len(cleaned_pages),
        chunks_added=written,
        chunks_skipped=skipped,
        total_chunks_in_store=store.count(),
        partial=partial,
    )
