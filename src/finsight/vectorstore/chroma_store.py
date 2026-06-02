"""ChromaDB vector store wrapper.

Design decisions:
- **One persistent collection** (``finsight_docs``) for everything. Metadata
  filtering (by company / doc_type / year) lets us scope queries, and a single
  collection makes cross-company comparison queries trivial later.
- We pass embeddings in explicitly (computed by our ``EmbeddingProvider``)
  rather than letting Chroma embed, so we control the model and task types.
- Cosine distance is configured on the collection; we convert Chroma's
  distance to a similarity score for the ``Source`` objects.
"""

import logging
from pathlib import Path

import chromadb

from config.settings import get_settings
from src.finsight.models import Chunk, Source

# ChromaDB 0.5.x has a posthog telemetry bug that logs a harmless
# "capture() takes 1 positional argument" error on every call even when
# telemetry is disabled. Silence its logger to keep output clean.
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)


class ChromaStore:
    """Thin wrapper around a persistent ChromaDB collection."""

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection: str | None = None,
    ) -> None:
        settings = get_settings()
        persist_dir = Path(persist_dir) if persist_dir else settings.chroma_path
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection or settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Upsert chunks and their vectors into the collection.

        Uses ``chunk_id`` as the Chroma id, so re-ingesting the same document
        overwrites rather than duplicates.

        Args:
            chunks: Chunks to store.
            embeddings: Vectors aligned 1:1 with ``chunks``.

        Returns:
            Number of chunks written.
        """
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
            )

        self.collection.upsert(
            ids=[c.metadata.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata.to_chroma() for c in chunks],
        )
        return len(chunks)

    def query(
        self,
        query_embedding: list[float],
        k: int = 5,
        filters: dict | None = None,
    ) -> list[Source]:
        """Run a similarity search and return cited sources.

        Args:
            query_embedding: The embedded user query.
            k: Number of results to return.
            filters: Optional Chroma ``where`` filter, e.g.
                ``{"company": "TCS"}`` or
                ``{"$and": [{"company": "TCS"}, {"year": 2026}]}``.

        Returns:
            A list of ``Source`` objects ordered most-similar first.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filters or None,
            include=["documents", "metadatas", "distances"],
        )

        sources: list[Source] = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        for text, meta, distance in zip(documents, metadatas, distances):
            sources.append(
                Source(
                    text=text,
                    company=meta.get("company", "Unknown"),
                    doc_type=meta.get("doc_type", "unknown"),
                    page=int(meta.get("page", 0)),
                    source_file=meta.get("source_file", "unknown"),
                    score=round(1.0 - distance, 4),  # cosine distance -> similarity
                )
            )
        return sources

    def delete_by_source(self, source_file: str) -> None:
        """Delete all chunks belonging to a given source file."""
        self.collection.delete(where={"source_file": source_file})

    def list_documents(self) -> list[dict]:
        """List distinct ingested documents with chunk counts.

        Returns:
            One dict per source file with company, doc_type, year, and the
            number of chunks stored for it.
        """
        records = self.collection.get(include=["metadatas"])
        by_source: dict[str, dict] = {}
        for meta in records["metadatas"]:
            source = meta.get("source_file", "unknown")
            if source not in by_source:
                by_source[source] = {
                    "source_file": source,
                    "company": meta.get("company", "Unknown"),
                    "doc_type": meta.get("doc_type", "unknown"),
                    "year": meta.get("year"),
                    "chunks": 0,
                }
            by_source[source]["chunks"] += 1
        return list(by_source.values())

    def existing_chunk_ids(self, source_file: str) -> set[str]:
        """Return chunk ids already stored for a given source file."""
        records = self.collection.get(
            where={"source_file": source_file},
            include=[],
        )
        return set(records["ids"])

    def count(self) -> int:
        """Total number of chunks stored."""
        return self.collection.count()
