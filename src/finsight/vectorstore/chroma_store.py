"""ChromaDB vector store wrapper.

Design decisions:
- **One persistent collection** (``finsight_docs``) for everything. Metadata
  filtering (by company / doc_type / year) lets us scope queries, and a single
  collection makes cross-company comparison queries trivial later.
- We pass embeddings in explicitly (computed by our ``EmbeddingProvider``)
  rather than letting Chroma embed, so we control the model and task types.
- Cosine distance is configured on the collection; we convert Chroma's
  distance to a similarity score for the ``Source`` objects.
- Collection metadata stores ``embedding_provider``, ``embedding_model``, and
  ``embedding_dimension`` so switching models fails fast instead of silently
  degrading retrieval quality.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb

from config.settings import get_settings
from src.finsight.embeddings.base import EmbeddingProvider
from src.finsight.embeddings.fingerprint import EmbeddingFingerprint, fingerprint_embedder
from src.finsight.errors import EmbeddingMismatchError
from src.finsight.models import Chunk, Source

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

    def stored_embedding_fingerprint(self) -> EmbeddingFingerprint | None:
        """Return embedding metadata stamped on the collection, if any."""
        return EmbeddingFingerprint.from_metadata(self.collection.metadata)

    def _peek_embedding_dimension(self) -> int | None:
        """Read vector dimension from one stored record (legacy indexes)."""
        try:
            data = self.collection.get(limit=1, include=["embeddings"])
            embeddings = data.get("embeddings")
            if embeddings is None or len(embeddings) == 0:
                return None
            first = embeddings[0]
            if first is None:
                return None
            return len(first)
        except Exception:  # noqa: BLE001
            return None

    def _stamp_embedding_metadata(self, fingerprint: EmbeddingFingerprint) -> None:
        """Persist embedding model info on the collection."""
        # Chroma rejects modify calls that include ``hnsw:space``.
        patch = fingerprint.as_metadata()
        self.collection.modify(metadata=patch)

    def ensure_embedding_compatible(
        self,
        embedder: EmbeddingProvider,
        *,
        write: bool = False,
    ) -> None:
        """Verify the active embedder matches the indexed vectors.

        Args:
            embedder: Embedding provider about to be used.
            write: When True and the store is empty or legacy, stamp metadata.

        Raises:
            EmbeddingMismatchError: Provider/model/dimension mismatch.
        """
        current = fingerprint_embedder(embedder)
        stored = self.stored_embedding_fingerprint()
        count = self.count()

        if stored is None:
            if count == 0:
                self._stamp_embedding_metadata(current)
                return
            legacy_dim = self._peek_embedding_dimension()
            if legacy_dim is not None and legacy_dim != current.dimension:
                raise EmbeddingMismatchError(
                    f"Embedding dimension mismatch: index has {legacy_dim}-dim vectors "
                    f"but {current.provider}/{current.model} produces {current.dimension}-dim. "
                    "Re-ingest with --replace or restore the original embedding settings.",
                    stored=f"legacy-{legacy_dim}d",
                    requested=f"{current.provider}/{current.model}",
                )
            if write:
                self._stamp_embedding_metadata(current)
            return

        if (
            stored.provider != current.provider
            or stored.model != current.model
            or stored.dimension != current.dimension
        ):
            raise EmbeddingMismatchError(
                "Embedding model mismatch. The vector index was built with "
                f"{stored.provider}/{stored.model} ({stored.dimension}-dim) but settings "
                f"request {current.provider}/{current.model} ({current.dimension}-dim). "
                "Switch back to the original embedding settings or re-ingest all documents "
                "with --replace after changing EMBEDDING_PROVIDER.",
                stored=f"{stored.provider}/{stored.model}",
                requested=f"{current.provider}/{current.model}",
            )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Upsert chunks and their vectors into the collection."""
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
        """Run a similarity search and return cited sources."""
        stored = self.stored_embedding_fingerprint()
        if stored is not None and len(query_embedding) != stored.dimension:
            raise EmbeddingMismatchError(
                f"Query embedding dimension {len(query_embedding)} does not match "
                f"indexed dimension {stored.dimension}. Check EMBEDDING_PROVIDER settings.",
            )

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
                    score=round(1.0 - distance, 4),
                )
            )
        return sources

    def delete_by_source(self, source_file: str) -> None:
        """Delete all chunks belonging to a given source file."""
        self.collection.delete(where={"source_file": source_file})

    def list_documents(self) -> list[dict]:
        """List distinct ingested documents with chunk counts."""
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
