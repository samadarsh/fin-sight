"""Embedding provider fingerprinting for vector-store compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import get_settings
from src.finsight.embeddings.base import EmbeddingProvider
from src.finsight.embeddings.gemini import GeminiEmbeddings
from src.finsight.embeddings.local import LocalEmbeddings


@dataclass(frozen=True)
class EmbeddingFingerprint:
    """Identifies the embedding model used to build a vector index."""

    provider: str
    model: str
    dimension: int

    def as_metadata(self) -> dict[str, str | int]:
        return {
            "embedding_provider": self.provider,
            "embedding_model": self.model,
            "embedding_dimension": self.dimension,
        }

    @classmethod
    def from_metadata(cls, metadata: dict | None) -> EmbeddingFingerprint | None:
        if not metadata:
            return None
        provider = metadata.get("embedding_provider")
        model = metadata.get("embedding_model")
        dimension = metadata.get("embedding_dimension")
        if provider is None or model is None or dimension is None:
            return None
        return cls(provider=str(provider), model=str(model), dimension=int(dimension))


def fingerprint_embedder(embedder: EmbeddingProvider) -> EmbeddingFingerprint:
    """Compute a stable fingerprint for the active embedding provider."""
    settings = get_settings()

    if isinstance(embedder, LocalEmbeddings):
        vector = embedder.embed_query("dimension probe")
        return EmbeddingFingerprint(
            provider="local",
            model=embedder.model_name,
            dimension=len(vector),
        )

    if isinstance(embedder, GeminiEmbeddings):
        vector = embedder.embed_query("dimension probe")
        return EmbeddingFingerprint(
            provider="gemini",
            model=embedder.model,
            dimension=len(vector),
        )

    vector = embedder.embed_query("dimension probe")
    provider = settings.embedding_provider.lower()
    model = getattr(embedder, "model_name", None) or getattr(embedder, "model", "unknown")
    return EmbeddingFingerprint(provider=provider, model=str(model), dimension=len(vector))
