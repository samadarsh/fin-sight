"""Factory for selecting the configured embedding provider."""

from functools import lru_cache

from config.settings import get_settings
from src.finsight.embeddings.base import EmbeddingProvider
from src.finsight.embeddings.gemini import GeminiEmbeddings
from src.finsight.embeddings.local import LocalEmbeddings


@lru_cache
def get_embedder() -> EmbeddingProvider:
    """Return a process-wide embedding provider (loads local models once)."""
    provider = get_settings().embedding_provider.lower()
    if provider == "local":
        return LocalEmbeddings()
    if provider == "gemini":
        return GeminiEmbeddings()
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER: {provider!r}. Use 'local' or 'gemini'."
    )
