"""Local embedding provider (sentence-transformers).

Runs entirely on your machine — no API key, no rate limits. Good for
ingesting large document sets during development.

Default model: ``BAAI/bge-small-en-v1.5`` (~130 MB download on first run).
BGE models expect a query prefix for retrieval; documents are embedded as-is.
"""

from sentence_transformers import SentenceTransformer

from config.settings import get_settings
from src.finsight.embeddings.base import EmbeddingProvider

# Recommended by the BGE authors for retrieval queries.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LocalEmbeddings(EmbeddingProvider):
    """Embedding provider using a local sentence-transformers model."""

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        batch_delay: float = 0.0,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.local_embedding_model
        self.batch_size = (
            batch_size if batch_size is not None else settings.local_embedding_batch_size
        )
        self.batch_delay = batch_delay  # unused; kept for ingest_pipeline compatibility
        self._model = SentenceTransformer(self.model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        query = text
        if "bge" in self.model_name.lower():
            query = _BGE_QUERY_PREFIX + text
        vector = self._model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()
