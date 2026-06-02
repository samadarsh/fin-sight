"""Gemini embedding provider (``gemini-embedding-001``).

Notes on the Gemini embeddings API:
- ``task_type`` matters: use ``retrieval_document`` when embedding chunks for
  storage and ``retrieval_query`` when embedding a search query. Matching the
  task types on both sides measurably improves retrieval quality.
- Free-tier keys are limited to ~100 embed requests per minute. Each string in
  a batch counts as one request, so we pace batches with a configurable delay.
- Daily quota errors (``PerDay``) fail immediately — retrying would hang for
  minutes without helping.
"""

import logging
import re
import time

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from config.settings import get_settings
from src.finsight.embeddings.base import EmbeddingProvider
from src.finsight.errors import GeminiQuotaError

logger = logging.getLogger(__name__)

_DOCUMENT_TASK = "retrieval_document"
_QUERY_TASK = "retrieval_query"
_RETRY_AFTER_RE = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)


def _retry_delay_seconds(exc: Exception, default: float = 65.0) -> float:
    """Parse Gemini's suggested retry delay, with a safe default."""
    match = _RETRY_AFTER_RE.search(str(exc))
    if match:
        return float(match.group(1)) + 1.0
    return default


def _is_daily_quota_exhausted(exc: Exception) -> bool:
    """True when the error is a daily quota limit (retrying today won't help)."""
    message = str(exc)
    return "PerDay" in message or "limit: 1000" in message


class GeminiEmbeddings(EmbeddingProvider):
    """Embedding provider backed by Google's Gemini embedding models."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        batch_size: int | None = None,
        batch_delay: float | None = None,
        max_retries: int = 3,
    ) -> None:
        settings = get_settings()
        api_key = api_key or settings.gemini_api_key
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Add it to your .env file "
                "(get a free key at https://aistudio.google.com/apikey)."
            )
        genai.configure(api_key=api_key)
        self.model = model or settings.embedding_model
        self.batch_size = (
            batch_size if batch_size is not None else settings.embedding_batch_size
        )
        self.batch_delay = (
            batch_delay if batch_delay is not None else settings.embedding_batch_delay
        )
        self.max_retries = max_retries

    def _embed(self, content: list[str] | str, task_type: str) -> list[list[float]]:
        """Call the Gemini API with retries for transient rate limits only."""
        model_name = self.model if self.model.startswith("models/") else f"models/{self.model}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                result = genai.embed_content(
                    model=model_name,
                    content=content,
                    task_type=task_type,
                )
                embedding = result["embedding"]
                if embedding and isinstance(embedding[0], float):
                    return [embedding]
                return embedding
            except google_exceptions.ResourceExhausted as exc:
                if _is_daily_quota_exhausted(exc):
                    raise GeminiQuotaError(
                        "Daily Gemini embedding quota exhausted (~1000/day on free tier). "
                        "Wait until the quota resets or upgrade your API key.",
                        daily=True,
                        kind="embedding",
                    ) from exc
                last_error = exc
                if attempt < self.max_retries - 1:
                    delay = _retry_delay_seconds(exc)
                    logger.warning(
                        "Embedding rate limited (attempt %d/%d), retrying in %.0fs…",
                        attempt + 1,
                        self.max_retries,
                        delay,
                    )
                    time.sleep(delay)
            except Exception as exc:  # noqa: BLE001 - retry transient API errors
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
        msg = f"Gemini embedding failed after {self.max_retries} attempts"
        raise RuntimeError(msg) from last_error

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for batch_index, start in enumerate(range(0, len(texts), self.batch_size), start=1):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed(batch, _DOCUMENT_TASK))
            if self.batch_delay and batch_index < total_batches:
                time.sleep(self.batch_delay)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, _QUERY_TASK)[0]
