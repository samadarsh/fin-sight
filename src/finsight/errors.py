"""Shared API error types."""


class GeminiQuotaError(RuntimeError):
    """Gemini quota exhausted; retrying immediately will not help."""

    def __init__(self, message: str, *, daily: bool = True, kind: str = "api") -> None:
        super().__init__(message)
        self.daily = daily
        self.kind = kind  # "embedding" | "llm" | "api"


class EmbeddingMismatchError(ValueError):
    """Query/embed attempted with an embedding model incompatible with the index."""

    def __init__(
        self,
        message: str,
        *,
        stored: str | None = None,
        requested: str | None = None,
    ) -> None:
        super().__init__(message)
        self.stored = stored
        self.requested = requested
