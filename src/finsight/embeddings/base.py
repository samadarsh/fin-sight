"""Embedding provider interface.

Defining an abstract base lets the rest of the system depend on a stable
interface rather than a specific vendor. Swapping Gemini for OpenAI / BGE /
E5 later means writing one new subclass, with no changes elsewhere.
"""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract embedding provider.

    Implementations turn text into fixed-length float vectors. Documents and
    queries are embedded via separate methods because some models use
    different task-type hints for each (which improves retrieval quality).
    """

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunks for storage."""
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single user query for similarity search."""
        raise NotImplementedError
