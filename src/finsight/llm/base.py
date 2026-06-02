"""LLM provider interface."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract LLM provider for answer generation."""

    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        """Generate a response from system + user messages."""
        raise NotImplementedError
