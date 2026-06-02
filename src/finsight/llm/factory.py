"""Factory for selecting the configured LLM provider."""

from functools import lru_cache

from config.settings import get_settings
from src.finsight.llm.base import LLMProvider
from src.finsight.llm.gemini import GeminiLLM
from src.finsight.llm.ollama import OllamaLLM


@lru_cache
def get_llm() -> LLMProvider:
    """Return a process-wide LLM provider."""
    provider = get_settings().llm_provider.lower()
    if provider == "gemini":
        return GeminiLLM()
    if provider == "ollama":
        return OllamaLLM()
    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r}. Use 'gemini' or 'ollama'."
    )
