"""Gemini LLM provider."""

import time

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from config.settings import get_settings
from src.finsight.embeddings.gemini import _is_daily_quota_exhausted, _retry_delay_seconds
from src.finsight.errors import GeminiQuotaError
from src.finsight.llm.base import LLMProvider


class GeminiLLM(LLMProvider):
    """LLM provider backed by Google's Gemini chat models."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 5,
    ) -> None:
        settings = get_settings()
        api_key = api_key or settings.gemini_api_key
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Add it to your .env file "
                "(get a free key at https://aistudio.google.com/apikey)."
            )
        genai.configure(api_key=api_key)
        self.model_name = model or settings.llm_model
        self.max_retries = max_retries

    def generate(self, system: str, user: str) -> str:
        model = genai.GenerativeModel(self.model_name, system_instruction=system)
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = model.generate_content(user)
                text = getattr(response, "text", None)
                if text:
                    return text.strip()
                raise RuntimeError("Gemini returned an empty response")
            except google_exceptions.ResourceExhausted as exc:
                if _is_daily_quota_exhausted(exc):
                    raise GeminiQuotaError(
                        "Daily Gemini LLM quota exhausted. "
                        "Wait until the quota resets, upgrade your API key, "
                        "or set LLM_PROVIDER=ollama in .env for local answers.",
                        daily=True,
                        kind="llm",
                    ) from exc
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(_retry_delay_seconds(exc))
            except Exception as exc:  # noqa: BLE001 - retry transient API errors
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)

        msg = f"Gemini generation failed after {self.max_retries} attempts"
        raise RuntimeError(msg) from last_error
