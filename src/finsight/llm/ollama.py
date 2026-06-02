"""Ollama LLM provider — runs locally, no API quota."""

import logging

import requests

from config.settings import get_settings
from src.finsight.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class OllamaLLM(LLMProvider):
    """LLM provider backed by a local Ollama server."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int = 300,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.ollama_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = timeout

    def _list_models(self) -> set[str]:
        response = requests.get(f"{self.base_url}/api/tags", timeout=10)
        response.raise_for_status()
        return {m["name"] for m in response.json().get("models", [])}

    def _ensure_model_available(self) -> None:
        """Fail fast if the configured model is not pulled yet."""
        available = self._list_models()
        if self.model in available:
            return
        # Allow short names: "llama3.1" matches "llama3.1:latest" etc.
        matches = [name for name in available if name.split(":")[0] == self.model.split(":")[0]]
        if matches:
            self.model = matches[0]
            logger.info("Using Ollama model %s", self.model)
            return
        pulled = ", ".join(sorted(available)) or "(none)"
        raise RuntimeError(
            f"Ollama model '{self.model}' is not installed. "
            f"You have: {pulled}. "
            f"Run: ollama pull {self.model}"
        )

    def generate(self, system: str, user: str) -> str:
        self._ensure_model_available()
        logger.info(
            "Generating answer with Ollama model %s (first run may take 1-2 min)…",
            self.model,
        )
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "options": {"num_ctx": 8192},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.ConnectionError as exc:
            raise RuntimeError(
                "Cannot connect to Ollama at "
                f"{self.base_url}. Start it with `ollama serve` in a separate terminal, "
                "or open the Ollama app."
            ) from exc
        except requests.Timeout as exc:
            raise RuntimeError(
                f"Ollama timed out after {self.timeout}s. Try a smaller model "
                f"(e.g. `ollama pull llama3.2:1b` and set OLLAMA_MODEL=llama3.2:1b)."
            ) from exc
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            raise RuntimeError(f"Ollama request failed: {detail}") from exc

        data = response.json()
        message = data.get("message", {})
        text = message.get("content", "").strip()
        if not text:
            raise RuntimeError("Ollama returned an empty response")
        return text
