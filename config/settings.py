"""Application configuration, loaded from environment / .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (config/settings.py -> root).
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central settings object. Reads from environment variables / .env."""

    # LLM / Embeddings
    gemini_api_key: str = ""
    llm_provider: str = "ollama"  # "gemini" | "ollama"
    llm_model: str = "gemini-2.0-flash"
    ollama_model: str = "llama3:latest"
    ollama_base_url: str = "http://localhost:11434"
    embedding_provider: str = "local"  # "local" | "gemini"
    embedding_model: str = "gemini-embedding-001"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    local_embedding_batch_size: int = 64

    # Vector store
    chroma_dir: str = "./storage/chroma"
    chroma_collection: str = "finsight_docs"

    # Document storage
    documents_dir: str = "./data/documents"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Retrieval
    retrieval_top_k: int = 5
    comparison_k_per_company: int = 4

    # Embedding throughput (free-tier Gemini ~100 requests/minute)
    embedding_batch_size: int = 10
    embedding_batch_delay: float = 70.0

    # API security / limits
    max_upload_bytes: int = 100 * 1024 * 1024  # 100 MB
    cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501"
    rate_limit: str = "60/minute"

    @property
    def cors_origin_list(self) -> list[str]:
        """Parsed CORS allowlist for FastAPI."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def chroma_path(self) -> Path:
        """Absolute path to the Chroma persistence directory."""
        return (ROOT_DIR / self.chroma_dir).resolve()

    @property
    def documents_path(self) -> Path:
        """Absolute path to the uploaded-documents directory."""
        return (ROOT_DIR / self.documents_dir).resolve()


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
