"""Shared FastAPI dependencies."""

from functools import lru_cache

from config.settings import get_settings
from src.finsight.vectorstore.chroma_store import ChromaStore


@lru_cache
def get_store() -> ChromaStore:
    """Return a process-wide ChromaDB store (one SQLite connection)."""
    return ChromaStore()


def get_documents_dir():
    return get_settings().documents_path
