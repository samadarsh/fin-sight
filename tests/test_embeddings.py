"""Unit tests for the embeddings layer.

These mock the network call so they run offline and fast; the live Gemini
call is exercised manually during development.
"""

import pytest
from google.api_core import exceptions as google_exceptions

from src.finsight.embeddings.gemini import GeminiEmbeddings, _is_daily_quota_exhausted
from src.finsight.errors import GeminiQuotaError


def _make_provider(monkeypatch) -> GeminiEmbeddings:
    monkeypatch.setattr(
        "src.finsight.embeddings.gemini.genai.configure", lambda **_: None
    )
    return GeminiEmbeddings(api_key="fake-key", batch_size=2, batch_delay=0)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.setattr(
        "src.finsight.embeddings.gemini.genai.configure", lambda **_: None
    )

    class _NoKeySettings:
        gemini_api_key = ""
        embedding_model = "gemini-embedding-001"

    monkeypatch.setattr(
        "src.finsight.embeddings.gemini.get_settings", lambda: _NoKeySettings()
    )
    with pytest.raises(ValueError):
        GeminiEmbeddings(api_key="")


def test_embed_documents_batches(monkeypatch):
    provider = _make_provider(monkeypatch)
    calls: list[list[str]] = []

    def fake_embed(content, task_type):
        calls.append(content)
        return [[0.1, 0.2, 0.3] for _ in content]

    monkeypatch.setattr(provider, "_embed", fake_embed)
    vectors = provider.embed_documents(["a", "b", "c"])  # batch_size=2 -> 2 calls
    assert len(vectors) == 3
    assert len(calls) == 2
    assert all(len(v) == 3 for v in vectors)


def test_embed_query_returns_single_vector(monkeypatch):
    provider = _make_provider(monkeypatch)
    monkeypatch.setattr(provider, "_embed", lambda content, task_type: [[0.5, 0.6]])
    vec = provider.embed_query("a question")
    assert vec == [0.5, 0.6]


def test_embed_documents_empty_input(monkeypatch):
    provider = _make_provider(monkeypatch)
    assert provider.embed_documents([]) == []


def test_is_daily_quota_exhausted_detects_per_day_limit():
    exc = google_exceptions.ResourceExhausted(
        "limit: 1000, quota_id: EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier"
    )
    assert _is_daily_quota_exhausted(exc) is True


def test_daily_quota_fails_immediately(monkeypatch):
    provider = _make_provider(monkeypatch)
    daily_error = google_exceptions.ResourceExhausted(
        "limit: 1000, PerDayPerUserPerProjectPerModel-FreeTier"
    )

    def fail_embed(**kwargs):
        raise daily_error

    monkeypatch.setattr("src.finsight.embeddings.gemini.genai.embed_content", fail_embed)

    def should_not_sleep(_):
        raise AssertionError("should not sleep on daily quota")

    monkeypatch.setattr("src.finsight.embeddings.gemini.time.sleep", should_not_sleep)

    with pytest.raises(GeminiQuotaError):
        provider.embed_query("test question")
