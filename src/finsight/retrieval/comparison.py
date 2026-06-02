"""Helpers for multi-company comparison queries."""

from __future__ import annotations

import re

_COMPARISON_KEYWORDS = (
    "compare",
    "comparison",
    "versus",
    " vs ",
    " vs.",
    " vs,",
    "difference between",
    "contrast",
    "side by side",
    "relative to",
    "compared to",
    "compared with",
    "between",
    "outperform",
    "underperform",
    "benchmark against",
)

# Optional aliases for company names mentioned in questions.
_COMPANY_ALIASES: dict[str, str] = {
    "INDIAN OIL": "IOC",
    "INDIAN OIL CORPORATION": "IOC",
    "TATA CONSULTANCY SERVICES": "TCS",
    "TATA CONSULTANCY": "TCS",
}


def indexed_companies(store) -> list[str]:
    """Return sorted company names present in the vector store."""
    return sorted({doc["company"] for doc in store.list_documents()})


def _normalize_company_tokens(question: str) -> str:
    """Expand known aliases so detection can match indexed tickers."""
    normalized = question.upper()
    for alias, ticker in _COMPANY_ALIASES.items():
        normalized = normalized.replace(alias, ticker)
    return normalized


def detect_companies_in_question(question: str, known_companies: list[str]) -> list[str]:
    """Find company names mentioned in the question using word boundaries."""
    normalized = _normalize_company_tokens(question)
    detected: list[str] = []
    for company in known_companies:
        pattern = rf"\b{re.escape(company.upper())}\b"
        if re.search(pattern, normalized):
            detected.append(company)
    return detected


def is_comparison_query(question: str, companies_in_question: list[str]) -> bool:
    """True if the question looks like a cross-company comparison."""
    q = question.lower()
    if any(keyword in q for keyword in _COMPARISON_KEYWORDS):
        return True
    return len(companies_in_question) >= 2


def resolve_comparison_companies(
    question: str,
    store,
    explicit: list[str] | None = None,
) -> list[str]:
    """Pick which companies to compare.

    Priority:
    1. Explicit list from CLI / API
    2. Companies named in the question (2+)
    3. All indexed companies when exactly two are in the store
    """
    if explicit:
        return explicit

    known = indexed_companies(store)
    detected = detect_companies_in_question(question, known)
    if len(detected) >= 2:
        return detected
    if len(known) == 2:
        return known
    return detected
