"""Citation formatting and post-processing for RAG answers."""

from __future__ import annotations

import re

from src.finsight.models import Source

# Matches [annual-report-2025-2026.pdf p.58] style citations.
INLINE_CITATION_RE = re.compile(
    r"\[[^\]\s]+\.pdf p\.\d+\]",
    re.IGNORECASE,
)


def format_citation(source_file: str, page: int) -> str:
    """Return the canonical inline citation token for a source."""
    return f"[{source_file} p.{page}]"


def citation_tokens_for_sources(sources: list[Source]) -> dict[int, str]:
    """Map 1-based source index to citation token."""
    return {
        index: format_citation(src.source_file, src.page)
        for index, src in enumerate(sources, start=1)
    }


def normalize_answer_citations(answer: str, sources: list[Source]) -> str:
    """Replace legacy labels and ensure inline filename/page citations appear.

    The LLM often echoes ``Source 1`` labels from context. This function:
    1. Rewrites ``Source N``, ``[CITE:N]``, and ``(CITE-N)`` to ``[file p.X]``.
    2. If no inline ``[*.pdf p.N]`` citations remain, appends a Citations line
       built from the retrieved sources so users always see page-level references.
    """
    if not sources:
        return answer

    tokens = citation_tokens_for_sources(sources)
    normalized = answer

    for index, cite in tokens.items():
        normalized = re.sub(rf"\bSource {index}\b", cite, normalized, flags=re.IGNORECASE)
        normalized = re.sub(rf"\[CITE:{index}\]", cite, normalized, flags=re.IGNORECASE)
        normalized = re.sub(rf"\(CITE-{index}\)", cite, normalized, flags=re.IGNORECASE)

    if INLINE_CITATION_RE.search(normalized):
        return normalized.strip()

    unique: list[str] = []
    seen: set[str] = set()
    for src in sources:
        cite = format_citation(src.source_file, src.page)
        if cite not in seen:
            seen.add(cite)
            unique.append(cite)

    citation_line = "Citations: " + ", ".join(unique[:8])
    return f"{normalized.rstrip()}\n\n{citation_line}"


def has_inline_citations(text: str) -> bool:
    """True when text contains at least one ``[file.pdf p.N]`` citation."""
    return bool(INLINE_CITATION_RE.search(text))
