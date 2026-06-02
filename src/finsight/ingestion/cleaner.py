"""Text cleaning for extracted PDF pages.

Two layers of cleaning:

1. Document-level: detect *repeated* short lines (running headers/footers,
   section banners, the report title) that appear on many pages and strip them.
2. Page-level: normalise whitespace and drop standalone page-number lines.

The goal is to remove boilerplate noise so chunks contain mostly real content,
which improves both embedding quality and retrieval precision.
"""

import re
from collections import Counter

# A line that is only a page number, optionally with surrounding punctuation
# or a "Page 12" / "12 | " style decoration.
_PAGE_NUMBER_RE = re.compile(r"^(page\s*)?[\|\-\s]*\d{1,4}[\|\-\s]*$", re.IGNORECASE)

# Runs of whitespace (incl. tabs) within a line.
_WS_RE = re.compile(r"[ \t]+")


def _normalize_line(line: str) -> str:
    """Strip and collapse internal whitespace for a single line."""
    return _WS_RE.sub(" ", line.replace("\t", " ")).strip()


def _is_noise_line(line: str) -> bool:
    """True if a line is a standalone page number or empty."""
    if not line:
        return True
    return bool(_PAGE_NUMBER_RE.match(line))


def find_boilerplate_lines(
    pages: list[tuple[int, str]],
    min_pages: int = 5,
    page_fraction: float = 0.2,
    max_line_len: int = 70,
) -> set[str]:
    """Detect repeated short lines that are likely running headers/footers.

    A normalized line is flagged as boilerplate if it is short and appears on
    at least ``threshold`` distinct pages, where ``threshold`` is the larger of
    ``min_pages`` and ``page_fraction`` of the total page count.

    Args:
        pages: ``(page_number, text)`` tuples from the loader.
        min_pages: Absolute minimum page count to consider a line repeated.
        page_fraction: Fraction of total pages a line must appear on.
        max_line_len: Only short lines are eligible (long lines are content).

    Returns:
        A set of normalized boilerplate lines to strip.
    """
    counts: Counter[str] = Counter()
    for _, text in pages:
        seen_on_page = {
            line
            for raw in text.splitlines()
            if (line := _normalize_line(raw)) and len(line) <= max_line_len
        }
        counts.update(seen_on_page)

    threshold = max(min_pages, int(len(pages) * page_fraction))
    return {line for line, count in counts.items() if count >= threshold}


def clean_text(text: str, boilerplate: set[str] | None = None) -> str:
    """Clean a single page's text.

    Removes boilerplate lines, standalone page numbers, and empty lines, and
    normalises internal whitespace. Lines are rejoined with ``\\n`` so the
    chunker can still use paragraph boundaries as split points.

    Args:
        text: Raw page text.
        boilerplate: Normalized lines to strip (from ``find_boilerplate_lines``).

    Returns:
        Cleaned text.
    """
    boilerplate = boilerplate or set()
    cleaned_lines: list[str] = []
    for raw in text.splitlines():
        line = _normalize_line(raw)
        if _is_noise_line(line) or line in boilerplate:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def clean_pages(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Clean every page using document-wide boilerplate detection.

    Args:
        pages: ``(page_number, text)`` tuples from the loader.

    Returns:
        ``(page_number, cleaned_text)`` tuples, with empty pages dropped.
    """
    boilerplate = find_boilerplate_lines(pages)
    result: list[tuple[int, str]] = []
    for page_no, text in pages:
        cleaned = clean_text(text, boilerplate)
        if cleaned:
            result.append((page_no, cleaned))
    return result
