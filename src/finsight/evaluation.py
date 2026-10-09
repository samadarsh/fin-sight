"""Scoring for the FinSight evaluation (``scripts/evaluate.py``).

An evaluation item is one question with the page(s) that hold its answer and, optionally,
the key facts a correct answer must state. A run reports:

- retrieval hit@k: did any retrieved chunk come from an expected page?
- citation accuracy: does the answer cite at least one expected page?
- fact match: does the answer contain every key fact?
- refusal accuracy: for questions the documents can't answer, did it decline instead of guessing?

Everything here is pure (no model or store calls), so it is unit-tested directly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.finsight.models import Source

# Same token the pipeline writes: [annual-report-2025.pdf p.58]. The file name excludes "[" so a
# model's stray double bracket ("[[file.pdf p.58]") still counts.
_CITATION_RE = re.compile(r"\[([^\[\]\s]+\.pdf) p\.(\d+)\]", re.IGNORECASE)

# How a declined answer reads. The prompt asks for the first phrasing; models paraphrase it.
_REFUSAL_RE = re.compile(
    r"don.t have enough information|not (?:mentioned|stated|provided|available|disclosed|found)"
    r"|(?:no|not enough) information|(?:do(?:es)? not|doesn.t) (?:contain|mention|include|provide)"
    r"|cannot (?:be )?(?:answer|determine)|unable to (?:answer|determine|find)",
    re.IGNORECASE,
)


@dataclass
class EvalItem:
    id: str
    question: str
    # Where the answer is: (file name, page numbers) pairs. Comparisons span several files.
    expected: list[tuple[str, list[int]]] = field(default_factory=list)
    # Each entry is a fact the answer must contain; a list means "any one of these".
    answer_contains: list[str | list[str]] = field(default_factory=list)
    company: str | None = None
    compare: bool | None = None
    companies: list[str] | None = None
    # The documents don't answer this; the right response is to decline.
    unanswerable: bool = False


@dataclass
class ItemResult:
    id: str
    question: str
    retrieval_hit: bool | None  # None for unanswerable questions
    first_hit_rank: int | None
    cited_pages: list[str]
    citation_correct: bool | None  # None in retrieval-only runs and for unanswerable questions
    facts_found: int
    facts_total: int
    latency_s: float
    unanswerable: bool = False
    refused: bool | None = None  # None in retrieval-only runs
    answer: str | None = None
    error: str | None = None

    @property
    def facts_ok(self) -> bool | None:
        return (
            None
            if self.facts_total == 0 or self.answer is None
            else (self.facts_found == self.facts_total)
        )


def _parse_expected(raw: dict) -> list[tuple[str, list[int]]]:
    if raw.get("unanswerable"):
        return []
    if "expected" in raw:
        return [(e["source_file"], [int(p) for p in e["pages"]]) for e in raw["expected"]]
    return [(raw["source_file"], [int(p) for p in raw["pages"]])]


def load_items(path: Path) -> list[EvalItem]:
    """Read a JSONL question file; blank lines and lines starting with # are skipped.

    Each item names where its answer is either as ``source_file`` + ``pages`` or, for answers
    spread over several files, as ``expected: [{source_file, pages}, ...]``. Items marked
    ``unanswerable`` need neither.
    """
    items: list[EvalItem] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            raw = json.loads(line)
            expected = _parse_expected(raw)
            if not expected and not raw.get("unanswerable"):
                raise ValueError("no expected pages")
            items.append(
                EvalItem(
                    id=str(raw.get("id") or f"q{n}"),
                    question=raw["question"],
                    expected=expected,
                    answer_contains=raw.get("answer_contains", []),
                    company=raw.get("company"),
                    compare=raw.get("compare"),
                    companies=raw.get("companies"),
                    unanswerable=bool(raw.get("unanswerable", False)),
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{n}: invalid evaluation item ({exc})") from exc
    if not items:
        raise ValueError(f"{path}: no evaluation items found")
    return items


def _same_file(a: str, b: str) -> bool:
    return Path(a).name.lower() == Path(b).name.lower()


def _on_expected_page(item: EvalItem, file: str, page: int) -> bool:
    return any(_same_file(file, f) and page in pages for f, pages in item.expected)


def first_hit_rank(item: EvalItem, sources: list[Source]) -> int | None:
    """1-based rank of the first retrieved chunk from an expected page, or None."""
    for rank, src in enumerate(sources, start=1):
        if _on_expected_page(item, src.source_file, src.page):
            return rank
    return None


def cited_pages(answer: str) -> list[tuple[str, int]]:
    """Every ``[file.pdf p.N]`` citation in an answer, in order, without repeats."""
    seen: list[tuple[str, int]] = []
    for file, page in _CITATION_RE.findall(answer):
        key = (file, int(page))
        if key not in seen:
            seen.append(key)
    return seen


def citation_correct(item: EvalItem, answer: str) -> bool:
    """True when the answer cites at least one page that holds the answer."""
    return any(_on_expected_page(item, file, page) for file, page in cited_pages(answer))


def refused(answer: str) -> bool:
    """True when the answer declines to answer from the documents."""
    return bool(_REFUSAL_RE.search(answer))


def _normalize(text: str) -> str:
    # Case, spacing, the rupee sign and thousands separators shouldn't decide a match:
    # "₹1,234 Cr" contains "1234 cr".
    text = text.lower().replace("₹", "")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def facts_found(item: EvalItem, answer: str) -> int:
    """How many of the item's key facts appear in the answer."""
    norm = _normalize(answer)
    found = 0
    for fact in item.answer_contains:
        options = fact if isinstance(fact, list) else [fact]
        if any(_normalize(opt) in norm for opt in options):
            found += 1
    return found


def score(
    item: EvalItem,
    sources: list[Source],
    answer: str | None,
    latency_s: float,
    error: str | None = None,
) -> ItemResult:
    """Score one question from what the pipeline retrieved and answered."""
    rank = None if item.unanswerable else first_hit_rank(item, sources)
    answered = answer is not None
    return ItemResult(
        id=item.id,
        question=item.question,
        retrieval_hit=None if item.unanswerable else rank is not None,
        first_hit_rank=rank,
        cited_pages=[f"{f} p.{p}" for f, p in cited_pages(answer or "")],
        citation_correct=(
            citation_correct(item, answer) if answered and not item.unanswerable else None
        ),
        facts_found=facts_found(item, answer) if answered else 0,
        facts_total=len(item.answer_contains),
        latency_s=round(latency_s, 3),
        unanswerable=item.unanswerable,
        refused=refused(answer) if answered else None,
        answer=answer,
        error=error,
    )


def _pct(n: int, d: int) -> float | None:
    return None if d == 0 else round(100 * n / d, 1)


def summarize(results: list[ItemResult], k: int) -> dict:
    """Headline numbers for a run. Percentages are over the questions each metric applies to."""
    ran = [r for r in results if r.error is None]
    answerable = [r for r in ran if not r.unanswerable]
    traps = [r for r in ran if r.unanswerable and r.answer is not None]
    answered = [r for r in answerable if r.answer is not None]
    with_facts = [r for r in answered if r.facts_total]
    hits = [r for r in answerable if r.retrieval_hit]
    latencies = sorted(r.latency_s for r in ran)
    return {
        "questions": len(results),
        "errors": len(results) - len(ran),
        "k": k,
        "retrieval_hit_at_k": _pct(len(hits), len(answerable)),
        "mrr": (
            round(sum(1 / r.first_hit_rank for r in hits) / len(answerable), 3)
            if answerable
            else None
        ),
        "citation_accuracy": _pct(sum(1 for r in answered if r.citation_correct), len(answered)),
        "fact_match": _pct(sum(1 for r in with_facts if r.facts_ok), len(with_facts)),
        "fact_match_questions": len(with_facts),
        # Answerable questions it declined: the cost of being cautious.
        "false_refusals": sum(1 for r in answered if r.refused),
        "refusal_accuracy": _pct(sum(1 for r in traps if r.refused), len(traps)),
        "refusal_questions": len(traps),
        "latency_median_s": latencies[len(latencies) // 2] if latencies else None,
        "latency_max_s": latencies[-1] if latencies else None,
    }


def report_markdown(summary: dict, *, label: str) -> str:
    """A short Markdown table of a run, for the README or a case study."""

    def fmt(value, suffix=""):
        return "–" if value is None else f"{value}{suffix}"

    rows = [
        ("Questions", fmt(summary["questions"])),
        (f"Retrieval hit@{summary['k']}", fmt(summary["retrieval_hit_at_k"], "%")),
        ("Mean reciprocal rank", fmt(summary["mrr"])),
        ("Citation accuracy", fmt(summary["citation_accuracy"], "%")),
        (
            f"Fact match ({summary['fact_match_questions']} questions)",
            fmt(summary["fact_match"], "%"),
        ),
    ]
    if summary["refusal_questions"]:
        rows.append(
            (
                f"Refusal accuracy ({summary['refusal_questions']} unanswerable questions)",
                fmt(summary["refusal_accuracy"], "%"),
            )
        )
        rows.append(("Answerable questions it declined", fmt(summary["false_refusals"])))
    rows.append(("Median latency", fmt(summary["latency_median_s"], " s")))
    lines = [f"**{label}**", "", "| Metric | Result |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    if summary["errors"]:
        lines += ["", f"_{summary['errors']} question(s) failed and are excluded._"]
    return "\n".join(lines) + "\n"
