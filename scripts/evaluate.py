#!/usr/bin/env python3
"""Evaluate FinSight on a question set with known answer pages.

Ingest the PDFs the questions refer to first (scripts/ingest.py), then:

    python scripts/evaluate.py eval/questions.jsonl                   # full run (uses the LLM)
    python scripts/evaluate.py eval/questions.jsonl --retrieval-only  # retrieval only, no LLM cost

Writes per-question results and a summary to eval/results/ and prints a Markdown table.
See eval/README.md for the question format and what each metric means.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import get_settings  # noqa: E402
from src.finsight.errors import GeminiQuotaError  # noqa: E402
from src.finsight.evaluation import (  # noqa: E402
    load_items,
    report_markdown,
    score,
    summarize,
)
from src.finsight.pipeline.query_pipeline import answer_question  # noqa: E402
from src.finsight.retrieval.retriever import retrieve  # noqa: E402
from src.finsight.vectorstore.chroma_store import ChromaStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate FinSight on a question set")
    parser.add_argument("questions", type=Path, help="JSONL file of questions (see eval/README.md)")
    parser.add_argument(
        "--k", type=int, default=None, help="Chunks to retrieve (default: settings)"
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Score retrieval only; skips the LLM (free and fast)",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "results")
    parser.add_argument("--label", default=None, help="Name for this run in the report")
    args = parser.parse_args()

    settings = get_settings()
    k = args.k or settings.retrieval_top_k
    items = load_items(args.questions)
    store = ChromaStore()
    if store.count() == 0:
        print(
            "The vector store is empty. Ingest the PDFs first (scripts/ingest.py).", file=sys.stderr
        )
        return 1

    mode = "retrieval only" if args.retrieval_only else f"{settings.llm_provider} LLM"
    label = args.label or f"FinSight · {mode} · {settings.embedding_provider} embeddings · k={k}"
    print(f"{label}\n{len(items)} questions\n", flush=True)

    results = []
    for i, item in enumerate(items, start=1):
        filters = {"company": item.company} if item.company and not item.compare else None
        started = time.perf_counter()
        sources, answer, error = [], None, None
        try:
            if args.retrieval_only:
                sources = retrieve(item.question, k=k, filters=filters, store=store)
            else:
                response = answer_question(
                    item.question,
                    k=k,
                    filters=filters,
                    compare=item.compare,
                    companies=item.companies,
                    store=store,
                )
                sources, answer = response.sources, response.answer
        except GeminiQuotaError as exc:
            # A quota limit stops the run; later questions would all fail the same way.
            print(f"\nStopped: {exc}", file=sys.stderr)
            break
        except Exception as exc:  # one bad question shouldn't end the run
            error = f"{type(exc).__name__}: {exc}"
        result = score(item, sources, answer, time.perf_counter() - started, error)
        results.append(result)

        marks = "hit " if result.retrieval_hit else "MISS"
        if result.citation_correct is not None:
            marks += "  cite ok" if result.citation_correct else "  cite WRONG"
        if result.facts_total and answer is not None:
            marks += f"  facts {result.facts_found}/{result.facts_total}"
        print(f"[{i:>3}/{len(items)}] {marks:<32} {item.id}" + (f"  ({error})" if error else ""))

    if not results:
        return 1
    summary = summarize(results, k)
    summary.update(
        {
            "label": label,
            "llm": None if args.retrieval_only else settings.llm_provider,
            "embeddings": settings.embedding_provider,
        }
    )

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + (
        "-retrieval" if args.retrieval_only else "-full"
    )
    (args.out / f"{stamp}-results.jsonl").write_text(
        "".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in results), encoding="utf-8"
    )
    (args.out / f"{stamp}-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table = report_markdown(summary, label=label)
    (args.out / f"{stamp}-report.md").write_text(table, encoding="utf-8")

    print("\n" + table)
    print(f"Saved to {args.out}/{stamp}-*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
