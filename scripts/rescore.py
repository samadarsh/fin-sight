#!/usr/bin/env python3
"""Re-score a saved evaluation run against an updated question file, without the LLM.

When the question set changes (a new valid answer page, an extra accepted figure), the saved
answers from a full run are still good; only the scoring is out of date. This keeps each saved
answer and its latency, re-runs retrieval (local and cheap) and scores everything again:

    python scripts/rescore.py eval/results/20261010-032413-full-results.jsonl eval/questions.jsonl

Use the same vector store and k as the original run, or the retrieval numbers won't match it.
Questions missing from the saved run are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import get_settings  # noqa: E402
from src.finsight.evaluation import load_items, report_markdown, score, summarize  # noqa: E402
from src.finsight.pipeline.query_pipeline import retrieve_context  # noqa: E402
from src.finsight.vectorstore.chroma_store import ChromaStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-score a saved FinSight evaluation run")
    parser.add_argument(
        "results", type=Path, help="A *-results.jsonl file from scripts/evaluate.py"
    )
    parser.add_argument("questions", type=Path, help="The (updated) question file")
    parser.add_argument(
        "--k", type=int, default=None, help="k of the original run (default: settings)"
    )
    parser.add_argument("--label", default=None, help="Name for the re-scored run in the report")
    args = parser.parse_args()

    k = args.k or get_settings().retrieval_top_k
    saved = {}
    for line in args.results.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            saved[row["id"]] = row
    store = ChromaStore()
    if store.count() == 0:
        print(
            "The vector store is empty. Ingest the PDFs first (scripts/ingest.py).", file=sys.stderr
        )
        return 1

    results = []
    for item in load_items(args.questions):
        row = saved.get(item.id)
        if row is None:
            print(f"skipped {item.id}: not in the saved run", file=sys.stderr)
            continue
        filters = {"company": item.company} if item.company and not item.compare else None
        sources = retrieve_context(
            item.question,
            k=k,
            filters=filters,
            compare=item.compare,
            companies=item.companies,
            store=store,
        )
        results.append(score(item, sources, row.get("answer"), row["latency_s"], row.get("error")))

    if not results:
        print("No questions in common with the saved run.", file=sys.stderr)
        return 1
    summary = summarize(results, k)
    label = args.label or f"{args.results.name} · re-scored against {args.questions.name}"
    summary["label"] = label

    stem = args.results.name.removesuffix("-results.jsonl") + "-rescored"
    out = args.results.parent
    (out / f"{stem}-results.jsonl").write_text(
        "".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in results), encoding="utf-8"
    )
    (out / f"{stem}-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table = report_markdown(summary, label=label)
    (out / f"{stem}-report.md").write_text(table, encoding="utf-8")

    print(table)
    print(f"Saved to {out}/{stem}-*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
