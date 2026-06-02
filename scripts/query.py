#!/usr/bin/env python3
"""CLI helper to ask a question against the indexed documents."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.finsight.errors import GeminiQuotaError  # noqa: E402
from src.finsight.pipeline.query_pipeline import answer_question, preview_query_mode  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask FinSight a question")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--company", help="Filter by one company (standard mode only)")
    parser.add_argument(
        "--companies",
        help="Comma-separated companies for comparison, e.g. TCS,IOC",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Force comparison mode (retrieves chunks per company)",
    )
    parser.add_argument("--k", type=int, default=None, help="Number of chunks to retrieve")
    parser.add_argument(
        "--k-per-company",
        type=int,
        default=None,
        help="Chunks per company in comparison mode (default: 3)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show progress logs")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    companies = None
    if args.companies:
        companies = [part.strip() for part in args.companies.split(",") if part.strip()]

    compare = True if args.compare else None
    if companies and len(companies) >= 2:
        compare = True

    filters = None if compare else ({"company": args.company} if args.company else None)

    if args.verbose:
        mode = preview_query_mode(
            args.question,
            compare=compare,
            companies=companies,
            filters=filters,
        )
        print(f"Mode: {mode}", flush=True)
        print("Step 1/3: Embedding question and retrieving chunks…", flush=True)

    try:
        if args.verbose:
            print("Step 2/3: Generating answer with LLM…", flush=True)
        result = answer_question(
            args.question,
            k=args.k,
            k_per_company=args.k_per_company,
            filters=filters,
            compare=compare,
            companies=companies,
        )
    except GeminiQuotaError as exc:
        print(str(exc), file=sys.stderr)
        if exc.daily and exc.kind == "embedding":
            print(
                "\nTip: switch to local embeddings with EMBEDDING_PROVIDER=local in .env",
                file=sys.stderr,
            )
        elif exc.daily and exc.kind == "llm":
            print(
                "\nTip: switch to a local LLM with LLM_PROVIDER=ollama in .env "
                "(requires Ollama installed).",
                file=sys.stderr,
            )
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Step 3/3: Done — {len(result.sources)} source(s) used.\n", flush=True)

    print(result.answer)
    print()
    if result.sources:
        print("Sources:")
        for i, source in enumerate(result.sources, start=1):
            print(
                f"  {i}. {source.company} | {source.source_file} p.{source.page} "
                f"(score={source.score})"
            )
            print(f"     {source.text[:180].replace(chr(10), ' ')}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
