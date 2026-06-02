#!/usr/bin/env python3
"""CLI helper to ingest a PDF into the vector store.

Examples:
    # Fresh ingest (clears any existing chunks for this file first)
    python scripts/ingest.py data/documents/annual-report-2025-2026.pdf \\
        --company TCS --doc-type annual_report --year 2026 --replace

    # Resume after hitting a rate limit (keeps existing chunks, skips done ones)
    python scripts/ingest.py data/documents/annual-report-2025-2026.pdf \\
        --company TCS --doc-type annual_report --year 2026
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.finsight.errors import GeminiQuotaError
from src.finsight.pipeline.ingest_pipeline import ingest_document  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a PDF into FinSight")
    parser.add_argument("pdf", type=Path, help="Path to the PDF file")
    parser.add_argument("--company", required=True, help="Company name, e.g. TCS")
    parser.add_argument(
        "--doc-type",
        required=True,
        choices=["annual_report", "transcript", "presentation", "filing"],
    )
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--quarter", default=None)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing chunks for this file before ingesting",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-embed all chunks even if some already exist (use with --replace)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        result = ingest_document(
            args.pdf,
            company=args.company,
            doc_type=args.doc_type,
            year=args.year,
            quarter=args.quarter,
            replace_existing=args.replace,
            resume=not args.no_resume,
        )
    except GeminiQuotaError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print()
    print(f"file    : {result.source_file}")
    print(f"pages   : {result.pages_processed}")
    print(f"added   : {result.chunks_added}")
    print(f"skipped : {result.chunks_skipped}")
    print(f"total   : {result.total_chunks_in_store} chunks in store")
    if result.partial:
        print()
        print("Stopped early (rate limit). Wait for quota to reset, then re-run")
        print("the same command WITHOUT --replace to resume.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
