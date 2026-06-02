"""PDF text extraction.

Uses PyMuPDF (``fitz``) to pull text out of a PDF, one entry per page.
Page numbers are 1-indexed to match how humans cite documents.
"""

from pathlib import Path

import fitz  # PyMuPDF


def load_pdf(path: str | Path) -> list[tuple[int, str]]:
    """Extract text from a PDF.

    Args:
        path: Path to the PDF file.

    Returns:
        A list of ``(page_number, text)`` tuples, page numbers 1-indexed.
        Pages that contain no extractable text are skipped (e.g. scanned
        images with no OCR layer).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages: list[tuple[int, str]] = []
    with fitz.open(path) as doc:
        for index, page in enumerate(doc):
            text = page.get_text("text")
            if text and text.strip():
                pages.append((index + 1, text))
    return pages
