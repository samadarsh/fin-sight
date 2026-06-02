#!/usr/bin/env python3
"""Generate FinSight-Learning-Guide.pdf from docs/FinSight-Learning-Guide.md."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "FinSight-Learning-Guide.md"
OUTPUT = ROOT / "docs" / "FinSight-Learning-Guide.pdf"


def _ascii_safe(text: str) -> str:
    """Replace Unicode punctuation for core fonts when DejaVu is unavailable."""
    return (
        text.replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u2192", "->")
        .replace("\u2190", "<-")
        .replace("\u20b9", "Rs.")
        .replace("\u2026", "...")
    )


def _find_dejavu() -> tuple[str | None, str | None]:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return str(path), str(path)
    for site in Path(sys.prefix).glob("lib/python*/site-packages/fpdf/font"):
        reg, bold = site / "DejaVuSans.ttf", site / "DejaVuSans-Bold.ttf"
        if reg.exists() and bold.exists():
            return str(reg), str(bold)
    return None, None


class GuidePDF(FPDF):
    def __init__(self) -> None:
        super().__init__(format="A4", unit="mm")
        self.set_auto_page_break(auto=True, margin=18)
        self.toc_entries: list[tuple[int, str, int]] = []
        reg, bold = _find_dejavu()
        if reg and bold:
            self.add_font("DV", "", reg)
            if reg != bold:
                self.add_font("DV", "B", bold)
            else:
                self.add_font("DV", "B", reg)
            self.family = "DV"
            self._unicode = True
        else:
            self.family = "Helvetica"
            self._unicode = False
        self.body_size = 10
        self.code_size = 8

    def footer(self) -> None:
        if self.page_no() == 1:
            return
        self.set_y(-12)
        self.set_font(self.family, "", 9)
        self.set_text_color(130, 130, 130)
        self.cell(0, 8, f"FinSight Learning Guide  —  Page {self.page_no()}", align="C")

    def title_page(self) -> None:
        self.add_page()
        self.ln(45)
        self.set_font(self.family, "B", 24)
        self.set_text_color(25, 70, 130)
        self.cell(0, 12, "FinSight", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font(self.family, "B", 16)
        self.cell(0, 10, "Complete Learning Guide", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(10)
        self.set_font(self.family, "", 11)
        self.set_text_color(70, 70, 70)
        self.multi_cell(
            0,
            7,
            "Educational documentation for understanding and rebuilding "
            "a Retrieval-Augmented Generation (RAG) system for financial documents.",
            align="C",
        )
        self.ln(15)
        self.set_font(self.family, "", 10)
        self.multi_cell(
            0,
            6,
            "Topics: architecture, tools, data flow, code organization, "
            "workflows, challenges, best practices, and step-by-step rebuild guide.",
            align="C",
        )

    def toc_page(self) -> None:
        self.add_page()
        self.set_font(self.family, "B", 16)
        self.set_text_color(25, 70, 130)
        self.cell(0, 10, "Table of Contents", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(4)
        self.set_font(self.family, "", 10)
        self.set_text_color(40, 40, 40)
        for level, title, page in self.toc_entries:
            indent = 6 * (level - 1)
            prefix = "  " * (level - 1)
            y = self.get_y()
            self.set_xy(15 + indent, y)
            self.cell(160 - indent, 6, f"{prefix}{title}")
            self.set_xy(175, y)
            self.cell(20, 6, str(page), align="R")
            self.ln(6)

    def _txt(self, text: str) -> str:
        # Always normalize Rs. to avoid missing-glyph warnings in PDF fonts.
        text = text.replace("\u20b9", "Rs.")
        return text if self._unicode else _ascii_safe(text)

    def heading(self, level: int, text: str) -> None:
        sizes = {1: 15, 2: 13, 3: 11}
        if level <= 2 and self.get_y() > 250:
            self.add_page()
        self.ln(5 if level == 1 else 3)
        self.set_font(self.family, "B", sizes.get(level, 11))
        color = (25, 70, 130) if level == 1 else (40, 40, 40)
        self.set_text_color(*color)
        self.multi_cell(0, 7, self._txt(text))
        if level <= 2:
            self.toc_entries.append((level, text, self.page_no()))
        self.ln(1)

    def para(self, text: str) -> None:
        self.set_font(self.family, "", self.body_size)
        self.set_text_color(35, 35, 35)
        clean = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        clean = re.sub(r"`(.+?)`", r"\1", clean)
        self.multi_cell(0, 5.5, self._txt(clean))
        self.ln(2)

    def bullet(self, text: str) -> None:
        self.set_font(self.family, "", self.body_size)
        self.set_text_color(35, 35, 35)
        clean = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        clean = re.sub(r"`(.+?)`", r"\1", clean)
        self.multi_cell(0, 5.5, self._txt(f"  •  {clean}"))
        self.ln(1)

    def code(self, block: str) -> None:
        self.set_fill_color(242, 244, 248)
        self.set_font(self.family, "", self.code_size)
        self.set_text_color(25, 25, 25)
        for line in block.splitlines():
            if self.get_y() > 275:
                self.add_page()
            self.cell(
                0,
                4.5,
                "  " + self._txt(line),
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
                fill=True,
            )
        self.ln(3)

    def table(self, rows: list[list[str]]) -> None:
        if not rows:
            return
        n = len(rows[0])
        w = (self.w - 30) / n
        for ri, row in enumerate(rows):
            if self.get_y() > 265:
                self.add_page()
            self.set_font(self.family, "B" if ri == 0 else "", self.code_size)
            fill = ri == 0
            if fill:
                self.set_fill_color(225, 232, 245)
            for cell in row:
                txt = self._txt(cell[:45] + ("…" if len(cell) > 45 else ""))
                self.cell(w, 6, txt, border=1, fill=fill)
            self.ln(6)
        self.ln(2)


def render(md: str, pdf: GuidePDF) -> None:
    lines = md.splitlines()
    i = 0
    in_code = False
    code_lines: list[str] = []
    table_rows: list[list[str]] = []

    while i < len(lines):
        line = lines[i]

        if line.strip().startswith("```"):
            if in_code:
                pdf.code("\n".join(code_lines))
                code_lines, in_code = [], False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if line.startswith("|") and "|" in line[1:]:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(set(c) <= {"-", " ", ":"} for c in cells):
                table_rows.append(cells)
            i += 1
            continue
        if table_rows:
            pdf.table(table_rows)
            table_rows = []

        if line.startswith("# "):
            pdf.heading(1, line[2:].strip())
        elif line.startswith("## "):
            pdf.heading(2, line[3:].strip())
        elif line.startswith("### "):
            pdf.heading(3, line[4:].strip())
        elif line.startswith("- ") or line.startswith("* "):
            pdf.bullet(line[2:].strip())
        elif re.match(r"^\d+\.\s", line):
            pdf.bullet(re.sub(r"^\d+\.\s", "", line).strip())
        elif line.strip() in ("---", ""):
            pass
        else:
            pdf.para(line.strip())

        i += 1

    if table_rows:
        pdf.table(table_rows)


def main() -> int:
    if not SOURCE.exists():
        print(f"Missing source: {SOURCE}", file=sys.stderr)
        return 1

    text = SOURCE.read_text(encoding="utf-8")

    # Pass 1: collect TOC page numbers
    draft = GuidePDF()
    draft.title_page()
    render(text, draft)

    # Pass 2: title + TOC + body
    pdf = GuidePDF()
    pdf.toc_entries = draft.toc_entries
    pdf.title_page()
    pdf.toc_page()
    render(text, pdf)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUTPUT))
    kb = OUTPUT.stat().st_size // 1024
    print(f"Generated: {OUTPUT} ({kb} KB, {pdf.page_no()} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
