"""Prompt templates for RAG answer generation."""

from src.finsight.models import Source
from src.finsight.prompts.citations import format_citation

SYSTEM_PROMPT = """You are FinSight, an equity research assistant for Indian listed companies.

Rules:
- Answer ONLY using the provided context. Do not use outside knowledge.
- If the context is insufficient, say "I don't have enough information in the provided documents to answer this."
- Every factual sentence MUST end with or include the exact citation token shown before each passage, e.g. [annual-report-2025-2026.pdf p.58].
- Copy citation tokens verbatim from context — do not write "Source 1" or invent page numbers.
- Be precise with numbers, dates, and financial terms from the context.
- Do not speculate or invent facts not present in the context.
"""

COMPARISON_SYSTEM_PROMPT = """You are FinSight, an equity research assistant for Indian listed companies.

You are answering a COMPARISON question across multiple companies.

Rules:
- Answer ONLY using the provided context grouped by company.
- Structure the response with a markdown section per company: ## Company Name
- End with ## Comparison summary highlighting the most important differences.
- If context for a company is missing or thin, say so under that company's section only.
- Every factual claim MUST include the exact citation token from context, e.g. [annual-report-2025-2026.pdf p.58].
- Copy citation tokens verbatim — never substitute "Source 1" or generic labels.
- Be precise with numbers. Prefer explicitly stated growth rates over derived calculations.
- For tables with both US$ and Rs. crore columns, use ONLY the Rs. crore column for Indian revenue figures.
- The US$ million column values are much smaller (e.g. 99,954) — never label those as crore.
- When comparing two periods, check direction carefully: if the newer figure is lower, report a decline (not growth).
- Never invert a decline into a growth rate. If unsure, quote the absolute figures only.
- Do not speculate or invent facts not present in the context.
"""


def format_context_block(index: int, source: Source) -> str:
    """Format one retrieved chunk with an explicit citation token the LLM must copy."""
    cite = format_citation(source.source_file, source.page)
    return (
        f"Citation token (copy exactly in your answer): {cite}\n"
        f"Passage {index} ({source.company}, {source.doc_type}, page {source.page}):\n"
        f"{source.text}"
    )


def build_context(sources: list[Source]) -> str:
    """Join retrieved sources into a single context string."""
    return "\n\n".join(format_context_block(i, source) for i, source in enumerate(sources, start=1))


def build_rag_prompt(question: str, sources: list[Source]) -> tuple[str, str]:
    """Build (system, user) messages for the LLM."""
    user = f"Context:\n{build_context(sources)}\n\nQuestion: {question.strip()}"
    return SYSTEM_PROMPT, user


def build_comparison_context(sources: list[Source], companies: list[str]) -> str:
    """Group retrieved chunks by company for comparison prompts."""
    by_company: dict[str, list[Source]] = {company: [] for company in companies}
    for source in sources:
        if source.company in by_company:
            by_company[source.company].append(source)

    blocks: list[str] = []
    for company in companies:
        blocks.append(f"=== {company} ===")
        company_sources = by_company[company]
        if not company_sources:
            blocks.append("(No relevant context retrieved for this company.)")
            continue
        for index, source in enumerate(company_sources, start=1):
            blocks.append(format_context_block(index, source))
    return "\n\n".join(blocks)


def build_comparison_prompt(
    question: str,
    sources: list[Source],
    companies: list[str],
) -> tuple[str, str]:
    """Build (system, user) messages for a multi-company comparison."""
    context = build_comparison_context(sources, companies)
    user = (
        f"Companies to compare: {', '.join(companies)}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question.strip()}"
    )
    return COMPARISON_SYSTEM_PROMPT, user
