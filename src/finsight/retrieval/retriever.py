"""Retrieval: embed a question and fetch top-k chunks from ChromaDB."""

from config.settings import get_settings
from src.finsight.embeddings.base import EmbeddingProvider
from src.finsight.embeddings.factory import get_embedder
from src.finsight.models import Source
from src.finsight.vectorstore.chroma_store import ChromaStore

_FINANCIAL_KEYWORDS = (
    "revenue",
    "growth",
    "sales",
    "ebitda",
    "margin",
    "profit",
    "earnings",
    "turnover",
    "income",
    "capex",
    "roe",
    "roce",
    "net profit",
    "pat",
    "ebit",
    "operating income",
    "yoy",
    "year on year",
    "year-on-year",
    "top line",
    "bottom line",
)


def retrieve(
    question: str,
    *,
    k: int | None = None,
    filters: dict | None = None,
    embedder: EmbeddingProvider | None = None,
    store: ChromaStore | None = None,
) -> list[Source]:
    """Retrieve the most relevant chunks for a question."""
    settings = get_settings()
    embedder = embedder or get_embedder()
    store = store or ChromaStore()
    store.ensure_embedding_compatible(embedder)
    k = k or settings.retrieval_top_k

    query_embedding = embedder.embed_query(question)
    return store.query(query_embedding, k=k, filters=filters)


def _is_financial_question(question: str) -> bool:
    q = question.lower()
    return any(keyword in q for keyword in _FINANCIAL_KEYWORDS)


def _fy_label(year: int) -> str:
    """Indian financial year label, e.g. 2025 -> '2024-25'."""
    return f"{year - 1}-{str(year)[2:]}"


def _company_years(store: ChromaStore, companies: list[str]) -> dict[str, int | None]:
    """Map company name to reporting year from indexed documents."""
    years: dict[str, int | None] = {company: None for company in companies}
    for doc in store.list_documents():
        company = doc.get("company")
        if company in years:
            years[company] = doc.get("year")
    return years


def _queries_for_company(company: str, question: str, year: int | None = None) -> list[str]:
    """Build one or more search queries for a company in comparison mode."""
    queries = [f"{company} {question}"]
    if _is_financial_question(question):
        if year:
            queries.append(
                f"{company} revenue from operations growth FY {_fy_label(year)} rupee crore"
            )
        else:
            queries.append(f"{company} revenue from operations growth rupee crore")
    return queries


def retrieve_per_company(
    question: str,
    companies: list[str],
    *,
    k_per_company: int | None = None,
    embedder: EmbeddingProvider | None = None,
    store: ChromaStore | None = None,
) -> list[Source]:
    """Retrieve top chunks separately for each company.

    For financial questions, runs a second targeted query per company (e.g.
    ``"IOC revenue from operations... rupee crore"``) and keeps the best-scoring
    chunks. This avoids comparison wording pulling accounting-policy pages
    instead of financial tables.
    """
    settings = get_settings()
    embedder = embedder or get_embedder()
    store = store or ChromaStore()
    store.ensure_embedding_compatible(embedder)
    k_per_company = k_per_company or settings.comparison_k_per_company

    results: list[Source] = []
    best_by_page: dict[tuple[str, str, int], Source] = {}
    company_years = _company_years(store, companies)

    for company in companies:
        for scoped_query in _queries_for_company(company, question, company_years.get(company)):
            query_embedding = embedder.embed_query(scoped_query)
            hits = store.query(
                query_embedding,
                k=k_per_company,
                filters={"company": company},
            )
            for hit in hits:
                key = (hit.company, hit.source_file, hit.page)
                existing = best_by_page.get(key)
                if existing is None or hit.score > existing.score:
                    best_by_page[key] = hit

    per_company: dict[str, list[Source]] = {company: [] for company in companies}
    for source in best_by_page.values():
        per_company[source.company].append(source)

    for company in companies:
        ranked = sorted(per_company[company], key=lambda s: s.score, reverse=True)
        results.extend(ranked[:k_per_company])

    results.sort(key=lambda source: source.score, reverse=True)
    return results
