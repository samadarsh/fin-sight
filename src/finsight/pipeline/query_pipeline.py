"""Query pipeline: retrieve context and generate an answer."""

from config.settings import get_settings
from src.finsight.embeddings.base import EmbeddingProvider
from src.finsight.llm.base import LLMProvider
from src.finsight.llm.factory import get_llm
from src.finsight.models import QueryResponse
from src.finsight.prompts.templates import build_comparison_prompt, build_rag_prompt
from src.finsight.retrieval.comparison import (
    detect_companies_in_question,
    indexed_companies,
    is_comparison_query,
    resolve_comparison_companies,
)
from src.finsight.retrieval.retriever import retrieve, retrieve_per_company
from src.finsight.vectorstore.chroma_store import ChromaStore

_NO_CONTEXT_ANSWER = (
    "I don't have any relevant documents indexed yet. "
    "Ingest a PDF first, then ask your question again."
)


def _should_compare(
    question: str,
    store: ChromaStore,
    *,
    compare: bool | None,
    companies: list[str] | None,
    filters: dict | None,
) -> bool:
    if compare is True:
        return True
    if compare is False:
        return False
    if filters and filters.get("company"):
        return False
    if companies and len(companies) >= 2:
        return True
    known = indexed_companies(store)
    detected = detect_companies_in_question(question, known)
    return is_comparison_query(question, detected)


def preview_query_mode(
    question: str,
    *,
    compare: bool | None = None,
    companies: list[str] | None = None,
    filters: dict | None = None,
    store: ChromaStore | None = None,
) -> str:
    """Human-readable mode label for CLI logging."""
    store = store or ChromaStore()
    if _should_compare(question, store, compare=compare, companies=companies, filters=filters):
        resolved = resolve_comparison_companies(question, store, companies)
        return f"comparison ({', '.join(resolved)})"
    if filters and filters.get("company"):
        return f"standard (filter: {filters['company']})"
    return "standard"


def answer_question(
    question: str,
    *,
    k: int | None = None,
    k_per_company: int | None = None,
    filters: dict | None = None,
    compare: bool | None = None,
    companies: list[str] | None = None,
    embedder: EmbeddingProvider | None = None,
    llm: LLMProvider | None = None,
    store: ChromaStore | None = None,
) -> QueryResponse:
    """Answer a question using retrieved document context.

    Args:
        question: User question.
        k: Number of chunks to retrieve (standard mode).
        k_per_company: Chunks per company in comparison mode.
        filters: Optional metadata filter passed to retrieval (standard mode).
        compare: Force comparison mode on/off; ``None`` auto-detects.
        companies: Explicit company list for comparison mode.
        embedder: Optional embedding provider.
        llm: Optional LLM provider.
        store: Optional vector store.

    Returns:
        ``QueryResponse`` with answer text and cited sources.
    """
    settings = get_settings()
    store = store or ChromaStore()
    k = k or settings.retrieval_top_k

    if store.count() == 0:
        return QueryResponse(answer=_NO_CONTEXT_ANSWER, sources=[])

    llm = llm or get_llm()

    use_compare = _should_compare(
        question, store, compare=compare, companies=companies, filters=filters
    )

    if use_compare:
        resolved = resolve_comparison_companies(question, store, companies)
        if len(resolved) < 2:
            indexed = indexed_companies(store)
            return QueryResponse(
                answer=(
                    "Comparison needs at least 2 companies. "
                    f"Indexed: {', '.join(indexed) or 'none'}. "
                    "Use --companies TCS,IOC or name both companies in your question."
                ),
                sources=[],
            )
        sources = retrieve_per_company(
            question,
            resolved,
            k_per_company=k_per_company,
            embedder=embedder,
            store=store,
        )
        if not sources:
            return QueryResponse(
                answer="I don't have enough information in the provided documents to answer this.",
                sources=[],
            )
        system, user = build_comparison_prompt(question, sources, resolved)
    else:
        sources = retrieve(
            question,
            k=k,
            filters=filters,
            embedder=embedder,
            store=store,
        )
        if not sources:
            return QueryResponse(
                answer="I don't have enough information in the provided documents to answer this.",
                sources=[],
            )
        system, user = build_rag_prompt(question, sources)

    answer = llm.generate(system, user)
    return QueryResponse(answer=answer, sources=sources)
