"""FinSight Streamlit frontend."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.api_client import FinSightAPIError, FinSightClient

st.set_page_config(
    page_title="FinSight",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto",
)

DOC_TYPES = ["annual_report", "transcript", "presentation", "filing"]

_RESPONSIVE_CSS = """
<style>
    .block-container { padding-top: 1rem; max-width: 100%; }
    [data-testid="stSidebar"] { min-width: 280px; }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        word-wrap: break-word;
        overflow-wrap: anywhere;
    }
    @media (max-width: 768px) {
        [data-testid="stSidebar"] { min-width: 100%; }
    }
</style>
"""


def _client() -> FinSightClient:
    return FinSightClient(st.session_state.api_url)


def _init_state() -> None:
    st.session_state.setdefault("api_url", os.getenv("FINSIGHT_API_URL", "http://127.0.0.1:8000"))
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_upload", None)
    st.session_state.setdefault("documents", [])


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for index, source in enumerate(sources, start=1):
            st.markdown(
                f"**{index}.** {source['company']} · `{source['source_file']}` · "
                f"p.{source['page']} · score={source['score']}"
            )
            st.caption(source["text"][:400] + ("…" if len(source["text"]) > 400 else ""))


def _connection_panel() -> FinSightClient | None:
    st.sidebar.subheader("Connection")
    st.session_state.api_url = st.sidebar.text_input(
        "API URL",
        value=st.session_state.api_url,
        help="FinSight FastAPI server, e.g. http://127.0.0.1:8000",
    )
    client = _client()
    try:
        client.health()
        st.sidebar.success("API connected")
        return client
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"API unreachable: {exc}")
        st.sidebar.info(
            "Start the backend:\n"
            "`uvicorn api.main:app --reload --host 127.0.0.1 --port 8000 "
            "--reload-dir api --reload-dir src --reload-dir config`"
        )
        return None


def _upload_panel(client: FinSightClient) -> None:
    st.subheader("Upload & ingest")
    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("PDF document", type=["pdf"], key="pdf_upload")
        company = st.text_input("Company", placeholder="TCS", key="upload_company")
    with col2:
        doc_type = st.selectbox("Document type", DOC_TYPES, key="upload_doc_type")
        year = st.number_input(
            "Year", min_value=1990, max_value=2100, value=2026, key="upload_year"
        )
        replace_existing = st.checkbox(
            "Replace existing chunks", value=False, key="upload_replace"
        )

    if st.button(
        "Upload & ingest",
        use_container_width=True,
        disabled=uploaded is None,
        key="upload_btn",
    ):
        if not uploaded or not company.strip():
            st.warning("Select a PDF and enter a company name.")
            return
        with st.status("Processing…", expanded=True) as status:
            try:
                st.write("Uploading PDF…")
                uploaded.seek(0)
                up = client.upload_pdf(uploaded.name, uploaded)
                st.session_state.last_upload = up["source_file"]
                st.write(f"Ingesting `{up['source_file']}`…")
                result = client.ingest(
                    {
                        "source_file": up["source_file"],
                        "company": company.strip().upper(),
                        "doc_type": doc_type,
                        "year": int(year),
                        "replace_existing": replace_existing,
                        "resume": True,
                    }
                )
                if result.get("partial"):
                    status.update(label="Partial ingest — resume later", state="error")
                    st.error(
                        f"Rate limit hit. Saved {result['chunks_added']} chunks. "
                        "Re-run ingest without replace to resume."
                    )
                else:
                    status.update(label="Ingest complete", state="complete")
                    st.success(
                        f"Added {result['chunks_added']} chunks "
                        f"({result['total_chunks_in_store']} total in store)"
                    )
                    st.session_state.documents = client.list_documents()
            except FinSightAPIError as exc:
                status.update(label="Failed", state="error")
                st.error(f"API error ({exc.status_code}): {exc.detail}")
            except Exception as exc:  # noqa: BLE001
                status.update(label="Failed", state="error")
                st.error(str(exc))


def _documents_panel(client: FinSightClient) -> None:
    st.subheader("Indexed documents")
    try:
        st.session_state.documents = client.list_documents()
    except FinSightAPIError as exc:
        st.error(exc.detail)
        return

    documents = st.session_state.documents
    if not documents:
        st.caption("No documents indexed yet.")
        return

    for doc in documents:
        cols = st.columns([4, 1])
        with cols[0]:
            st.markdown(
                f"**{doc['company']}** · `{doc['source_file']}` · {doc['chunks']} chunks"
            )
        with cols[1]:
            if st.button("Delete", key=f"del-{doc['source_file']}"):
                try:
                    client.delete_document(doc["source_file"])
                    st.session_state.documents = client.list_documents()
                    st.rerun()
                except FinSightAPIError as exc:
                    st.error(exc.detail)


def _query_settings(documents: list[dict]) -> tuple[str | None, list[str] | None, bool | None]:
    companies = sorted({doc["company"] for doc in documents})
    query_mode = st.radio(
        "Query mode",
        ["Auto", "Single company", "Compare companies"],
        horizontal=True,
        help="Auto detects comparison questions like 'Compare TCS and IOC'",
    )

    company_filter: str | None = None
    compare_companies: list[str] | None = None
    compare_flag: bool | None = None

    if query_mode == "Single company":
        company_filter = st.selectbox("Company filter", companies) if companies else None
    elif query_mode == "Compare companies":
        compare_companies = st.multiselect("Companies", companies, default=companies[:2])
        compare_flag = True

    return company_filter, compare_companies, compare_flag


def main() -> None:
    _init_state()
    st.markdown(_RESPONSIVE_CSS, unsafe_allow_html=True)

    st.title("FinSight")
    st.caption("Ask questions about financial PDFs with page-level citations.")

    with st.sidebar:
        client = _connection_panel()

    tab_chat, tab_manage = st.tabs(["Chat", "Documents & upload"])

    company_filter: str | None = None
    compare_companies: list[str] | None = None
    compare_flag: bool | None = None

    with tab_manage:
        if client is None:
            st.info("Connect to the API using the sidebar to manage documents.")
        else:
            _upload_panel(client)
            st.divider()
            _documents_panel(client)

    with tab_chat:
        if client is None:
            st.info("Connect to the API using the sidebar to start chatting.")
            return

        if st.session_state.documents:
            company_filter, compare_companies, compare_flag = _query_settings(
                st.session_state.documents
            )
        else:
            try:
                st.session_state.documents = client.list_documents()
            except FinSightAPIError:
                st.session_state.documents = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    if message.get("mode"):
                        st.caption(f"Mode: {message['mode']}")
                    _render_sources(message.get("sources", []))

        question = st.chat_input("Ask a question about your documents…")
        if not question:
            return

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            payload: dict = {"question": question}
            if company_filter:
                payload["company"] = company_filter
            if compare_companies and len(compare_companies) >= 2:
                payload["companies"] = compare_companies
                payload["compare"] = True
            elif compare_flag is True:
                payload["compare"] = True

            try:
                with st.spinner("Searching documents and generating answer…"):
                    result = client.query(payload)
                st.markdown(result["answer"])
                st.caption(f"Mode: {result.get('mode', 'standard')}")
                _render_sources(result.get("sources", []))
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                        "mode": result.get("mode"),
                    }
                )
            except FinSightAPIError as exc:
                st.error(f"API error ({exc.status_code}): {exc.detail}")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))


if __name__ == "__main__":
    main()
