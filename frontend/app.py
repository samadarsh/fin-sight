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
    initial_sidebar_state="expanded",
)

DOC_TYPES = ["annual_report", "transcript", "presentation", "filing"]


def _client() -> FinSightClient:
    return FinSightClient(st.session_state.api_url)


def _init_state() -> None:
    st.session_state.setdefault("api_url", os.getenv("FINSIGHT_API_URL", "http://127.0.0.1:8000"))
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_upload", None)


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


def sidebar() -> tuple[str | None, list[str] | None, bool | None]:
    st.sidebar.title("FinSight")
    st.sidebar.caption("Financial document RAG")

    st.session_state.api_url = st.sidebar.text_input(
        "API URL",
        value=st.session_state.api_url,
        help="FinSight FastAPI server, e.g. http://127.0.0.1:8000",
    )

    client = _client()
    try:
        client.health()
        st.sidebar.success("API connected")
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"API unreachable: {exc}")
        st.sidebar.info(
            "Start the backend:\n"
            "`uvicorn api.main:app --reload --host 127.0.0.1 --port 8000 "
            "--reload-dir api --reload-dir src --reload-dir config`"
        )
        return None, None, None

    st.sidebar.divider()
    st.sidebar.subheader("Upload & ingest")

    uploaded = st.sidebar.file_uploader("PDF document", type=["pdf"])
    company = st.sidebar.text_input("Company", placeholder="TCS")
    doc_type = st.sidebar.selectbox("Document type", DOC_TYPES)
    year = st.sidebar.number_input("Year (optional)", min_value=1990, max_value=2100, value=2026)
    replace_existing = st.sidebar.checkbox("Replace existing chunks", value=False)

    if st.sidebar.button("Upload & ingest", use_container_width=True, disabled=uploaded is None):
        if not uploaded or not company.strip():
            st.sidebar.warning("Select a PDF and enter a company name.")
        else:
            with st.sidebar.status("Processing…", expanded=True) as status:
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
                except FinSightAPIError as exc:
                    status.update(label="Failed", state="error")
                    st.error(f"API error ({exc.status_code}): {exc.detail}")
                except Exception as exc:  # noqa: BLE001
                    status.update(label="Failed", state="error")
                    st.error(str(exc))

    st.sidebar.divider()
    st.sidebar.subheader("Indexed documents")

    documents: list[dict] = []
    try:
        documents = client.list_documents()
    except FinSightAPIError as exc:
        st.sidebar.error(exc.detail)

    if not documents:
        st.sidebar.caption("No documents indexed yet.")
    else:
        for doc in documents:
            label = (
                f"**{doc['company']}** · {doc['source_file']} · "
                f"{doc['chunks']} chunks"
            )
            st.sidebar.markdown(label)
            if st.sidebar.button(f"Delete {doc['source_file']}", key=f"del-{doc['source_file']}"):
                try:
                    client.delete_document(doc["source_file"])
                    st.sidebar.success(f"Deleted {doc['source_file']}")
                    st.rerun()
                except FinSightAPIError as exc:
                    st.sidebar.error(exc.detail)

    st.sidebar.divider()
    st.sidebar.subheader("Query settings")

    companies = sorted({doc["company"] for doc in documents})
    query_mode = st.sidebar.radio(
        "Mode",
        ["Auto", "Single company", "Compare companies"],
        help="Auto detects comparison questions like 'Compare TCS and IOC'",
    )

    company_filter: str | None = None
    compare_companies: list[str] | None = None
    compare_flag: bool | None = None

    if query_mode == "Single company":
        company_filter = st.sidebar.selectbox("Company", companies) if companies else None
    elif query_mode == "Compare companies":
        compare_companies = st.sidebar.multiselect("Companies", companies, default=companies[:2])
        compare_flag = True

    return company_filter, compare_companies, compare_flag


def main() -> None:
    _init_state()
    company_filter, compare_companies, compare_flag = sidebar()

    st.title("FinSight")
    st.markdown(
        "Ask questions about **annual reports**, **earnings transcripts**, and other "
        "financial filings. Answers include page-level citations."
    )

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
                result = _client().query(payload)
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
