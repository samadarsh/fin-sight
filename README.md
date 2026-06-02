# FinSight

A Retrieval-Augmented Generation (RAG) system for financial documents — annual
reports, earnings call transcripts, investor presentations, and SEBI filings.
Upload PDFs, ask questions in natural language, and get answers grounded in the
source documents with page-level citations.

## Architecture

```
User → Streamlit UI → FastAPI → Ingestion → Embeddings → ChromaDB
                                                  ↓
                          Retriever → Prompt → LLM (Ollama/Gemini) → Answer + Sources
```

## Tech Stack

| Layer        | Choice                                |
| ------------ | ------------------------------------- |
| Backend API  | FastAPI + Uvicorn                     |
| PDF parsing  | PyMuPDF                               |
| Chunking     | LangChain recursive text splitter     |
| Embeddings   | Local `BAAI/bge-small-en-v1.5` (default) or Gemini |
| Vector DB    | ChromaDB (persistent, local)          |
| LLM          | Ollama `llama3:latest` (default) or Gemini |
| Frontend     | Streamlit                             |

## Setup

```bash
# 1. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Defaults use local embeddings + Ollama (no API key required).
# Start Ollama and pull the model: ollama pull llama3:latest
# For Gemini instead, set LLM_PROVIDER=gemini and add GEMINI_API_KEY.
```

## Run Ollama (default LLM)

```bash
ollama serve          # if not already running via the Ollama app
ollama pull llama3:latest
```

## Run the API

```bash
source .venv/bin/activate
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000 \
  --reload-dir api --reload-dir src --reload-dir config
```

Open http://127.0.0.1:8000/docs for interactive Swagger UI.

Example flow:

```bash
# Upload a PDF
curl -F "file=@data/documents/annual-report-2025-2026.pdf" http://127.0.0.1:8000/upload

# Ingest into the vector store
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_file":"annual-report-2025-2026.pdf","company":"TCS","doc_type":"annual_report","year":2026}'

# Ask a question
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What risks did TCS mention?","company":"TCS"}'
```

## Run the frontend

Requires the API server to be running (see above).

```bash
source .venv/bin/activate
streamlit run frontend/app.py
```

Opens http://localhost:8501 by default. Set `FINSIGHT_API_URL` if the API is not on port 8000.

## Project Structure

```
config/            App settings (Pydantic)
src/finsight/       Core library
  ingestion/        PDF loading, cleaning, chunking
  embeddings/        Embedding providers
  vectorstore/       ChromaDB wrapper
  retrieval/         Query -> top-k chunks
  llm/               LLM providers
  prompts/           Prompt templates
  pipeline/          Ingest & query orchestration
api/                FastAPI app
frontend/           Streamlit UI
data/documents/     Uploaded PDFs (gitignored)
storage/chroma/     Vector DB persistence (gitignored)
evaluation/         RAG evaluation harness
tests/              Unit & integration tests
```

## Status

- [x] Step 1 — Project skeleton + config
- [x] Step 2 — Ingestion (loader → cleaner → chunker)
- [x] Step 3 — Embeddings provider
- [x] Step 4 — Vector store
- [x] Step 5 — Ingest pipeline
- [x] Step 6 — LLM provider + prompts
- [x] Step 7 — Retrieval + query pipeline
- [x] Step 8 — FastAPI backend
- [x] Step 9 — Streamlit frontend
