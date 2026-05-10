# Chat With Your Codebase

A free-first RAG developer assistant that indexes a public GitHub repository and lets you ask source-grounded questions about the code.

## What It Does

- Clones a GitHub repository.
- Loads supported code and documentation files.
- Splits files into searchable chunks with file and line metadata.
- Creates local HuggingFace embeddings by default.
- Stores vectors in FAISS.
- Answers questions with citations to exact files and line ranges.
- Builds a lightweight architecture summary.
- Suggests useful onboarding questions.
- Scans for simple risk signals like TODOs, broad exception handling, debug logs, and possible secret-like assignments.
- Serves a no-build browser UI from FastAPI.

## Quick Start

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app
```

Open:

```text
http://127.0.0.1:8000
```

If you need auto-reload while editing code, exclude generated folders:

```bash
uvicorn backend.main:app --reload --reload-exclude faiss_index/* --reload-exclude cloned_repos/*
```

Plain `--reload` watches generated files like `faiss_index/repos.json`; when that file changes during indexing, the server can restart and interrupt the background job.

## Free Mode

The default mode uses:

- `sentence-transformers/all-MiniLM-L6-v2` for local embeddings.
- FAISS for local vector search.
- A deterministic answer formatter that cites retrieved chunks.

No API key is required.

## Optional LLM Mode

Create `.env` from `.env.example` and add:

```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

When the key is present, chat responses use an LLM with the retrieved code context. If the call fails, the app falls back to free retrieval mode.

## API

```text
POST /api/index
GET  /api/repos
GET  /api/repos/{repo_id}
POST /api/repos/{repo_id}/chat
GET  /api/repos/{repo_id}/sources
GET  /api/repos/{repo_id}/risks
```

## Project Structure

```text
backend/
  main.py          FastAPI app and repo session orchestration
  repo_loader.py   GitHub cloning and file loading
  chunking.py      Language-aware chunking
  embeddings.py    Local/OpenAI embedding providers
  vector_store.py  FAISS index build/load/search
  insights.py      Free architecture, suggestions, risks, and answers
frontend/
  index.html       Static app UI served by FastAPI
```

## Resume Angle

Built a full-stack RAG developer assistant that indexes GitHub repositories, performs semantic code search with FAISS and local embeddings, and answers developer questions with source citations, architecture summaries, suggested onboarding prompts, and lightweight risk scanning.
