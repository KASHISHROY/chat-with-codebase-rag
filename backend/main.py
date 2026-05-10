"""
FastAPI entry point for Chat With Your Codebase.

Run the app:
    uvicorn backend.main:app --reload

The backend stays free by default:
    - local HuggingFace embeddings
    - FAISS vector search
    - deterministic source-grounded answers when no LLM key is configured
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from threading import Lock
from typing import Any

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from .chunking import chunk_documents
    from .embeddings import embed_chunks, get_embedding_model
    from .insights import (
        create_onboarding_summary,
        find_risk_signals,
        generate_free_answer,
        hybrid_rerank,
        suggest_questions,
        summarize_architecture,
    )
    from .repo_loader import clone_repository, load_code_files
    from .vector_store import build_faiss_index, load_faiss_index, search_similar_chunks
except ImportError:  # Allows `python backend/main.py` during quick local testing.
    from chunking import chunk_documents
    from embeddings import embed_chunks, get_embedding_model
    from insights import (
        create_onboarding_summary,
        find_risk_signals,
        generate_free_answer,
        hybrid_rerank,
        suggest_questions,
        summarize_architecture,
    )
    from repo_loader import clone_repository, load_code_files
    from vector_store import build_faiss_index, load_faiss_index, search_similar_chunks


load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
INDEX_ROOT = ROOT_DIR / "faiss_index"
CLONE_ROOT = ROOT_DIR / "cloned_repos"
MANIFEST_FILE = INDEX_ROOT / "repos.json"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")


app = FastAPI(
    title="Chat With Your Codebase",
    description="Free local RAG assistant for asking questions about GitHub repositories.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class IndexRequest(BaseModel):
    github_url: str = Field(..., min_length=8)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2)
    top_k: int = Field(5, ge=1, le=12)


class RepoSession(BaseModel):
    repo_id: str
    github_url: str
    status: str
    message: str = ""
    progress: int = 0
    created_at: float
    updated_at: float
    file_count: int = 0
    chunk_count: int = 0
    index_dir: str
    clone_dir: str
    architecture: dict[str, Any] | None = None
    suggestions: list[str] = []
    onboarding: list[str] = []
    error: str | None = None


repo_lock = Lock()
indexing_lock = Lock()
repos: dict[str, dict[str, Any]] = {}
runtime_cache: dict[str, dict[str, Any]] = {}


def now() -> float:
    return time.time()


def make_repo_id(github_url: str) -> str:
    normalized = github_url.strip().rstrip("/")
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def load_manifest() -> None:
    INDEX_ROOT.mkdir(exist_ok=True)
    CLONE_ROOT.mkdir(exist_ok=True)
    if not MANIFEST_FILE.exists():
        return
    try:
        data = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    for repo in data.values():
        if repo.get("status") in {"queued", "indexing"}:
            repo["status"] = "failed"
            repo["message"] = "Indexing was interrupted. Click Index Repo again to retry."
            repo["error"] = "Previous server process stopped before indexing finished."
            repo["progress"] = 100
    with repo_lock:
        repos.update(data)
    save_manifest()


def save_manifest() -> None:
    INDEX_ROOT.mkdir(exist_ok=True)
    with repo_lock:
        MANIFEST_FILE.write_text(json.dumps(repos, indent=2), encoding="utf-8")


def update_repo(repo_id: str, **updates: Any) -> None:
    with repo_lock:
        repo = repos[repo_id]
        repo.update(updates)
        repo["updated_at"] = now()
    save_manifest()


def get_repo_or_404(repo_id: str) -> dict[str, Any]:
    repo = repos.get(repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository session not found.")
    return repo


def ensure_ready(repo_id: str) -> dict[str, Any]:
    repo = get_repo_or_404(repo_id)
    if repo["status"] != "ready":
        raise HTTPException(status_code=409, detail=f"Repository is {repo['status']}: {repo.get('message', '')}")
    return repo


def load_runtime(repo_id: str) -> dict[str, Any]:
    if repo_id in runtime_cache:
        return runtime_cache[repo_id]

    repo = ensure_ready(repo_id)
    embedding_model = get_embedding_model(provider=EMBEDDING_PROVIDER)
    index, metadata = load_faiss_index(repo["index_dir"])
    runtime_cache[repo_id] = {
        "embedding_model": embedding_model,
        "index": index,
        "metadata": metadata,
    }
    return runtime_cache[repo_id]


def refresh_architecture_if_needed(repo_id: str) -> None:
    repo = repos.get(repo_id)
    if not repo or repo.get("status") != "ready":
        return
    architecture = repo.get("architecture") or {}
    if architecture.get("code_facts"):
        return

    try:
        _, metadata = load_faiss_index(repo["index_dir"])
    except Exception:
        return

    by_file: dict[str, list[dict]] = {}
    for chunk in metadata:
        by_file.setdefault(chunk["file_path"], []).append(chunk)

    documents = []
    for file_path, chunks in by_file.items():
        ordered = sorted(chunks, key=lambda item: item.get("chunk_index", 0))
        documents.append(
            {
                "file_path": file_path,
                "content": "\n".join(item.get("content", "") for item in ordered),
            }
        )

    updated_architecture = summarize_architecture(documents, metadata)
    update_repo(
        repo_id,
        architecture=updated_architecture,
        onboarding=create_onboarding_summary(updated_architecture),
        suggestions=suggest_questions(updated_architecture),
    )


def build_generation_prompt(question: str, results: list[dict], architecture: dict | None = None) -> str:
    architecture = architecture or {}
    facts = architecture.get("code_facts") or {}
    fact_summary = json.dumps(
        {
            "frameworks": architecture.get("frameworks", []),
            "entrypoints": architecture.get("entrypoints", []),
            "routes": facts.get("routes", [])[:12],
            "forms": facts.get("forms", [])[:8],
            "models": facts.get("models", [])[:8],
            "symbols": facts.get("symbols", [])[:20],
        },
        indent=2,
    )
    context = "\n\n".join(
        f"Source {i + 1}: {chunk['file_path']} lines {chunk['start_line']}-{chunk['end_line']}\n"
        f"```{chunk.get('extension', '').lstrip('.')}\n{chunk['content'][:2600]}\n```"
        for i, chunk in enumerate(results[:8])
    )
    return f"""You are a senior software engineer helping a developer understand a codebase.
Answer the question using ONLY the repository facts and source snippets below.
If the evidence is incomplete, say what is missing instead of guessing.
Be direct, practical, and cite file paths with line ranges.

Question:
{question}

Repository facts:
{fact_summary}

Retrieved source snippets:
{context}

Answer:"""


def maybe_generate_ollama_answer(question: str, results: list[dict], architecture: dict | None = None) -> str | None:
    model = os.getenv("OLLAMA_MODEL", "").strip()
    if not model:
        return None

    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    payload = {
        "model": model,
        "prompt": build_generation_prompt(question, results, architecture),
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192")),
        },
    }
    request = urllib.request.Request(
        f"{base_url}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=int(os.getenv("OLLAMA_TIMEOUT", "90"))) as response:
            data = json.loads(response.read().decode("utf-8"))
        answer = (data.get("response") or "").strip()
        return answer or None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def maybe_generate_llm_answer(question: str, results: list[dict], architecture: dict | None = None) -> tuple[str, str] | None:
    ollama_answer = maybe_generate_ollama_answer(question, results, architecture)
    if ollama_answer:
        return ollama_answer, "ollama"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.1, api_key=api_key)
        return llm.invoke([HumanMessage(content=build_generation_prompt(question, results, architecture))]).content, "llm"
    except Exception:
        return None


def index_repository(repo_id: str, github_url: str) -> None:
    index_dir = INDEX_ROOT / repo_id
    clone_dir = CLONE_ROOT / repo_id

    try:
        if not indexing_lock.acquire(blocking=False):
            update_repo(repo_id, status="queued", message="Waiting for another repo to finish indexing", progress=5)
            with indexing_lock:
                pass
        else:
            indexing_lock.release()

        with indexing_lock:
            run_indexing_steps(repo_id, github_url, index_dir, clone_dir)
    except Exception as exc:
        update_repo(repo_id, status="failed", message="Indexing failed", progress=100, error=str(exc))


def run_indexing_steps(repo_id: str, github_url: str, index_dir: Path, clone_dir: Path) -> None:
        update_repo(repo_id, status="indexing", message="Cloning repository", progress=10)
        clone_repository(github_url, str(clone_dir))

        update_repo(repo_id, message="Loading supported source files", progress=25)
        documents = load_code_files(str(clone_dir))
        if not documents:
            raise ValueError("No supported files found. Try a repo with code or Markdown files.")

        update_repo(repo_id, message="Chunking files", progress=45, file_count=len(documents))
        chunks = chunk_documents(documents)
        if not chunks:
            raise ValueError("No chunks were created from the repository.")

        update_repo(repo_id, message="Creating local embeddings", progress=65, chunk_count=len(chunks))
        embedding_model = get_embedding_model(provider=EMBEDDING_PROVIDER)
        embedded_chunks = embed_chunks(chunks, embedding_model)

        update_repo(repo_id, message="Building FAISS index", progress=82)
        index, metadata = build_faiss_index(embedded_chunks, index_dir=str(index_dir))

        update_repo(repo_id, message="Creating architecture summary", progress=94)
        architecture = summarize_architecture(documents, chunks)
        onboarding = create_onboarding_summary(architecture)
        suggestions = suggest_questions(architecture)

        runtime_cache[repo_id] = {
            "embedding_model": embedding_model,
            "index": index,
            "metadata": metadata,
        }

        update_repo(
            repo_id,
            status="ready",
            message="Ready to chat",
            progress=100,
            file_count=len(documents),
            chunk_count=len(chunks),
            architecture=architecture,
            onboarding=onboarding,
            suggestions=suggestions,
            error=None,
        )


@app.on_event("startup")
def startup() -> None:
    load_manifest()


@app.get("/", response_model=None)
def frontend():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Chat With Your Codebase API is running."}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "embedding_provider": EMBEDDING_PROVIDER}


@app.post("/api/index")
def start_indexing(payload: IndexRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    github_url = payload.github_url.strip()
    if not github_url.startswith(("https://github.com/", "git@github.com:")):
        raise HTTPException(status_code=400, detail="Please enter a valid GitHub repository URL.")

    repo_id = make_repo_id(github_url)
    existing = repos.get(repo_id)
    if existing and existing["status"] == "ready":
        return {"repo_id": repo_id, "status": existing["status"], "message": existing["message"]}
    if existing and existing["status"] in {"queued", "indexing"}:
        age_seconds = now() - existing.get("updated_at", 0)
        if age_seconds < 120:
            return {"repo_id": repo_id, "status": existing["status"], "message": existing["message"]}

    repo = RepoSession(
        repo_id=repo_id,
        github_url=github_url,
        status="queued",
        message="Waiting to start indexing",
        progress=0,
        created_at=now(),
        updated_at=now(),
        index_dir=str(INDEX_ROOT / repo_id),
        clone_dir=str(CLONE_ROOT / repo_id),
    ).dict()

    with repo_lock:
        repos[repo_id] = repo
    save_manifest()
    background_tasks.add_task(index_repository, repo_id, github_url)
    return {"repo_id": repo_id, "status": "queued", "message": "Indexing started"}


@app.get("/api/repos")
def list_repos() -> list[dict[str, Any]]:
    for repo_id in list(repos):
        refresh_architecture_if_needed(repo_id)
    return sorted(repos.values(), key=lambda item: item["updated_at"], reverse=True)


@app.get("/api/repos/{repo_id}")
def get_repo(repo_id: str) -> dict[str, Any]:
    refresh_architecture_if_needed(repo_id)
    return get_repo_or_404(repo_id)


@app.post("/api/repos/{repo_id}/chat")
def chat(repo_id: str, payload: ChatRequest) -> dict[str, Any]:
    runtime = load_runtime(repo_id)
    repo = ensure_ready(repo_id)
    results = search_similar_chunks(
        query=payload.question,
        embedding_model=runtime["embedding_model"],
        index=runtime["index"],
        metadata=runtime["metadata"],
        top_k=max(payload.top_k, 10),
    )
    results = hybrid_rerank(
        payload.question,
        results,
        runtime["metadata"],
        repo.get("architecture"),
    )[:payload.top_k]

    free_answer = generate_free_answer(
        payload.question,
        results,
        metadata=runtime["metadata"],
        architecture=repo.get("architecture"),
    )
    generated = maybe_generate_llm_answer(payload.question, free_answer.get("sources") or results, repo.get("architecture"))
    if generated:
        answer, mode = generated
        free_answer["answer"] = answer
        free_answer["mode"] = mode
    else:
        free_answer["mode"] = "free"
    free_answer["question"] = payload.question
    return free_answer


@app.get("/api/repos/{repo_id}/sources")
def sources(repo_id: str, limit: int = 300) -> dict[str, Any]:
    runtime = load_runtime(repo_id)
    metadata = runtime["metadata"][:limit]
    files: dict[str, dict[str, Any]] = {}
    for chunk in runtime["metadata"]:
        current = files.setdefault(
            chunk["file_path"],
            {"file_path": chunk["file_path"], "extension": chunk.get("extension", ""), "chunks": 0},
        )
        current["chunks"] += 1
    return {"files": sorted(files.values(), key=lambda item: item["file_path"]), "sample_chunks": metadata}


@app.get("/api/repos/{repo_id}/chunks/{chunk_index}")
def chunk_detail(repo_id: str, chunk_index: int) -> dict[str, Any]:
    runtime = load_runtime(repo_id)
    metadata = runtime["metadata"]
    if chunk_index < 0 or chunk_index >= len(metadata):
        raise HTTPException(status_code=404, detail="Chunk not found.")
    return metadata[chunk_index]


@app.get("/api/repos/{repo_id}/risks")
def risks(repo_id: str) -> dict[str, Any]:
    repo = ensure_ready(repo_id)
    clone_dir = repo["clone_dir"]
    documents = load_code_files(clone_dir)
    return {"findings": find_risk_signals(documents)}


def cli() -> None:
    print("FastAPI app is ready.")
    print("Run: uvicorn backend.main:app")
    print("Tip: avoid --reload while indexing because generated FAISS files can restart the server.")
    print("Then open: http://127.0.0.1:8000")


if __name__ == "__main__":
    cli()
