# Chat With Your Codebase — RAG Developer Assistant

A Retrieval-Augmented Generation (RAG) system that lets you ask questions about any GitHub repository in plain English.

## 🚀 Quick Start

### 1. Clone this project
```bash
git clone <this-repo-url>
cd chat-with-codebase-rag
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env and optionally add your OPENAI_API_KEY
```

### 5. Run the system
```bash
cd backend
python main.py
```

### 6. Use it!
```
Enter GitHub repository URL: https://github.com/tiangolo/fastapi
Your question: Where is the routing logic implemented?
```

---

## 📁 Project Structure

```
chat-with-codebase-rag/
├── backend/
│   ├── main.py          # Entry point — runs the full pipeline
│   ├── repo_loader.py   # Clones GitHub repos and loads files
│   ├── chunking.py      # Splits files into meaningful chunks
│   ├── embeddings.py    # Converts text to embedding vectors
│   └── vector_store.py  # FAISS index: store and search vectors
├── frontend/            # (Day 2+) React UI
├── requirements.txt     # Python dependencies
├── .env.example         # Template for environment variables
└── README.md
```

---

## 🧠 How It Works (RAG Pipeline)

```
GitHub URL
    ↓
[repo_loader.py]  → Clone repo → Load .py, .js, .md files
    ↓
[chunking.py]     → Split files into 1500-char chunks with metadata
    ↓
[embeddings.py]   → Convert each chunk to a 384-dim vector
    ↓
[vector_store.py] → Store vectors in FAISS index
    ↓
User asks a question
    ↓
[vector_store.py] → Embed question → Find top-5 similar chunks
    ↓
[main.py]         → Send question + chunks to LLM → Return answer
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_PROVIDER` | `huggingface` | Use `huggingface` (free) or `openai` (paid) |
| `OPENAI_API_KEY` | None | Required only for OpenAI embeddings + LLM answers |

---

## 💡 Tips

- **First run downloads ~90MB** for the HuggingFace model — this is normal
- **Index is saved** to `./faiss_index/` — reload it without re-cloning
- **Large repos** (like React source) may take 2–5 minutes to index
- For **better answers**, add `OPENAI_API_KEY` to your `.env`

---

## 🗺️ Roadmap

- [x] Day 1: Backend RAG pipeline (CLI)
- [ ] Day 2: FastAPI REST endpoints
- [ ] Day 3: React frontend with chat UI
- [ ] Day 4: Streaming responses + syntax highlighting
- [ ] Day 5: Multi-repo support + authentication