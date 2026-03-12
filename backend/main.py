"""
main.py
-------
RESPONSIBILITY: The main entry point that ties the entire RAG pipeline together.

This file is the "conductor" — it calls each module in order:
    1. repo_loader  → Clone & load the repository
    2. chunking     → Split files into small pieces
    3. embeddings   → Convert chunks into vectors
    4. vector_store → Store vectors in FAISS and search

For Day 1, this is a COMMAND-LINE script.
On Day 2+, we will wrap this with FastAPI to expose HTTP endpoints.

HOW TO RUN:
    python main.py

The script will ask you for a GitHub URL and a question interactively.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import our custom modules
from repo_loader import load_repository
from chunking import chunk_documents
from embeddings import get_embedding_model, embed_chunks
from vector_store import (
    build_faiss_index,
    load_faiss_index,
    search_similar_chunks,
    index_exists,
)

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

# Choose embedding provider: "huggingface" (free) or "openai" (needs API key)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")

# Where to store the FAISS index files
INDEX_DIR = "./faiss_index"


# -------------------------------------------------------
# LLM ANSWER GENERATION
# -------------------------------------------------------

def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Use an LLM to generate a natural language answer based on retrieved code chunks.

    HOW THIS WORKS (RAG):
    1. We have the user's question
    2. We have the most relevant code chunks (retrieved from FAISS)
    3. We combine them into a PROMPT
    4. We send the prompt to the LLM
    5. The LLM generates a helpful explanation

    This is the final step of the RAG pipeline — "Generation" in Retrieval-Augmented Generation.

    Args:
        question (str): The user's question
        retrieved_chunks (list[dict]): Top-K chunks from vector search

    Returns:
        str: The LLM's answer
    """

    # ---- Build the context string ----
    # We combine all retrieved chunks into a single readable block
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(
            f"--- Chunk {i+1} ---\n"
            f"File: {chunk['file_path']} (Lines {chunk['start_line']}–{chunk['end_line']})\n"
            f"```\n{chunk['content']}\n```"
        )
    context = "\n\n".join(context_parts)

    # ---- Build the full prompt ----
    prompt = f"""You are an expert software developer assistant. 
A user is exploring a codebase and has a question about it.

Below are the most relevant code snippets retrieved from the codebase:

{context}

Based ONLY on the code snippets above, please answer the user's question clearly and helpfully.
- Identify WHICH FILE contains the relevant code
- Explain WHAT the code does in simple terms
- Quote the relevant function/class name
- Mention the line numbers

User Question: {question}

Answer:"""

    # ---- Choose LLM ----
    # Try OpenAI first, fall back to a simple template-based answer
    openai_key = os.getenv("OPENAI_API_KEY")

    if openai_key:
        try:
            from langchain_openai import ChatOpenAI
            from langchain.schema import HumanMessage

            llm = ChatOpenAI(
                model="gpt-3.5-turbo",  # Cheap and fast; use gpt-4o for better answers
                temperature=0.2,         # Lower = more factual, less creative
                openai_api_key=openai_key,
            )

            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content

        except Exception as e:
            print(f"OpenAI call failed: {e}\nFalling back to template answer.\n")

    # ---- Fallback: Template-based answer (no LLM needed) ----
    # If no LLM is configured, we format the retrieved chunks nicely
    # and return them directly. This still gives useful information!
    lines = [
        f"Here are the most relevant code sections for your question: '{question}'\n",
        "=" * 60,
    ]
    for i, chunk in enumerate(retrieved_chunks):
        lines.append(f"\n[Result {i+1}] File: {chunk['file_path']}")
        lines.append(f"Lines: {chunk['start_line']}–{chunk['end_line']}")
        lines.append(f"Relevance Score: {chunk['score']:.4f}")
        lines.append(f"\nCode:\n{chunk['content'][:800]}...")  # Show first 800 chars
        lines.append("-" * 60)

    lines.append(
        "\nTip: Set OPENAI_API_KEY in your .env file to get AI-generated explanations!"
    )
    return "\n".join(lines)


# -------------------------------------------------------
# PIPELINE FUNCTIONS
# -------------------------------------------------------

def run_indexing_pipeline(github_url: str):
    """
    Run the full RAG indexing pipeline for a GitHub repository.

    Steps:
    1. Clone the repo
    2. Load all code files
    3. Split files into chunks
    4. Generate embeddings for chunks
    5. Store in FAISS

    This only needs to run ONCE per repository.
    After that, you can query it repeatedly without re-indexing.

    Args:
        github_url (str): GitHub repository URL to index
    """

    print("\n" + "="*60)
    print("   STARTING RAG INDEXING PIPELINE")
    print("="*60 + "\n")

    # STEP 1 & 2: Clone and load the repository
    documents, local_path = load_repository(github_url)

    if not documents:
        print("ERROR: No supported files found in the repository.")
        print("Check that the repo contains .py, .js, .ts, .md, or similar files.")
        return

    # STEP 3: Split files into chunks
    chunks = chunk_documents(documents)

    if not chunks:
        print("ERROR: Chunking produced no output. Files may be empty.")
        return

    # STEP 4: Load embedding model and generate embeddings
    print("[4/4] Generating embeddings...")
    embedding_model = get_embedding_model(provider=EMBEDDING_PROVIDER)
    embedded_chunks = embed_chunks(chunks, embedding_model)

    # STEP 5: Build FAISS index and save to disk
    index, metadata = build_faiss_index(embedded_chunks, index_dir=INDEX_DIR)

    print("="*60)
    print("   INDEXING COMPLETE!")
    print(f"   Repository: {github_url}")
    print(f"   Files processed: {len(documents)}")
    print(f"   Chunks created: {len(chunks)}")
    print(f"   Vectors stored: {index.ntotal}")
    print(f"   Index saved to: {INDEX_DIR}/")
    print("="*60 + "\n")

    return embedding_model, index, metadata


def run_query(question: str, embedding_model, index, metadata, top_k: int = 5):
    """
    Answer a user question by searching the FAISS index and calling the LLM.

    Steps:
    1. Embed the user's question
    2. Search FAISS for similar chunks
    3. Pass chunks + question to the LLM
    4. Display the answer

    Args:
        question (str): The user's question
        embedding_model: The loaded embedding model
        index: The loaded FAISS index
        metadata: The chunk metadata list
        top_k (int): Number of chunks to retrieve
    """

    # Search for relevant chunks
    results = search_similar_chunks(
        query=question,
        embedding_model=embedding_model,
        index=index,
        metadata=metadata,
        top_k=top_k,
    )

    if not results:
        print("No relevant code found for that question.")
        return

    # Generate an answer using the LLM (or fallback template)
    answer = generate_answer(question, results)

    # Display the answer
    print("\n" + "="*60)
    print("   ANSWER")
    print("="*60)
    print(answer)
    print("="*60 + "\n")


# -------------------------------------------------------
# MAIN INTERACTIVE LOOP
# -------------------------------------------------------

def main():
    """
    Interactive command-line interface for the RAG system.

    Run this script and it will guide you through:
    1. Entering a GitHub URL (or using an existing index)
    2. Asking questions about the codebase
    3. Seeing AI-powered answers

    Type 'quit' to exit.
    """

    print("\n" + "="*60)
    print("   CHAT WITH YOUR CODEBASE — RAG Developer Assistant")
    print("="*60 + "\n")

    embedding_model = None
    index = None
    metadata = None

    # Check if an index already exists from a previous run
    if index_exists(INDEX_DIR):
        print(f"Found an existing index at '{INDEX_DIR}/'")
        choice = input("Load existing index? (y/n): ").strip().lower()

        if choice == "y":
            embedding_model = get_embedding_model(provider=EMBEDDING_PROVIDER)
            index, metadata = load_faiss_index(INDEX_DIR)
        else:
            print("Will create a new index.\n")

    # If no index is loaded yet, ask for a GitHub URL and index it
    if index is None:
        github_url = input("Enter GitHub repository URL: ").strip()
        if not github_url:
            print("No URL entered. Exiting.")
            return

        result = run_indexing_pipeline(github_url)
        if result is None:
            return
        embedding_model, index, metadata = result

    # ---- Interactive Q&A loop ----
    print("\nYou can now ask questions about the codebase!")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        run_query(question, embedding_model, index, metadata)


if __name__ == "__main__":
    main()