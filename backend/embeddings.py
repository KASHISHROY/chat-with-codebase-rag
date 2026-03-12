"""
embeddings.py
-------------
RESPONSIBILITY: Convert code chunks (text) into embeddings (numbers/vectors).

WHAT IS AN EMBEDDING?
An embedding is a way of representing text as a list of numbers (a vector).
These numbers capture the MEANING of the text, not just the words.

For example:
    "How do I log in?" → [0.23, -0.87, 0.14, 0.66, ...]  (384 numbers)
    "user authentication" → [0.25, -0.81, 0.19, 0.70, ...]  (384 numbers)

Notice that even though the words are different, the numbers are SIMILAR
because the MEANING is similar. This is what makes semantic search work!

We support two embedding providers:
1. HuggingFace (free, runs locally — DEFAULT)
2. OpenAI (paid, very accurate — set OPENAI_API_KEY in .env)

This is STEP 4 of the RAG pipeline.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
# This is how we safely store API keys without hardcoding them
load_dotenv()


def get_embedding_model(provider: str = "huggingface"):
    """
    Return a configured embedding model.

    SUPPORTED PROVIDERS:
    - "huggingface" (default): Uses sentence-transformers locally.
      - Free to use, no API key needed
      - Model: 'all-MiniLM-L6-v2' — small, fast, and surprisingly good
      - Downloads the model once (~90MB) and caches it locally
      
    - "openai": Uses OpenAI's text-embedding-3-small model.
      - Requires OPENAI_API_KEY in your .env file
      - More accurate than HuggingFace for many tasks
      - Costs a tiny amount per request (~$0.00002 per 1K tokens)

    Args:
        provider (str): "huggingface" or "openai"

    Returns:
        An embedding model object (LangChain-compatible)
    """

    if provider == "openai":
        # Check that the user has provided an API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables.\n"
                "Please create a .env file and add: OPENAI_API_KEY=sk-..."
            )

        from langchain_openai import OpenAIEmbeddings

        print("Using OpenAI embeddings (text-embedding-3-small)")
        return OpenAIEmbeddings(
            model="text-embedding-3-small",  # Cheap and fast; use ada-002 for older setups
            openai_api_key=api_key,
        )

    elif provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        print("Using HuggingFace embeddings (all-MiniLM-L6-v2)")
        print("(First run will download ~90MB model — this is normal!)\n")

        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            # Run on CPU by default; change to "cuda" if you have a GPU
            model_kwargs={"device": "cpu"},
            # Normalize vectors so cosine similarity works correctly
            encode_kwargs={"normalize_embeddings": True},
        )

    else:
        raise ValueError(f"Unknown embedding provider: '{provider}'. Choose 'huggingface' or 'openai'.")


def embed_chunks(chunks: list[dict], embedding_model) -> list[dict]:
    """
    Generate embeddings for a list of code chunks.

    For each chunk, we embed the CONTENT (code text) and store the
    resulting vector alongside the original chunk metadata.

    Args:
        chunks (list[dict]): Output from chunking.chunk_documents()
        embedding_model: An embedding model from get_embedding_model()

    Returns:
        list[dict]: Same chunks, but each now has an "embedding" key
                    containing a list of floats (the vector).

    Example:
        chunk = {
            "content": "def validate_user(username, password):\n    ...",
            "file_path": "auth/login.py",
            "embedding": [0.12, -0.45, 0.78, ...]  # ← NEW
        }
    """

    print(f"Generating embeddings for {len(chunks)} chunks...")
    print("(This may take a minute for large repos — please wait)\n")

    # Extract just the text content from each chunk
    texts = [chunk["content"] for chunk in chunks]

    # Generate embeddings in one batch (much faster than one by one)
    # The result is a list of vectors: [[0.1, 0.2, ...], [0.3, 0.1, ...], ...]
    vectors = embedding_model.embed_documents(texts)

    # Attach each vector back to its corresponding chunk
    embedded_chunks = []
    for chunk, vector in zip(chunks, vectors):
        enriched_chunk = chunk.copy()  # Don't modify the original
        enriched_chunk["embedding"] = vector
        embedded_chunks.append(enriched_chunk)

    print(f"Generated {len(embedded_chunks)} embeddings. "
          f"Each vector has {len(vectors[0])} dimensions.\n")

    return embedded_chunks