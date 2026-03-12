"""
vector_store.py
---------------
RESPONSIBILITY: Store embeddings in FAISS and perform semantic search.

WHAT IS FAISS?
FAISS (Facebook AI Similarity Search) is a library that stores thousands
of embedding vectors and lets you find the MOST SIMILAR ones to a query
vector — extremely fast, even with millions of vectors.

Think of it like this:
    - Normal database search: "Find rows where column = 'login'"  (exact match)
    - FAISS search: "Find chunks that MEAN something similar to 'authentication'" (semantic)

HOW FAISS WORKS (simplified):
1. You give it thousands of vectors (one per code chunk)
2. It organizes them in a special data structure (an index)
3. When you search, you give it a query vector
4. It returns the K most similar vectors (nearest neighbors)
5. Similarity is measured by cosine distance or L2 distance

This is STEP 5 (storing) and STEP 6 (searching) of the RAG pipeline.
"""

import os
import json
import pickle
import numpy as np
import faiss  # Facebook AI Similarity Search

# -------------------------------------------------------
# STORAGE SETTINGS
# We save the FAISS index AND the chunk metadata to disk
# so you don't have to re-process the repo every time.
# -------------------------------------------------------
DEFAULT_INDEX_DIR = "./faiss_index"     # Folder to save index files
INDEX_FILE = "index.faiss"              # The FAISS binary index file
METADATA_FILE = "metadata.pkl"          # Chunk metadata (file paths, content, etc.)


def build_faiss_index(embedded_chunks: list[dict], index_dir: str = DEFAULT_INDEX_DIR) -> faiss.Index:
    """
    Build a FAISS index from embedded chunks and save it to disk.

    Steps:
    1. Extract all embedding vectors from chunks
    2. Convert to a NumPy array (FAISS requires NumPy format)
    3. Create a FAISS IndexFlatIP (Inner Product = cosine similarity for normalized vectors)
    4. Add all vectors to the index
    5. Save the index AND metadata to disk

    Args:
        embedded_chunks (list[dict]): Chunks with "embedding" key from embeddings.py
        index_dir (str): Directory to save the index files

    Returns:
        faiss.Index: The built FAISS index (also saved to disk)
    """

    print(f"[4/4] Building FAISS vector index...")

    # Create the save directory if it doesn't exist
    os.makedirs(index_dir, exist_ok=True)

    # ---- STEP 1: Extract vectors ----
    # Each embedding is a Python list of floats.
    # We stack them into a 2D NumPy array: shape = (num_chunks, embedding_dim)
    vectors = np.array(
        [chunk["embedding"] for chunk in embedded_chunks],
        dtype=np.float32  # FAISS requires float32, not float64
    )

    # Determine embedding dimensions from the first vector
    num_chunks, embedding_dim = vectors.shape
    print(f"      Index dimensions: {num_chunks} chunks × {embedding_dim} dimensions")

    # ---- STEP 2: Create FAISS index ----
    # IndexFlatIP = "Flat Index using Inner Product (dot product)"
    # For normalized vectors, dot product = cosine similarity
    # "Flat" means it does exact (brute-force) search — slower but 100% accurate
    # For production with millions of vectors, use IndexIVFFlat instead
    index = faiss.IndexFlatIP(embedding_dim)

    # ---- STEP 3: Add vectors to the index ----
    index.add(vectors)  # This is the core FAISS operation

    # ---- STEP 4: Save the index to disk ----
    index_path = os.path.join(index_dir, INDEX_FILE)
    faiss.write_index(index, index_path)
    print(f"      FAISS index saved to: {index_path}")

    # ---- STEP 5: Save metadata separately ----
    # FAISS only stores vectors — it doesn't know about file paths, content, etc.
    # We save that metadata in a pickle file alongside the index.
    # Metadata is saved WITHOUT the embedding vectors (saves space).
    metadata = [
        {k: v for k, v in chunk.items() if k != "embedding"}
        for chunk in embedded_chunks
    ]

    metadata_path = os.path.join(index_dir, METADATA_FILE)
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"      Metadata saved to: {metadata_path}")
    print(f"      Index built with {index.ntotal} vectors.\n")

    return index, metadata


def load_faiss_index(index_dir: str = DEFAULT_INDEX_DIR) -> tuple:
    """
    Load a previously saved FAISS index and its metadata from disk.

    Use this to avoid re-processing the repository on every run.

    Args:
        index_dir (str): Directory where index files are saved.

    Returns:
        tuple: (faiss.Index, list of metadata dicts)

    Raises:
        FileNotFoundError: If index files don't exist yet.
    """

    index_path = os.path.join(index_dir, INDEX_FILE)
    metadata_path = os.path.join(index_dir, METADATA_FILE)

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"No FAISS index found in '{index_dir}'.\n"
            "Run the pipeline first to build the index."
        )

    print(f"Loading existing FAISS index from: {index_dir}")
    index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    print(f"Loaded index with {index.ntotal} vectors and {len(metadata)} metadata entries.\n")
    return index, metadata


def search_similar_chunks(
    query: str,
    embedding_model,
    index: faiss.Index,
    metadata: list[dict],
    top_k: int = 5
) -> list[dict]:
    """
    Search the FAISS index for chunks most relevant to a user query.

    HOW THIS WORKS:
    1. Convert the user's question into an embedding vector
    2. Ask FAISS: "Which stored vectors are closest to this?"
    3. FAISS returns the indices (positions) of the top-K matches
    4. We use those indices to look up the full chunk metadata
    5. Return the matched chunks sorted by relevance

    Args:
        query (str): The user's question, e.g. "Where is authentication implemented?"
        embedding_model: Same model used to build the index
        index (faiss.Index): The loaded FAISS index
        metadata (list[dict]): The chunk metadata list
        top_k (int): How many results to return (default: 5)

    Returns:
        list[dict]: Top-K most relevant chunks, each with a "score" field added.
                    Higher score = more relevant (for IndexFlatIP / cosine similarity).
    """

    print(f"\nSearching for: '{query}'")

    # ---- Step 1: Embed the query ----
    # We convert the user's question to the same vector space as the chunks
    query_vector = embedding_model.embed_query(query)

    # Convert to numpy float32 array with shape (1, embedding_dim)
    # The "1" is because FAISS expects a batch, even for a single query
    query_array = np.array([query_vector], dtype=np.float32)

    # ---- Step 2: Search FAISS ----
    # index.search() returns:
    #   scores  — similarity scores for each result (higher is more similar)
    #   indices — positions in the index of the matched vectors
    scores, indices = index.search(query_array, top_k)

    # scores and indices are 2D arrays with shape (1, top_k)
    # We take [0] to get the results for our single query
    scores = scores[0]
    indices = indices[0]

    # ---- Step 3: Build results ----
    results = []
    for rank, (idx, score) in enumerate(zip(indices, scores)):
        if idx == -1:
            # FAISS returns -1 when there are fewer results than top_k
            continue

        result = metadata[idx].copy()   # Get the chunk's metadata
        result["score"] = float(score)  # Attach the similarity score
        result["rank"] = rank + 1       # Human-friendly rank (1-indexed)
        results.append(result)

    print(f"Found {len(results)} relevant chunks.\n")
    return results


def index_exists(index_dir: str = DEFAULT_INDEX_DIR) -> bool:
    """
    Check whether a saved FAISS index already exists on disk.

    Useful to skip re-processing if the index was already built.

    Args:
        index_dir (str): Directory to check.

    Returns:
        bool: True if both index and metadata files exist.
    """
    index_path = os.path.join(index_dir, INDEX_FILE)
    metadata_path = os.path.join(index_dir, METADATA_FILE)
    return os.path.exists(index_path) and os.path.exists(metadata_path)