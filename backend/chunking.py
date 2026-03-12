"""
chunking.py
-----------
RESPONSIBILITY: Split large code files into smaller, meaningful chunks.

WHY DO WE NEED CHUNKING?
When we send code to an AI (LLM), there is a limit on how much text it can
process at once (called the "context window"). A large codebase might have
hundreds of files, each with thousands of lines — way too much to send at once.

Chunking solves this by:
1. Breaking code into small pieces (e.g., 500–1500 characters each)
2. Adding overlap between chunks so we don't lose context at the boundaries
3. Storing metadata about WHERE each chunk came from (file name, line numbers)

This is STEP 3 of the RAG pipeline.
"""
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
)
# -------------------------------------------------------
# CHUNK SETTINGS
# These control how code is divided.
#
# CHUNK_SIZE: Maximum number of characters per chunk.
#   - Too small → chunks lose context (a function gets split mid-way)
#   - Too large → fewer chunks, less precision when retrieving
#
# CHUNK_OVERLAP: Number of characters shared between adjacent chunks.
#   - This prevents losing context at chunk boundaries.
#   - e.g., the end of chunk 1 and start of chunk 2 will share 200 chars.
# -------------------------------------------------------
CHUNK_SIZE = 1500       # ~400 tokens — good balance for code
CHUNK_OVERLAP = 200     # overlap to preserve context between chunks


# Map file extensions to LangChain's Language enum
# This lets the splitter understand language-specific syntax
# e.g., for Python it splits on "def " and "class " boundaries
LANGUAGE_MAP = {
    ".py":   Language.PYTHON,
    ".js":   Language.JS,
    ".jsx":  Language.JS,
    ".ts":   Language.JS,   # TypeScript uses same separators as JS
    ".tsx":  Language.JS,
    ".html": Language.HTML,
    ".md":   Language.MARKDOWN,
}


def get_splitter_for_extension(extension: str) -> RecursiveCharacterTextSplitter:
    """
    Return a text splitter tuned for the given file extension.

    For known code languages (Python, JS, etc.), we use a language-aware
    splitter that tries to split on function/class boundaries.

    For unknown extensions (.json, .txt, etc.), we use a generic splitter
    that splits on blank lines and whitespace.

    Args:
        extension (str): File extension like ".py", ".js", ".md"

    Returns:
        RecursiveCharacterTextSplitter: A configured splitter instance
    """

    if extension in LANGUAGE_MAP:
        # Language-aware splitter: understands Python "def", "class", etc.
        return RecursiveCharacterTextSplitter.from_language(
            language=LANGUAGE_MAP[extension],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    else:
        # Generic splitter: splits on blank lines, then sentences, then chars
        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],  # Try these separators in order
        )


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Take a list of loaded documents and split each into smaller chunks.

    Each output chunk is a dictionary with:
        - content:      The actual code text of this chunk
        - file_path:    Which file this chunk came from
        - chunk_index:  Position of this chunk within its file (0, 1, 2, ...)
        - extension:    File extension (useful for syntax highlighting later)
        - start_line:   Approximate starting line number in the original file
        - end_line:     Approximate ending line number in the original file

    Args:
        documents (list[dict]): Output from repo_loader.load_code_files()
                                Each dict has "file_path" and "content".

    Returns:
        list[dict]: All chunks from all files, as a flat list.

    Example:
        chunks = chunk_documents(documents)
        # chunks[0] = {
        #     "content": "def validate_user(username, password):\n    ...",
        #     "file_path": "auth/login.py",
        #     "chunk_index": 0,
        #     "extension": ".py",
        #     "start_line": 20,
        #     "end_line": 40,
        # }
    """

    all_chunks = []  # Collect all chunks across all files

    print(f"[3/4] Chunking {len(documents)} files...")

    for doc in documents:
        file_path = doc["file_path"]
        content = doc["content"]

        # Determine the file extension for language-aware splitting
        extension = "." + file_path.split(".")[-1].lower() if "." in file_path else ""

        # Get the right splitter for this file type
        splitter = get_splitter_for_extension(extension)

        # Split the content into chunks
        # LangChain returns a list of strings (the text of each chunk)
        text_chunks = splitter.split_text(content)

        # For each chunk, calculate approximate line numbers
        # We do this by counting newlines before each chunk appears in the file
        current_line = 1

        for i, chunk_text in enumerate(text_chunks):
            # Find where this chunk appears in the original content
            chunk_start_pos = content.find(chunk_text)

            if chunk_start_pos != -1:
                # Count newlines up to the start of this chunk
                start_line = content[:chunk_start_pos].count("\n") + 1
                end_line = start_line + chunk_text.count("\n")
            else:
                # Fallback if chunk text can't be found exactly (due to overlap)
                start_line = current_line
                end_line = current_line + chunk_text.count("\n")

            current_line = end_line + 1

            # Build the chunk dictionary with all metadata
            chunk = {
                "content":     chunk_text,        # The actual code text
                "file_path":   file_path,          # Source file
                "chunk_index": i,                  # Position in file
                "extension":   extension,          # File type
                "start_line":  start_line,         # Approx. starting line
                "end_line":    end_line,           # Approx. ending line
            }

            all_chunks.append(chunk)

    print(f"      Created {len(all_chunks)} chunks from {len(documents)} files.\n")
    return all_chunks