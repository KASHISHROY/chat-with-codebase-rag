"""
repo_loader.py
--------------
RESPONSIBILITY: Clone a GitHub repository to local disk and load all
readable code files from it.

WHAT THIS FILE DOES:
1. Takes a GitHub URL from the user
2. Clones the repository into a local temp folder
3. Walks through every file in the cloned repo
4. Reads only the file types we care about (.py, .js, .ts, .md, etc.)
5. Returns a list of documents (file path + file content)

This is STEP 1 and STEP 2 of the RAG pipeline.
"""

import os
import shutil
import tempfile
from git import Repo  # GitPython library — lets us clone repos in Python

# -------------------------------------------------------
# SUPPORTED FILE EXTENSIONS
# These are the file types we will read from the repo.
# You can add more extensions here if needed.
# -------------------------------------------------------
SUPPORTED_EXTENSIONS = {
    ".py",    # Python
    ".js",    # JavaScript
    ".ts",    # TypeScript
    ".jsx",   # React JSX
    ".tsx",   # React TSX
    ".md",    # Markdown docs
    ".txt",   # Plain text
    ".json",  # JSON config files
    ".yaml",  # YAML config files
    ".yml",   # YAML (alternate extension)
    ".html",  # HTML files
    ".css",   # CSS stylesheets
    ".sh",    # Shell scripts
    ".env.example",  # Example env files (safe — no real secrets)
}

# -------------------------------------------------------
# FOLDERS TO SKIP
# These folders usually contain auto-generated code or
# dependencies that we do NOT want to index.
# -------------------------------------------------------
SKIP_FOLDERS = {
    "node_modules",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".env",
    "dist",
    "build",
    ".next",
    ".cache",
    "coverage",
}


def clone_repository(github_url: str, clone_dir: str = None) -> str:
    """
    Clone a GitHub repository to a local directory.

    Args:
        github_url (str): The full GitHub URL, e.g. https://github.com/user/repo
        clone_dir (str): Optional path to clone into. If None, a temp folder is created.

    Returns:
        str: The local path where the repo was cloned.

    Example:
        local_path = clone_repository("https://github.com/tiangolo/fastapi")
    """

    # If no directory is specified, create a temporary one
    if clone_dir is None:
        clone_dir = tempfile.mkdtemp(prefix="codebase_rag_")

    print(f"[1/4] Cloning repository: {github_url}")
    print(f"      Destination folder: {clone_dir}")

    # If the folder already has content from a previous run, clean it first
    if os.path.exists(clone_dir) and os.listdir(clone_dir):
        print("      Folder already exists. Cleaning it before re-cloning...")
        shutil.rmtree(clone_dir)
        os.makedirs(clone_dir)

    # Use GitPython to clone — this is the equivalent of running:
    # git clone <github_url> <clone_dir>
    Repo.clone_from(github_url, clone_dir)

    print(f"      Repository cloned successfully!\n")
    return clone_dir


def load_code_files(repo_path: str) -> list[dict]:
    """
    Walk through the cloned repository and load all supported code files.

    For each file, we create a "document" dictionary with:
        - file_path: relative path of the file within the repo
        - content:   the raw text content of the file

    Args:
        repo_path (str): Local path to the cloned repository.

    Returns:
        list[dict]: A list of documents, each being a dict with keys
                    'file_path' and 'content'.

    Example:
        documents = load_code_files("/tmp/codebase_rag_abc123")
        # documents[0] = {"file_path": "src/main.py", "content": "import os\n..."}
    """

    documents = []  # This will hold all our loaded files

    print(f"[2/4] Loading code files from: {repo_path}")

    # os.walk() traverses the entire directory tree recursively.
    # For each folder it gives us:
    #   root      = current folder path
    #   dirs      = list of subfolders in this folder
    #   files     = list of files in this folder
    for root, dirs, files in os.walk(repo_path):

        # ---- SKIP UNWANTED FOLDERS ----
        # We modify `dirs` IN PLACE so os.walk won't descend into them.
        # This is important for performance — node_modules alone can have
        # thousands of files!
        dirs[:] = [
            d for d in dirs
            if d not in SKIP_FOLDERS and not d.startswith(".")
        ]

        for file_name in files:
            full_path = os.path.join(root, file_name)

            # Get the file extension, e.g. ".py" from "main.py"
            _, extension = os.path.splitext(file_name)

            # Skip files with unsupported extensions
            if extension.lower() not in SUPPORTED_EXTENSIONS:
                continue

            # Skip very large files (> 500KB) — they may be auto-generated
            try:
                file_size = os.path.getsize(full_path)
                if file_size > 500_000:  # 500 KB limit
                    print(f"      Skipping large file: {file_name} ({file_size // 1024} KB)")
                    continue
            except OSError:
                continue

            # Try to read the file as UTF-8 text
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Skip empty files — nothing useful to index
                if not content.strip():
                    continue

                # Make the path relative to the repo root (cleaner display)
                relative_path = os.path.relpath(full_path, repo_path)

                # Add this file as a document
                documents.append({
                    "file_path": relative_path,
                    "content": content,
                })

            except Exception as e:
                # If we can't read a file for any reason, just skip it
                print(f"      Warning: Could not read {file_name}: {e}")
                continue

    print(f"      Loaded {len(documents)} code files.\n")
    return documents


def load_repository(github_url: str) -> tuple[list[dict], str]:
    """
    High-level function: clone a GitHub repo and load all its code files.

    This is the main entry point for Step 1 & 2 of the RAG pipeline.

    Args:
        github_url (str): GitHub repository URL.

    Returns:
        tuple: (list of document dicts, local path to the cloned repo)
    """
    # Step 1: Clone the repository
    local_path = clone_repository(github_url)

    # Step 2: Load all code files from it
    documents = load_code_files(local_path)

    return documents, local_path