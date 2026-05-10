"""
Free codebase insight helpers.

These functions avoid paid LLM calls. They use file paths, extensions, imports,
and retrieved chunks to create useful onboarding and citation-rich answers.
"""

from __future__ import annotations

import ast
import os
import re
from collections import Counter, defaultdict
from pathlib import PurePosixPath


ENTRYPOINT_NAMES = {
    "main.py",
    "app.py",
    "server.py",
    "index.js",
    "index.ts",
    "index.tsx",
    "App.jsx",
    "App.tsx",
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "Dockerfile",
    "README.md",
}

QUESTION_BANK = {
    "python": [
        "Where is the main application entry point?",
        "Which files define API routes or request handlers?",
        "Where is database or persistence logic implemented?",
    ],
    "javascript": [
        "Where are the main UI components defined?",
        "How does data flow through the frontend?",
        "Where are API calls made?",
    ],
    "typescript": [
        "Which types or interfaces are central to the app?",
        "Where is state managed?",
        "Where are API contracts defined?",
    ],
    "markdown": [
        "How do I run this project locally?",
        "What does the README say this project does?",
    ],
}


def extension_label(extension: str) -> str:
    labels = {
        ".py": "Python",
        ".js": "JavaScript",
        ".jsx": "React",
        ".ts": "TypeScript",
        ".tsx": "React TypeScript",
        ".md": "Markdown",
        ".json": "JSON",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".html": "HTML",
        ".css": "CSS",
        ".sh": "Shell",
    }
    return labels.get(extension.lower(), extension.lstrip(".").upper() or "Text")


def detect_frameworks(documents: list[dict]) -> list[str]:
    combined_paths = " ".join(doc["file_path"].lower() for doc in documents)
    combined_content = "\n".join(doc["content"][:4000].lower() for doc in documents[:80])
    frameworks = []

    checks = [
        ("FastAPI", "fastapi" in combined_content),
        ("Flask", "from flask" in combined_content or "import flask" in combined_content),
        ("Django", "django" in combined_content or "manage.py" in combined_paths),
        ("React", "react" in combined_content or ".tsx" in combined_paths or ".jsx" in combined_paths),
        ("Next.js", "next.config" in combined_paths or '"next"' in combined_content),
        ("Express", "express()" in combined_content or '"express"' in combined_content),
        ("LangChain", "langchain" in combined_content),
        ("FAISS", "faiss" in combined_content),
    ]
    for name, found in checks:
        if found:
            frameworks.append(name)
    return frameworks


def extract_python_imports(content: str) -> list[str]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".")[0])
    return sorted(set(imports))


def extract_js_imports(content: str) -> list[str]:
    patterns = [
        r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]",
        r"require\(['\"]([^'\"]+)['\"]\)",
    ]
    imports = []
    for pattern in patterns:
        imports.extend(re.findall(pattern, content))
    return sorted(set(item.split("/")[0] for item in imports))


def build_dependency_signals(documents: list[dict], limit: int = 20) -> list[dict]:
    signals = []
    for doc in documents:
        path = doc["file_path"]
        ext = os.path.splitext(path)[1].lower()
        if ext == ".py":
            imports = extract_python_imports(doc["content"])
        elif ext in {".js", ".jsx", ".ts", ".tsx"}:
            imports = extract_js_imports(doc["content"])
        else:
            imports = []
        if imports:
            signals.append({"file_path": path, "imports": imports[:12]})
    return signals[:limit]


def summarize_architecture(documents: list[dict], chunks: list[dict]) -> dict:
    extension_counts = Counter(
        os.path.splitext(doc["file_path"])[1].lower() or "[no extension]"
        for doc in documents
    )
    folder_counts = Counter(
        PurePosixPath(doc["file_path"].replace("\\", "/")).parts[0]
        for doc in documents
        if PurePosixPath(doc["file_path"].replace("\\", "/")).parts
    )
    entrypoints = [
        doc["file_path"]
        for doc in documents
        if os.path.basename(doc["file_path"]) in ENTRYPOINT_NAMES
    ]
    largest_files = sorted(
        (
            {
                "file_path": doc["file_path"],
                "lines": doc["content"].count("\n") + 1,
                "characters": len(doc["content"]),
            }
            for doc in documents
        ),
        key=lambda item: item["characters"],
        reverse=True,
    )[:8]

    languages = [
        {"name": extension_label(ext), "extension": ext, "files": count}
        for ext, count in extension_counts.most_common()
    ]

    return {
        "file_count": len(documents),
        "chunk_count": len(chunks),
        "languages": languages,
        "top_folders": [
            {"name": folder, "files": count}
            for folder, count in folder_counts.most_common(10)
        ],
        "frameworks": detect_frameworks(documents),
        "entrypoints": entrypoints[:12],
        "largest_files": largest_files,
        "dependency_signals": build_dependency_signals(documents),
    }


def create_onboarding_summary(architecture: dict) -> list[str]:
    languages = ", ".join(item["name"] for item in architecture["languages"][:4]) or "text/code"
    frameworks = ", ".join(architecture["frameworks"]) or "no obvious framework detected"
    entrypoints = ", ".join(architecture["entrypoints"][:4]) or "no standard entrypoint detected"
    folders = ", ".join(item["name"] for item in architecture["top_folders"][:5]) or "repo root"

    return [
        f"This repo has {architecture['file_count']} indexed files split into {architecture['chunk_count']} searchable chunks.",
        f"Primary detected stack: {languages}. Framework signals: {frameworks}.",
        f"Start reading from: {entrypoints}.",
        f"Important top-level areas: {folders}.",
        "Use chat questions with file names, features, or flows; answers include the exact retrieved source snippets.",
    ]


def suggest_questions(architecture: dict) -> list[str]:
    suggestions = [
        "Explain this repository like I just joined the team.",
        "What are the main modules and how do they connect?",
        "Where should I start reading the code?",
        "Find risky code or TODOs that may need attention.",
    ]

    language_names = {item["name"].lower() for item in architecture["languages"]}
    for key, questions in QUESTION_BANK.items():
        if any(key in name.lower() for name in language_names):
            suggestions.extend(questions)

    for entrypoint in architecture["entrypoints"][:3]:
        suggestions.append(f"What happens in {entrypoint}?")

    unique = []
    for question in suggestions:
        if question not in unique:
            unique.append(question)
    return unique[:12]


def calculate_confidence(results: list[dict]) -> str:
    if not results:
        return "low"
    best = results[0].get("score", 0)
    if best >= 0.55:
        return "high"
    if best >= 0.35:
        return "medium"
    return "low"


def extract_symbols(content: str, extension: str) -> list[str]:
    if extension == ".py":
        return re.findall(r"^\s*(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", content, re.MULTILINE)[:6]
    if extension in {".js", ".jsx", ".ts", ".tsx"}:
        return re.findall(
            r"(?:function|class|const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)",
            content,
        )[:6]
    return []


def generate_free_answer(question: str, results: list[dict]) -> dict:
    if not results:
        return {
            "answer": "I could not find matching code in the indexed repository for that question.",
            "confidence": "low",
            "sources": [],
        }

    confidence = calculate_confidence(results)
    top = results[0]
    symbols = extract_symbols(top["content"], top.get("extension", ""))
    symbol_text = f" The most relevant symbols here are: {', '.join(symbols)}." if symbols else ""

    source_lines = []
    for item in results[:3]:
        source_lines.append(
            f"- {item['file_path']} lines {item['start_line']}-{item['end_line']} "
            f"(score {item['score']:.2f})"
        )

    answer = (
        f"Best match: {top['file_path']} lines {top['start_line']}-{top['end_line']}."
        f"{symbol_text}\n\n"
        "Why this is relevant: the semantic search found this code closest to your question. "
        "Review the cited snippets below for the exact implementation context.\n\n"
        "Sources:\n" + "\n".join(source_lines)
    )

    return {
        "answer": answer,
        "confidence": confidence,
        "sources": results,
    }


def find_risk_signals(documents: list[dict], limit: int = 20) -> list[dict]:
    patterns = {
        "TODO/FIXME": re.compile(r"\b(TODO|FIXME|HACK)\b", re.IGNORECASE),
        "Broad exception": re.compile(r"except\s+Exception|catch\s*\([^)]*\)", re.IGNORECASE),
        "Possible secret": re.compile(r"(api[_-]?key|secret|password|token)\s*[:=]", re.IGNORECASE),
        "Debug print/log": re.compile(r"\b(print\(|console\.log\()", re.IGNORECASE),
    }
    findings = []
    for doc in documents:
        lines = doc["content"].splitlines()
        for line_no, line in enumerate(lines, start=1):
            for label, pattern in patterns.items():
                if pattern.search(line):
                    findings.append(
                        {
                            "type": label,
                            "file_path": doc["file_path"],
                            "line": line_no,
                            "preview": line.strip()[:180],
                        }
                    )
                    break
            if len(findings) >= limit:
                return findings
    return findings
