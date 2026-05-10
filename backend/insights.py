"""
Free codebase insight helpers.

These functions avoid paid LLM calls. They use file paths, extensions, imports,
and retrieved chunks to create useful onboarding and citation-rich answers.
"""

from __future__ import annotations

import ast
import os
import re
from collections import Counter
from pathlib import PurePosixPath


ENTRYPOINT_NAMES = {
    "main.py",
    "app.py",
    "application.py",
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

    architecture = {
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
    architecture["code_facts"] = build_code_facts(chunks, architecture)
    return architecture


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


def is_overview_question(question: str) -> bool:
    q = question.lower()
    phrases = [
        "what does this repo do",
        "what does this project do",
        "explain this repo",
        "explain this repository",
        "overview",
        "summary",
        "like i just joined",
    ]
    return any(phrase in q for phrase in phrases)


def is_main_code_question(question: str) -> bool:
    q = question.lower()
    phrases = [
        "main code",
        "entry point",
        "start reading",
        "where should i start",
        "main file",
        "application start",
    ]
    return any(phrase in q for phrase in phrases)


def is_prediction_question(question: str) -> bool:
    q = question.lower()
    phrases = [
        "what is being predicted",
        "what does it predict",
        "prediction output",
        "target variable",
        "model predict",
        "what is the model predicting",
    ]
    return any(phrase in q for phrase in phrases)


def route_lines(content: str) -> list[str]:
    routes = re.findall(r"@app\.route\(['\"]([^'\"]+)['\"](?:,\s*methods=\[([^\]]+)\])?", content)
    formatted = []
    seen = set()
    for path, methods in routes:
        method_text = methods.replace("'", "").replace('"', "") if methods else "GET"
        label = f"{path} ({method_text})"
        if label not in seen:
            formatted.append(label)
            seen.add(label)
    return formatted


def form_fields(content: str) -> list[str]:
    return re.findall(r'name=["\']([^"\']+)["\']', content)[:15]


def template_outputs(content: str) -> list[str]:
    outputs = re.findall(r"\{\{\s*([^}\s]+)\s*\}\}", content)
    labels = re.findall(r"<h[1-6][^>]*>\s*([^<{}]+?)\s*\{\{", content, flags=re.IGNORECASE)
    found = [item.strip() for item in labels + outputs if item.strip()]
    unique = []
    for item in found:
        if item not in unique:
            unique.append(item)
    return unique[:10]


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "being", "by", "do", "does",
    "for", "from", "here", "how", "i", "in", "is", "it", "main", "of", "on",
    "or", "repo", "repository", "the", "this", "to", "what", "where", "which",
    "with", "work", "works",
}


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", text.lower())
        if token not in STOPWORDS
    ]


def infer_intent(question: str) -> str:
    q = question.lower()
    if any(word in q for word in ["predict", "prediction", "target", "model output"]):
        return "prediction"
    if any(word in q for word in ["route", "endpoint", "api", "url"]):
        return "routes"
    if any(word in q for word in ["form", "input", "field", "parameter", "takes", "collect"]):
        return "frontend"
    if any(word in q for word in ["run", "start", "install", "setup", "command"]):
        return "setup"
    if any(word in q for word in ["database", "db", "schema", "table", "sql"]):
        return "database"
    if any(word in q for word in ["component", "page", "ui", "frontend", "screen"]):
        return "frontend"
    if any(word in q for word in ["risk", "bug", "todo", "fixme", "security", "issue"]):
        return "risk"
    if is_main_code_question(question):
        return "entrypoint"
    if is_overview_question(question):
        return "overview"
    return "general"


def extract_routes_from_chunks(metadata: list[dict]) -> list[dict]:
    routes = []
    seen = set()
    for chunk in metadata:
        content = chunk.get("content", "")
        for match in re.finditer(
            r"@app\.route\(['\"]([^'\"]+)['\"](?:,\s*methods=\[([^\]]+)\])?\)\s*\n\s*def\s+([A-Za-z_][A-Za-z0-9_]*)",
            content,
        ):
            path, methods, function = match.groups()
            method_list = [m.strip().strip("'\"") for m in methods.split(",")] if methods else ["GET"]
            key = (chunk["file_path"], path, function)
            if key not in seen:
                routes.append(
                    {
                        "path": path,
                        "methods": method_list,
                        "function": function,
                        "file_path": chunk["file_path"],
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                    }
                )
                seen.add(key)
    return routes


def extract_symbols_from_chunks(metadata: list[dict]) -> list[dict]:
    symbols = []
    seen = set()
    for chunk in metadata:
        extension = chunk.get("extension", "")
        content = chunk.get("content", "")
        if extension == ".py":
            pattern = r"^\s*(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)"
        elif extension in {".js", ".jsx", ".ts", ".tsx"}:
            pattern = r"(function|class|const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)"
        else:
            continue
        for kind, name in re.findall(pattern, content, flags=re.MULTILINE):
            key = (chunk["file_path"], kind, name)
            if key not in seen:
                symbols.append(
                    {
                        "kind": kind,
                        "name": name,
                        "file_path": chunk["file_path"],
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                    }
                )
                seen.add(key)
    return symbols[:80]


def extract_forms_from_chunks(metadata: list[dict]) -> list[dict]:
    forms = []
    for chunk in metadata:
        content = chunk.get("content", "")
        fields = form_fields(content)
        if fields:
            action_match = re.search(r"<form[^>]+action=[\"']([^\"']+)[\"']", content, re.IGNORECASE)
            method_match = re.search(r"<form[^>]+method=[\"']([^\"']+)[\"']", content, re.IGNORECASE)
            forms.append(
                {
                    "file_path": chunk["file_path"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "action": action_match.group(1) if action_match else "",
                    "method": method_match.group(1).upper() if method_match else "GET",
                    "fields": fields,
                    "outputs": template_outputs(content),
                }
            )
    return forms


def extract_model_facts_from_chunks(metadata: list[dict]) -> list[dict]:
    models = []
    seen = set()
    for chunk in metadata:
        content = chunk.get("content", "")
        for variable, path in re.findall(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*pickle\.load\(open\(['\"]([^'\"]+)['\"]",
            content,
        ):
            key = (variable, path, chunk["file_path"])
            if key not in seen:
                models.append(
                    {
                        "name": variable,
                        "loaded_from": path,
                        "file_path": chunk["file_path"],
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                        "predict_called": f"{variable}.predict" in content,
                    }
                )
                seen.add(key)
        for variable in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\.predict\(", content):
            key = (variable, "predict", chunk["file_path"])
            if key not in seen:
                models.append(
                    {
                        "name": variable,
                        "loaded_from": "",
                        "file_path": chunk["file_path"],
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                        "predict_called": True,
                    }
                )
                seen.add(key)
    return models


def extract_setup_hints_from_chunks(metadata: list[dict]) -> list[dict]:
    hints = []
    patterns = [
        r"pip install[^\n`]*",
        r"uvicorn\s+[^\n`]*",
        r"python\s+[A-Za-z0-9_./\\-]+\.py",
        r"npm\s+(?:install|run|start|dev)[^\n`]*",
        r"flask\s+run[^\n`]*",
        r"docker compose[^\n`]*",
    ]
    for chunk in metadata:
        content = chunk.get("content", "")
        found = []
        for pattern in patterns:
            found.extend(re.findall(pattern, content, flags=re.IGNORECASE))
        if found:
            hints.append(
                {
                    "file_path": chunk["file_path"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "commands": list(dict.fromkeys(item.strip() for item in found))[:8],
                }
            )
    return hints[:12]


def build_code_facts(metadata: list[dict], architecture: dict | None = None) -> dict:
    joined = "\n".join(chunk.get("content", "") for chunk in metadata)
    forms = extract_forms_from_chunks(metadata)
    outputs = []
    for form in forms:
        outputs.extend(form.get("outputs", []))
    return {
        "frameworks": (architecture or {}).get("frameworks", []),
        "entrypoints": (architecture or {}).get("entrypoints", []),
        "routes": extract_routes_from_chunks(metadata),
        "symbols": extract_symbols_from_chunks(metadata),
        "forms": forms,
        "models": extract_model_facts_from_chunks(metadata),
        "setup": extract_setup_hints_from_chunks(metadata),
        "template_outputs": list(dict.fromkeys(outputs + template_outputs(joined)))[:12],
        "form_fields": form_fields(joined),
    }


def hybrid_rerank(question: str, semantic_results: list[dict], metadata: list[dict], architecture: dict | None) -> list[dict]:
    query_tokens = set(tokenize(question))
    intent = infer_intent(question)
    semantic_by_key = {
        (item.get("file_path"), item.get("chunk_index")): item.get("score", 0)
        for item in semantic_results
    }
    entrypoints = set((architecture or {}).get("entrypoints") or [])

    ranked = []
    for chunk in metadata:
        content = chunk.get("content", "")
        path = chunk.get("file_path", "")
        key = (path, chunk.get("chunk_index"))
        chunk_tokens = set(tokenize(path + " " + content[:2500]))
        overlap = len(query_tokens & chunk_tokens)
        score = semantic_by_key.get(key, 0) * 4 + overlap * 0.45

        lowered = content.lower()
        path_lower = path.lower()
        if path in entrypoints:
            score += 2.5
        if os.path.basename(path) in ENTRYPOINT_NAMES:
            score += 1.5
        if intent == "routes" and ("@app.route" in content or "router." in lowered or "express" in lowered):
            score += 3
        if intent == "prediction" and (".predict(" in content or "pickle.load" in content or "prediction" in lowered):
            score += 4
        if intent == "frontend" and chunk.get("extension") in {".html", ".jsx", ".tsx", ".css"}:
            score += 2
        if intent == "setup" and ("readme" in path_lower or "requirements" in path_lower or "package.json" in path_lower):
            score += 3
        if intent == "database" and any(term in lowered for term in ["database", "sql", "db", "model", "schema"]):
            score += 3

        enriched = chunk.copy()
        enriched["score"] = float(score)
        enriched["semantic_score"] = float(semantic_by_key.get(key, 0))
        ranked.append(enriched)

    ranked.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(ranked[:12], start=1):
        item["rank"] = rank
    return ranked[:12]


def important_sources(metadata: list[dict], architecture: dict | None, limit: int = 4) -> list[dict]:
    if not metadata:
        return []

    entrypoints = set((architecture or {}).get("entrypoints") or [])
    scored = []
    for chunk in metadata:
        content = chunk.get("content", "")
        path = chunk.get("file_path", "")
        score = 0
        if path in entrypoints:
            score += 8
        if os.path.basename(path) in ENTRYPOINT_NAMES:
            score += 6
        if "@app.route" in content or "Flask(__name__)" in content:
            score += 5
        if "if __name__" in content:
            score += 4
        if "pickle.load" in content or ".predict(" in content:
            score += 3
        if chunk.get("extension") == ".py":
            score += 2
        scored.append((score, chunk))

    chosen = []
    seen_paths = set()
    for score, chunk in sorted(scored, key=lambda item: item[0], reverse=True):
        if score <= 0:
            continue
        if chunk["file_path"] in seen_paths:
            continue
        chosen.append(chunk)
        seen_paths.add(chunk["file_path"])
        if len(chosen) >= limit:
            break
    return chosen or metadata[:limit]


def compose_project_overview(metadata: list[dict], architecture: dict | None) -> str:
    architecture = architecture or {}
    frameworks = architecture.get("frameworks") or []
    languages = ", ".join(item["name"] for item in architecture.get("languages", [])[:4]) or "code"
    entrypoints = architecture.get("entrypoints") or []
    important = important_sources(metadata, architecture, limit=5)
    joined = "\n".join(chunk.get("content", "") for chunk in important)
    joined_all = "\n".join(chunk.get("content", "") for chunk in metadata)

    parts = []
    if "Flask" in frameworks and ("pickle.load" in joined_all or ".predict(" in joined_all):
        parts.append(
            "This looks like a Flask machine-learning web app. It loads saved model/scaler pickle files, "
            "shows HTML pages, accepts form inputs, scales the input values, runs a prediction, and renders "
            "the result back into a template."
        )
    elif "Flask" in frameworks:
        parts.append("This looks like a Flask web app with Python route handlers and HTML templates.")
    else:
        parts.append(f"This repository is mainly {languages}.")

    routes = route_lines(joined_all)
    if routes:
        parts.append("Important routes: " + ", ".join(routes[:6]) + ".")

    fields = form_fields(joined_all)
    if fields:
        parts.append("The user-facing form collects: " + ", ".join(fields[:12]) + ".")

    if entrypoints:
        parts.append("Start reading from: " + ", ".join(entrypoints[:5]) + ".")

    folders = ", ".join(item["name"] for item in architecture.get("top_folders", [])[:5])
    if folders:
        parts.append("Main areas in the repo: " + folders + ".")

    return "\n\n".join(parts)


def compose_main_code_answer(metadata: list[dict], architecture: dict | None) -> tuple[str, list[dict]]:
    sources = important_sources(metadata, architecture, limit=4)
    if not sources:
        return "I could not identify a main code file from the indexed chunks.", []

    primary = sources[0]
    content = primary.get("content", "")
    path = primary.get("file_path", "")
    routes = route_lines(content)
    symbols = extract_symbols(content, primary.get("extension", ""))

    details = [
        f"The main code appears to be in {path}, especially lines {primary['start_line']}-{primary['end_line']}.",
    ]
    if routes:
        details.append("It defines these routes: " + ", ".join(routes[:6]) + ".")
    if symbols:
        details.append("Important functions/classes here: " + ", ".join(symbols) + ".")
    if "pickle.load" in content:
        details.append("It also loads saved ML artifacts with pickle, so this file connects the web app to the model.")
    if ".predict(" in content:
        details.append("It runs prediction logic here, so this is the best place to understand the app behavior.")

    source_lines = [
        f"- {item['file_path']} lines {item['start_line']}-{item['end_line']}"
        for item in sources
    ]
    return "\n\n".join(details) + "\n\nSources:\n" + "\n".join(source_lines), sources


def compose_prediction_answer(metadata: list[dict], architecture: dict | None) -> tuple[str, list[dict]]:
    sources = important_sources(metadata, architecture, limit=5)
    joined_all = "\n".join(chunk.get("content", "") for chunk in metadata)
    fields = form_fields(joined_all)
    outputs = template_outputs(joined_all)

    model_names = []
    if "ridge_model" in joined_all:
        model_names.append("ridge_model")
    if "standard_scaler" in joined_all:
        model_names.append("standard_scaler")

    predicted_label = "the model output"
    if "FWI prediction" in joined_all or "FWI Prediction" in joined_all:
        predicted_label = "FWI, the Fire Weather Index"
    elif outputs:
        predicted_label = outputs[0]

    details = [
        f"It is predicting {predicted_label}.",
    ]

    if fields:
        details.append(
            "The prediction is based on these form inputs: "
            + ", ".join(fields[:12])
            + "."
        )

    if model_names:
        details.append(
            "The app loads "
            + ", ".join(model_names)
            + " from pickle files, scales the input values, then calls the model's predict method."
        )
    elif ".predict(" in joined_all:
        details.append("The app collects the form values and calls a model's predict method.")

    routes = route_lines(joined_all)
    prediction_routes = [route for route in routes if "predict" in route.lower() or "predicate" in route.lower()]
    if prediction_routes:
        details.append("The prediction happens through this route: " + ", ".join(prediction_routes) + ".")

    source_lines = [
        f"- {item['file_path']} lines {item['start_line']}-{item['end_line']}"
        for item in sources
    ]
    return "\n\n".join(details) + "\n\nSources:\n" + "\n".join(source_lines), sources


def source_lines(sources: list[dict]) -> str:
    if not sources:
        return ""
    return "\n".join(
        f"- {item['file_path']} lines {item['start_line']}-{item['end_line']}"
        + (f" (score {item['score']:.2f})" if "score" in item else "")
        for item in sources[:5]
    )


def compose_routes_answer(facts: dict, sources: list[dict]) -> str:
    routes = facts.get("routes", [])
    if not routes:
        return "I did not find explicit route definitions in the indexed code.\n\nSources:\n" + source_lines(sources)

    lines = ["I found these application routes/endpoints:"]
    for route in routes[:10]:
        methods = ",".join(route.get("methods", []))
        lines.append(
            f"- {route['path']} [{methods}] handled by {route['function']} in "
            f"{route['file_path']} lines {route['start_line']}-{route['end_line']}"
        )
    return "\n".join(lines) + "\n\nSources:\n" + source_lines(sources)


def compose_setup_answer(facts: dict, sources: list[dict]) -> str:
    hints = facts.get("setup", [])
    frameworks = facts.get("frameworks", [])
    routes = facts.get("routes", [])
    if not hints:
        if "Flask" in frameworks or routes:
            entry_files = list(dict.fromkeys(route["file_path"] for route in routes)) or ["application.py or app.py"]
            return (
                "I did not find README setup commands, but the code looks like a Flask app. "
                f"Try installing dependencies, then run the Flask entry file: {entry_files[0]}.\n\n"
                "Common commands to try:\n"
                f"- python {entry_files[0]}\n"
                "- flask run\n\n"
                "Sources:\n" + source_lines(sources)
            )
        return (
            "I did not find clear setup commands in the indexed files. Check README, requirements, "
            "package, Docker, or entrypoint files if they exist.\n\nSources:\n" + source_lines(sources)
        )
    lines = ["I found these likely setup/run commands:"]
    for hint in hints[:6]:
        lines.append(f"- From {hint['file_path']} lines {hint['start_line']}-{hint['end_line']}:")
        for command in hint.get("commands", [])[:5]:
            lines.append(f"  {command}")
    return "\n".join(lines) + "\n\nSources:\n" + source_lines(sources)


def compose_database_answer(facts: dict, sources: list[dict]) -> str:
    db_sources = [
        item for item in sources
        if re.search(r"database|sql|db|schema|model|table", item.get("content", ""), re.IGNORECASE)
    ]
    if not db_sources:
        return (
            "I did not find strong database or schema evidence in the retrieved code. "
            "This repo may not use a database, or the database code may be outside indexed files."
            "\n\nSources:\n" + source_lines(sources)
        )
    lines = ["The strongest database-related evidence is in:"]
    for item in db_sources[:5]:
        symbols = extract_symbols(item.get("content", ""), item.get("extension", ""))
        suffix = f" Symbols: {', '.join(symbols)}." if symbols else ""
        lines.append(f"- {item['file_path']} lines {item['start_line']}-{item['end_line']}.{suffix}")
    return "\n".join(lines) + "\n\nSources:\n" + source_lines(db_sources)


def compose_frontend_answer(facts: dict, sources: list[dict]) -> str:
    forms = facts.get("forms", [])
    html_sources = [item for item in sources if item.get("extension") in {".html", ".jsx", ".tsx", ".css"}]
    lines = []
    if forms:
        lines.append("The main user-facing form/template pieces I found are:")
        for form in forms[:5]:
            fields = ", ".join(form.get("fields", [])[:12])
            lines.append(
                f"- {form['file_path']} lines {form['start_line']}-{form['end_line']} "
                f"uses {form.get('method', 'GET')} and fields: {fields}"
            )
    elif html_sources:
        lines.append("The likely frontend/template code is in:")
        for item in html_sources[:5]:
            lines.append(f"- {item['file_path']} lines {item['start_line']}-{item['end_line']}")
    else:
        lines.append("I did not find obvious frontend files in the retrieved context.")
    return "\n".join(lines) + "\n\nSources:\n" + source_lines(html_sources or sources)


def compose_general_answer(question: str, facts: dict, sources: list[dict]) -> str:
    top = sources[0]
    content = top.get("content", "")
    symbols = extract_symbols(content, top.get("extension", ""))
    routes = route_lines(content)
    fields = form_fields(content)
    outputs = template_outputs(content)

    lines = [
        f"The strongest evidence is in {top['file_path']} lines {top['start_line']}-{top['end_line']}.",
    ]
    if symbols:
        lines.append("Relevant symbols: " + ", ".join(symbols) + ".")
    if routes:
        lines.append("Routes in this chunk: " + ", ".join(routes) + ".")
    if fields:
        lines.append("Form/input fields here: " + ", ".join(fields[:12]) + ".")
    if outputs:
        lines.append("Template/output hints: " + ", ".join(outputs[:8]) + ".")
    if ".predict(" in content:
        lines.append("This chunk calls a model prediction method, so it likely controls ML inference behavior.")
    if "pickle.load" in content:
        lines.append("This chunk loads saved pickle artifacts, likely model/scaler files.")

    lines.append(
        "I am grounding this answer in retrieved code evidence. If this is not enough, ask with a file name, "
        "function name, route, or feature keyword and I will narrow it down."
    )
    return "\n\n".join(lines) + "\n\nSources:\n" + source_lines(sources)


def compose_answer_by_intent(question: str, metadata: list[dict], architecture: dict | None, sources: list[dict]) -> dict:
    intent = infer_intent(question)
    facts = build_code_facts(metadata, architecture)

    if intent == "overview":
        return {
            "answer": compose_project_overview(metadata, architecture),
            "confidence": "medium",
            "sources": important_sources(metadata, architecture, limit=5),
        }
    if intent == "entrypoint":
        answer, chosen = compose_main_code_answer(metadata, architecture)
        return {"answer": answer, "confidence": "medium" if chosen else "low", "sources": chosen}
    if intent == "prediction":
        answer, chosen = compose_prediction_answer(metadata, architecture)
        return {"answer": answer, "confidence": "medium" if chosen else "low", "sources": chosen}
    if intent == "routes":
        return {"answer": compose_routes_answer(facts, sources), "confidence": "medium", "sources": sources}
    if intent == "setup":
        return {"answer": compose_setup_answer(facts, sources), "confidence": "medium", "sources": sources}
    if intent == "database":
        return {"answer": compose_database_answer(facts, sources), "confidence": calculate_confidence(sources), "sources": sources}
    if intent == "frontend":
        return {"answer": compose_frontend_answer(facts, sources), "confidence": "medium", "sources": sources}

    return {
        "answer": compose_general_answer(question, facts, sources),
        "confidence": calculate_confidence(sources),
        "sources": sources,
    }


def extract_symbols(content: str, extension: str) -> list[str]:
    if extension == ".py":
        return re.findall(r"^\s*(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", content, re.MULTILINE)[:6]
    if extension in {".js", ".jsx", ".ts", ".tsx"}:
        return re.findall(
            r"(?:function|class|const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)",
            content,
        )[:6]
    return []


def generate_free_answer(
    question: str,
    results: list[dict],
    metadata: list[dict] | None = None,
    architecture: dict | None = None,
) -> dict:
    metadata = metadata or results
    if not results:
        return {
            "answer": "I could not find matching code in the indexed repository for that question.",
            "confidence": "low",
            "sources": [],
        }

    return compose_answer_by_intent(question, metadata, architecture, results)

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
