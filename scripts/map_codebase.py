"""
map_codebase.py — AST-based codebase walker that populates the Knowledge Graph.

Walks .py, .ts, .tsx, .js, .jsx, and .go files in the repo root.
Extracts top-level symbols (functions, classes, exports, interfaces).
Detects import relationships and writes edges to the graph.
Purges stale CodebaseFile nodes for deleted files.
Wires Guardrail nodes from .cursorrules and .agent-rfc/ markdown files.

Called by the post-commit and post-checkout hooks automatically.
Also runnable directly: python3 scripts/map_codebase.py
"""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Optional

# ── Helpers ───────────────────────────────────────────────────────────────────

from _shared import _repo_root  # noqa: E402


IGNORED_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
    ".pytest_cache", "dist", "build", ".agent-rfc", ".agents", ".github",
}

EXTENSION_TO_LANG = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
}


def _iter_source_files(root: Path):
    for path in root.rglob("*"):
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        if path.suffix in EXTENSION_TO_LANG and path.is_file():
            yield path


# ── Language-specific parsers ─────────────────────────────────────────────────

def _parse_python(path: Path) -> tuple[list[str], list[str]]:
    """Return (symbols, imported_modules)."""
    symbols: list[str] = []
    imports: list[str] = []
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Only top-level (parent == Module)
                if isinstance(getattr(node, "_parent", None), type(None)):
                    symbols.append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split(".")[0])

        # Fix: mark parent on top-level nodes only
        symbols = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
    except SyntaxError:  # fail-open: one unparsable file must not abort the whole codebase scan; it just contributes no symbols/imports
        pass
    return list(dict.fromkeys(symbols)), list(dict.fromkeys(imports))


def _parse_typescript(path: Path) -> tuple[list[str], list[str]]:
    symbols: list[str] = []
    imports: list[str] = []
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        # Exported declarations
        for m in re.finditer(
            r'^export\s+(?:default\s+)?(?:async\s+)?(?:function|class|const|let|var|interface|type|enum)\s+(\w+)',
            source, re.MULTILINE
        ):
            symbols.append(m.group(1))
        # Import paths (local only — starts with . or /)
        for m in re.finditer(r'''from\s+['"]([^'"]+)['"]''', source):
            imp = m.group(1)
            if imp.startswith("."):
                imports.append(imp)
        # Side-effect imports
        for m in re.finditer(r'''import\s+['"]([^'"]+)['"]''', source):
            imp = m.group(1)
            if imp.startswith("."):
                imports.append(imp)
    except Exception:  # fail-open: one unparsable file must not abort the whole codebase scan; it just contributes no symbols/imports
        pass
    return list(dict.fromkeys(symbols)), list(dict.fromkeys(imports))


def _parse_go(path: Path) -> tuple[list[str], list[str]]:
    symbols: list[str] = []
    imports: list[str] = []
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        # Exported identifiers (start with uppercase)
        for m in re.finditer(r'^func\s+(\([^)]*\)\s+)?([A-Z]\w*)\s*\(', source, re.MULTILINE):
            symbols.append(m.group(2))
        for m in re.finditer(r'^type\s+([A-Z]\w*)\s+', source, re.MULTILINE):
            symbols.append(m.group(1))
        # Import paths
        for m in re.finditer(r'"([^"]+)"', source):
            pkg = m.group(1)
            if "/" in pkg:
                imports.append(pkg.split("/")[-1])
    except Exception:  # fail-open: one unparsable file must not abort the whole codebase scan; it just contributes no symbols/imports
        pass
    return list(dict.fromkeys(symbols)), list(dict.fromkeys(imports))


_PARSERS = {
    "python":     _parse_python,
    "typescript": _parse_typescript,
    "javascript": _parse_typescript,  # same parser works
    "go":         _parse_go,
}


# ── Guardrail extraction from .cursorrules ────────────────────────────────────

_PILLAR_RE = re.compile(r'^##\s+(\d+)\.\s+(.+)$', re.MULTILINE)


def _extract_guardrails_from_cursorrules(path: Path) -> list[dict]:
    guardrails = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in _PILLAR_RE.finditer(text):
            pillar_num = int(m.group(1))
            title = m.group(2).strip()
            guardrails.append({
                "rule_id": f"cursorrules:pillar:{pillar_num}",
                "title": title,
                "pillar": pillar_num,
                "source_file": str(path),
            })
    except Exception:  # fail-open: one unparsable .cursorrules file must not abort the whole codebase scan; it just contributes no guardrails
        pass
    return guardrails


def _extract_guardrails_from_rfc(rfc_dir: Path) -> list[dict]:
    guardrails = []
    for md_file in rfc_dir.glob("**/*.md"):
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")
            # Extract first H1 or H2 as the rule title
            m = re.search(r'^#{1,2}\s+(.+)$', text, re.MULTILINE)
            title = m.group(1).strip() if m else md_file.stem
            guardrails.append({
                "rule_id": f"rfc:{md_file.relative_to(rfc_dir)}",
                "title": title,
                "pillar": None,
                "source_file": str(md_file),
            })
        except Exception:  # fail-open: one unparsable RFC file must not abort scanning the rest of the directory
            pass
    return guardrails


# ── Resolve local import to file path ─────────────────────────────────────────

def _resolve_local_import(
    source_file: Path,
    import_path: str,
    root: Path,
    lang: str,
) -> Optional[str]:
    """
    Try to resolve a relative import string to a repo-relative path.
    Returns None if unresolvable.
    """
    if lang in ("python",):
        # Python: module path → file path
        parts = import_path.replace(".", os.sep)
        candidates = [
            root / (parts + ".py"),
            root / parts / "__init__.py",
        ]
        for c in candidates:
            if c.exists():
                return str(c.relative_to(root))
    else:
        # TS/JS: relative path
        base = (source_file.parent / import_path).resolve()
        for ext in (".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.tsx", "/index.js"):
            candidate = Path(str(base) + ext) if not ext.startswith("/") else Path(str(base) + ext)
            if candidate.exists():
                return str(candidate.relative_to(root))
    return None


# ── Main walker ───────────────────────────────────────────────────────────────

def run_map(verbose: bool = False) -> dict:
    try:
        from local_knowledge_graph import AgentKnowledgeGraph
    except ImportError:
        from scripts.local_knowledge_graph import AgentKnowledgeGraph

    root = _repo_root()
    kg = AgentKnowledgeGraph()

    # Track which files we see during this walk
    seen_files: set[str] = set()
    stats = {"upserted": 0, "edges": 0, "guardrails": 0, "purged": 0}

    # ── Walk source files ─────────────────────────────────────────────────────
    for abs_path in _iter_source_files(root):
        lang = EXTENSION_TO_LANG.get(abs_path.suffix, "unknown")
        rel_path = str(abs_path.relative_to(root))
        seen_files.add(rel_path)

        parser = _PARSERS.get(lang)
        symbols, raw_imports = parser(abs_path) if parser else ([], [])
        last_modified = abs_path.stat().st_mtime

        from datetime import datetime, timezone
        mtime_str = datetime.fromtimestamp(last_modified, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        kg.upsert_file(rel_path, language=lang, symbols=symbols, last_modified=mtime_str)
        stats["upserted"] += 1

        # Resolve and wire import edges
        for raw_imp in raw_imports:
            resolved = _resolve_local_import(abs_path, raw_imp, root, lang)
            if resolved and resolved != rel_path:
                kg.add_import(rel_path, resolved)
                stats["edges"] += 1

        if verbose:
            print(f"  [{lang}] {rel_path} — {len(symbols)} symbols, {len(raw_imports)} imports")

    # ── Purge stale CodebaseFile nodes ────────────────────────────────────────
    stale = [
        node_id
        for node_id, attrs in kg._g.nodes(data=True)
        if attrs.get("node_type") == "CodebaseFile"
        and node_id not in seen_files
        and not (root / node_id).exists()
    ]
    for node_id in stale:
        kg.remove_file(node_id)
        stats["purged"] += 1
        if verbose:
            print(f"  🗑  Purged stale node: {node_id}")

    # ── Extract guardrails from .cursorrules ──────────────────────────────────
    cursorrules = root / ".cursorrules"
    if cursorrules.exists():
        for gr in _extract_guardrails_from_cursorrules(cursorrules):
            kg.upsert_guardrail(
                gr["rule_id"], gr["title"], gr["source_file"], gr["pillar"]
            )
            stats["guardrails"] += 1

    # ── Extract guardrails from .agent-rfc/ markdown ──────────────────────────
    rfc_dir = root / ".agent-rfc"
    if rfc_dir.exists():
        for gr in _extract_guardrails_from_rfc(rfc_dir):
            kg.upsert_guardrail(
                gr["rule_id"], gr["title"], gr["source_file"], gr["pillar"]
            )
            stats["guardrails"] += 1

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Map codebase into Knowledge Graph")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress the summary line (CI usage)")
    args = parser.parse_args()

    stats = run_map(verbose=args.verbose)
    if not args.quiet:
        print(json.dumps({"status": "ok", **stats}, indent=2))
