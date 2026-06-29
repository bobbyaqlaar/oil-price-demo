"""
scripts/check_bare_except.py — AST-based detector for empty exception
handlers ("swallowed" errors), used by hooks/pre-commit's Guardrail 2.

Replaces a regex check (`except\\s*(\\w+)?:\\s*$`) that matched the *header
line* of almost any `except` clause — since Python's `except X:` line
always ends right at the colon regardless of how long or short the body
underneath is, that regex flagged ordinary multi-line handlers (e.g.
`except KeyError:\\n    raise ValueError(...) from None`) as "bare except
with no handler body" while doing nothing to verify the body was actually
empty. Parsing the real syntax tree instead of pattern-matching text lines
is the only way to tell "this handler's body is genuinely a no-op" from
"this handler's body just happens to start on the next line," which is
true of essentially all idiomatically-formatted Python.

A handler is flagged only when its body is exactly one statement and that
statement is `pass` or a bare `...` (Ellipsis) expression — i.e. it does
nothing at all. `except Exception: pass` is sometimes an intentional,
documented fail-open pattern (e.g. runtime/llm_gateway.py's
_record_span_attributes, where a tracing failure must never break the
actual LLM call) — append `# noqa: bare-except` to the `except` line to
opt that specific handler out, the same way the rest of this codebase
uses inline suppression comments rather than a blanket exemption list.
"""

from __future__ import annotations

import ast
import sys


def _is_noop_body(body: list[ast.stmt]) -> bool:
    if len(body) != 1:
        return False
    stmt = body[0]
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is Ellipsis:
        return True
    return False


def find_violations(source: str, path: str) -> list[tuple[str, int]]:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        return []  # not this checker's job to report syntax errors

    lines = source.splitlines()
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if not _is_noop_body(node.body):
            continue
        header_line = lines[node.lineno - 1] if 0 < node.lineno <= len(lines) else ""
        if "noqa: bare-except" in header_line:
            continue
        violations.append((path, node.lineno))
    return violations


def main(argv: list[str]) -> int:
    failed = False
    for path in argv:
        try:
            with open(path, encoding="utf-8") as f:
                source = f.read()
        except OSError:
            continue
        for file_path, lineno in find_violations(source, path):
            print(f"{file_path}:{lineno}: empty except handler (pass/... only) — log or re-raise, "
                  f"or append '# noqa: bare-except' to the except line if this is an intentional fail-open")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
