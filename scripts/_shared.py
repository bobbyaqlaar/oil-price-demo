"""
scripts/_shared.py — small helpers duplicated, byte-for-byte in most cases,
across most of scripts/*.py before this consolidation. Import directly
(`from _shared import _repo_root`) — every scripts/*.py file is always
invoked as `python3 scripts/whatever.py`, which puts this directory on
sys.path[0] automatically, the same mechanism scripts/run-evals.py and
scripts/shadow-eval.py already rely on to import scripts/eval_judge.py.

Deliberately NOT shared with runtime/llm_gateway.py's own copy of
_repo_root() — runtime/ is vendored/deployed independently of scripts/
(a tenant repo can carry runtime/ without scripts/ at all), so importing
from here would create a coupling that breaks that independence. The
duplication between scripts/ and runtime/ is a real architectural
boundary, not an oversight; only the duplication *within* scripts/ is
consolidated here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def _repo_root() -> Path:
    """Walk up from cwd until .git is found; fall back to cwd."""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / ".git").exists():
            return parent
    return cwd


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tenant_id() -> Optional[str]:
    """Read tenant.id from .agenticframework/tenant.yaml if present.
    Prefers a real YAML parse (handles any valid tenant.yaml shape);
    falls back to a line-regex scan if PyYAML isn't installed, since
    several scripts/*.py callers run in minimal environments."""
    tenant_file = _repo_root() / ".agenticframework" / "tenant.yaml"
    if not tenant_file.exists():
        return None
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(tenant_file.read_text())
        return (data or {}).get("tenant", {}).get("id")
    except ImportError:
        try:
            for line in tenant_file.read_text().splitlines():
                if line.strip().startswith("id:"):
                    return line.split(":", 1)[1].strip()
        except Exception:  # fail-open: best-effort tenant-id lookup; None is a valid "no tenant" result, same as the yaml-parse path below
            pass
        return None
    except Exception:
        return None


def _phoenix_get(
    phoenix_endpoint: str, path: str, params: Optional[dict] = None
) -> Any:
    """GET against a Phoenix REST endpoint. Raises RuntimeError with the
    failing path in the message on any error — callers get a useful
    message without each having to wrap this themselves."""
    import httpx

    url = f"{phoenix_endpoint.rstrip('/')}{path}"
    try:
        resp = httpx.get(url, params=params, timeout=30.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        raise RuntimeError(f"Phoenix API error [{path}]: {exc}") from exc


def _phoenix_post(phoenix_endpoint: str, path: str, body: dict) -> Any:
    import httpx

    url = f"{phoenix_endpoint.rstrip('/')}{path}"
    try:
        resp = httpx.post(url, json=body, timeout=30.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        raise RuntimeError(f"Phoenix API error [{path}]: {exc}") from exc
