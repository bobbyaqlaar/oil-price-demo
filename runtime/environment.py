"""
runtime/environment.py — canonical $ENVIRONMENT resolver.

Before this module existed, runtime/trace_redactor.py and
scripts/multi_agent_system.py each read os.environ.get("ENVIRONMENT", ...)
independently, with different fallback philosophy for the same missing/
unrecognized-value condition: trace_redactor.py defaulted to "development"
(the least restrictive redaction profile — full unredacted payloads
exported), while multi_agent_system.py's checkpointer selector only hard-
errors when ENVIRONMENT is explicitly "staging" or "production" with no
DATABASE_URL set — an unrecognized value (e.g. a typo'd "produciton") fell
through to the same permissive MemorySaver path with nothing louder than a
stderr warning (FIXES_AND_CLEANUP.md 2.8).

get_environment() is fail-closed: missing or unrecognized values resolve to
"production" (the most restrictive profile/path), not "development". Both
callers should import this instead of reading os.environ directly.
"""

from __future__ import annotations

import os

_ALIASES = {
    "development": "development",
    "dev": "development",
    "testing": "development",
    "test": "development",
    "staging": "staging",
    "stage": "staging",
    "production": "production",
    "prod": "production",
}


def get_environment() -> str:
    """Returns one of "development", "staging", "production".

    Fail-closed: an empty/unset ENVIRONMENT, or a value that doesn't match a
    known alias, resolves to "production" — never silently to "development".
    """
    raw = os.environ.get("ENVIRONMENT", "").strip().lower()
    return _ALIASES.get(raw, "production")
