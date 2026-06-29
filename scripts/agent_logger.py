"""
agent_logger.py — AgentSmith structured log writer.

Writes JSON-Lines to stdout and appends to .agent-history.log in the repo root.
Enforces 4-level severity model: INFO / MINOR / MAJOR / CRITICAL.
  - INFO / MINOR: pruned via FIFO at 10,000 entries (post-commit hook handles pruning).
  - MAJOR / CRITICAL: never pruned until hitl_resolved: true.

Calls audit_token_velocity_circuit() on every LLM invocation log entry.
All entries carry full agent identity and project attribution.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from typing import Any, Literal, Optional

# ── Types ─────────────────────────────────────────────────────────────────────

Level = Literal["INFO", "MINOR", "MAJOR", "CRITICAL"]

# ── Helpers ───────────────────────────────────────────────────────────────────
# _repo_root/_iso_now/_tenant_id used to be defined here; consolidated into
# _shared.py since they were byte-for-byte duplicated across most of
# scripts/*.py (this file was the canonical version _shared.py was lifted
# from — see that module's docstring for why it's not also shared with
# runtime/llm_gateway.py's separate copy).
from _shared import _repo_root, _iso_now, _tenant_id  # noqa: E402


def _project_name() -> str:
    root = _repo_root()
    remote = ""
    try:
        import subprocess
        remote = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, cwd=root
        ).stdout.strip()
        if remote:
            return remote.rstrip("/").split("/")[-1].removesuffix(".git")
    except Exception:  # fail-open: no git remote / not a git repo / git not installed all fall back to the dir name below
        pass
    return root.name


# ── Core logger ───────────────────────────────────────────────────────────────

class AgentLogger:
    """
    Structured logger for agent sessions.

    Usage:
        logger = AgentLogger(agent_name="Developer", agent_role="subagent",
                             orchestrator="Supervisor")
        logger.info("tool_invoked", tool="write_file", path="src/api.py")
        logger.major("empty_catch_detected", file="src/handler.py", line=42)
    """

    def __init__(
        self,
        agent_name: str,
        agent_role: Literal["orchestrator", "subagent", "standalone"] = "standalone",
        orchestrator: Optional[str] = None,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.orchestrator = orchestrator
        self.session_id = session_id or str(uuid.uuid4())
        self.model = model or os.environ.get("AGENT_DEFAULT_MODEL", "unknown")
        self.owner_id = os.environ.get("AGENT_OWNER_ID", "unknown")
        self.owner_name = os.environ.get("AGENT_OWNER_NAME", "unknown")
        self.project = _project_name()
        self.tenant_id = _tenant_id()
        self._log_path = _repo_root() / ".agent-history.log"

    # ── Public API ────────────────────────────────────────────────────────────

    def info(self, event: str, **kwargs: Any) -> dict:
        return self._write("INFO", event, **kwargs)

    def minor(self, event: str, **kwargs: Any) -> dict:
        return self._write("MINOR", event, **kwargs)

    def major(self, event: str, **kwargs: Any) -> dict:
        return self._write("MAJOR", event, **kwargs)

    def critical(self, event: str, **kwargs: Any) -> dict:
        return self._write("CRITICAL", event, **kwargs)

    def llm_call(
        self,
        event: str,
        input_tokens: int,
        output_tokens: int,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Log an LLM invocation and run circuit breaker check."""
        entry = self._write(
            "INFO",
            event,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model or self.model,
            **kwargs,
        )
        # Circuit breaker — import lazily to avoid circular deps
        try:
            from circuit_breaker import audit_token_velocity_circuit
            audit_token_velocity_circuit(input_tokens, output_tokens)
        except Exception:  # fail-open: circuit breaker is a side-effect check; it must never prevent the log entry above from being written
            pass
        return entry

    # ── Internal ──────────────────────────────────────────────────────────────

    def _write(self, level: Level, event: str, **kwargs: Any) -> dict:
        entry: dict[str, Any] = {
            "timestamp": _iso_now(),
            "level": level,
            "event": event,
            "agent": self.agent_name,
            "agent_role": self.agent_role,
            "session_id": self.session_id,
            "project": self.project,
            "model": self.model,
            "owner_id": self.owner_id,
            "owner_name": self.owner_name,
        }
        if self.tenant_id:
            entry["tenant_id"] = self.tenant_id
        if self.orchestrator:
            entry["orchestrator"] = self.orchestrator
        if level in ("MAJOR", "CRITICAL"):
            entry["hitl_resolved"] = False
            entry["hitl_resolved_by"] = None
            entry["hitl_resolved_at"] = None
        entry.update(kwargs)

        line = json.dumps(entry, default=str)

        # stdout
        print(line, flush=True)

        # append to .agent-history.log
        try:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:  # fail-open: read-only filesystem (CI without checkout) — stdout only
            pass

        return entry

    def resolve_hitl(self, event_filter: str, resolved_by: Optional[str] = None) -> int:
        """
        Mark all unresolved MAJOR/CRITICAL entries whose 'event' matches
        event_filter as resolved.  Returns the count of updated entries.
        """
        if not self._log_path.exists():
            return 0
        resolver = resolved_by or self.owner_id
        ts = _iso_now()
        updated = 0
        lines: list[str] = []
        with self._log_path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                    if (
                        entry.get("event") == event_filter
                        and entry.get("level") in ("MAJOR", "CRITICAL")
                        and not entry.get("hitl_resolved", True)
                    ):
                        entry["hitl_resolved"] = True
                        entry["hitl_resolved_by"] = resolver
                        entry["hitl_resolved_at"] = ts
                        raw = json.dumps(entry, default=str)
                        updated += 1
                except Exception:  # fail-open: one malformed JSON-lines entry must not abort resolving the rest; raw line is preserved unchanged below either way
                    pass
                lines.append(raw)
        with self._log_path.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        return updated

    def unresolved_issues(self) -> list[dict]:
        """Return all unresolved MAJOR/CRITICAL entries for the current project."""
        if not self._log_path.exists():
            return []
        results = []
        with self._log_path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                    if (
                        entry.get("project") == self.project
                        and entry.get("level") in ("MAJOR", "CRITICAL")
                        and not entry.get("hitl_resolved", True)
                    ):
                        results.append(entry)
                except Exception:  # fail-open: one malformed JSON-lines entry must not abort scanning the rest of the log for unresolved issues
                    pass
        return results


# ── Module-level convenience instance ─────────────────────────────────────────

_default_logger: Optional[AgentLogger] = None


def get_logger(
    agent_name: str = "Agent",
    agent_role: Literal["orchestrator", "subagent", "standalone"] = "standalone",
    orchestrator: Optional[str] = None,
    session_id: Optional[str] = None,
    model: Optional[str] = None,
) -> AgentLogger:
    global _default_logger
    if _default_logger is None:
        _default_logger = AgentLogger(
            agent_name=agent_name,
            agent_role=agent_role,
            orchestrator=orchestrator,
            session_id=session_id,
            model=model,
        )
    return _default_logger


# ── CLI: dump unresolved issues ───────────────────────────────────────────────

if __name__ == "__main__":
    logger = AgentLogger("CLI", "standalone")
    issues = logger.unresolved_issues()
    if not issues:
        print(json.dumps({"status": "clean", "project": logger.project}))
        sys.exit(0)
    for issue in issues:
        print(json.dumps(issue))
    sys.exit(1)
