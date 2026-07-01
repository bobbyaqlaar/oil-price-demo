"""
runtime/dead_letter.py — Dead-letter queue and replay API.

Failed activities that exhaust retries — or that hit a recoverable error a
human can fix (see runtime/workflows/base_workflow.py's
run_with_recoverable_step) — are moved here.
DLQ entries surface in the Ops Portal unresolved queue (`GET /api/dlq`,
`portal/lib/dlq.ts`, which expects exactly the `dlq_entries` schema created
below).

Operations:
  enqueue(payload, error, tenant_id, task_id=None,
          reason=None, workflow_id=None, gate_id=None)  — add failed task to DLQ
  list(tenant_id, limit, status)                          — list DLQ entries
  replay(task_id, override_payload=None)                  — re-submit to workflow engine
  discard(task_id)                                         — mark resolved, remove from active queue

Backend: Postgres only (recommended for auditability — see SPECS.md §25).

Replay is workflow-engine-specific and intentionally pluggable: pass a
`replay_handler` callable to the constructor to actually re-enqueue to
Temporal/Celery — see runtime/temporal_replay.py for the concrete Temporal
implementation, which signals the *live, still-parked* workflow identified
by `workflow_id` rather than restarting a terminated one. Without a
handler, replay() marks the entry `status="replayed"` and records the
attempt without re-running anything — useful for manual review workflows,
but `replay_handler` is required for true automatic replay.

`reason` is a structured failure category (see REASONS below) so the Ops
Portal can render "needs a human decision" differently from "needs an
engineer" — `error`'s free text alone doesn't let a UI distinguish those
without string-matching. `workflow_id`/`gate_id` identify which live
workflow and which specific gate within it this entry came from — needed
because a workflow can have multiple recoverable steps (sequential or
concurrent), so a single global signal isn't enough to route a fix back to
the right one (FIXES_AND_CLEANUP.md HITL/DLQ redesign).

On enqueue, posts to SLACK_WEBHOOK_URL/TEAMS_WEBHOOK_URL if configured —
same fail-open notify-don't-block philosophy as
.github/actions/rollback-notify — so a human is pinged the moment
something needs attention, not only when they happen to check `/dlq`.

See SPECS.md §25 for the full specification.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional
import uuid

logger = logging.getLogger(__name__)

# Structured failure categories — not exhaustive/closed (no DB CHECK
# constraint enforces this list), but these are the ones the framework's
# own callers use; the Ops Portal renders unknown values as-is.
REASON_VALIDATION_ERROR = (
    "validation_error"  # e.g. the CRM hallucinated-field-name case
)
REASON_TOOL_CALL_ERROR = "tool_call_error"
REASON_HITL_TIMEOUT = "hitl_timeout"
REASON_HITL_REJECTED = "hitl_rejected"
REASON_INFRA_ERROR = "infra_error"


def _notify(tenant_id: str, task_id: str, reason: Optional[str], error: str) -> None:
    """Best-effort Slack/Teams ping — never raises, never blocks enqueue()."""
    text = f"🔴 DLQ entry tenant={tenant_id} task={task_id} reason={reason or 'unspecified'}: {error}"
    for env_var in ("SLACK_WEBHOOK_URL", "TEAMS_WEBHOOK_URL"):
        url = os.environ.get(env_var)
        if not url:
            continue
        try:
            import httpx

            httpx.post(url, json={"text": text}, timeout=5.0)
        except Exception as exc:
            logger.debug("DLQ notify via %s failed: %s", env_var, exc)


@dataclass
class DLQEntry:
    task_id: str
    tenant_id: str
    payload: Any
    error: str
    reason: Optional[str] = None
    workflow_id: Optional[str] = None
    gate_id: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    replayed_at: Optional[str] = None
    discarded_at: Optional[str] = None
    status: str = "pending"  # pending | replayed | discarded


class DeadLetterQueue:
    """
    Dead-letter queue for failed production activities.

    Cheap to instantiate per call (e.g. once per Temporal activity
    invocation, as base_workflow.py's dlq_enqueue_activity and
    examples/oil-price-agent's dead_letter_activity both do) — the
    CREATE TABLE/ALTER TABLE migration below only actually runs once per
    DSN per process (cached in _MIGRATED_DSNS), not on every construction.
    Without that cache, every recoverable-step failure would run 4 DDL
    round-trips (CREATE TABLE IF NOT EXISTS + 3x ALTER TABLE ADD COLUMN IF
    NOT EXISTS) before the real INSERT — wasted latency, and DDL takes a
    brief table-level lock even when a no-op, so concurrent workers all
    constructing a fresh DeadLetterQueue at once would serialize on it.
    """

    _MIGRATED_DSNS: set = set()

    def __init__(
        self, replay_handler: Optional[Callable[["DLQEntry"], None]] = None
    ) -> None:
        self._dsn = os.environ["DATABASE_URL"]
        self._replay_handler = replay_handler
        if self._dsn not in DeadLetterQueue._MIGRATED_DSNS:
            self._migrate()
            DeadLetterQueue._MIGRATED_DSNS.add(self._dsn)

    def _migrate(self) -> None:
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS dlq_entries (
                        task_id      TEXT PRIMARY KEY,
                        tenant_id    TEXT NOT NULL,
                        payload      JSONB NOT NULL,
                        error        TEXT NOT NULL,
                        reason       TEXT,
                        workflow_id  TEXT,
                        gate_id      TEXT,
                        status       TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'replayed', 'discarded')),
                        created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                        replayed_at  TIMESTAMPTZ,
                        discarded_at TIMESTAMPTZ
                    )
                    """
                )
                # CREATE TABLE IF NOT EXISTS above is a no-op against an
                # already-existing table from before this column set —
                # same gotcha as portal/db/schema.sql's budget_cap_usd
                # (FIXES_AND_CLEANUP.md P2b) — ALTER is what actually
                # applies these columns to a pre-existing dlq_entries.
                cur.execute(
                    "ALTER TABLE dlq_entries ADD COLUMN IF NOT EXISTS reason TEXT"
                )
                cur.execute(
                    "ALTER TABLE dlq_entries ADD COLUMN IF NOT EXISTS workflow_id TEXT"
                )
                cur.execute(
                    "ALTER TABLE dlq_entries ADD COLUMN IF NOT EXISTS gate_id TEXT"
                )
        finally:
            conn.close()

    def _connect(self):
        import psycopg2  # type: ignore

        return psycopg2.connect(self._dsn)

    @staticmethod
    def _row_to_entry(row: tuple) -> DLQEntry:
        (
            task_id,
            tenant_id,
            payload,
            error,
            reason,
            workflow_id,
            gate_id,
            status,
            created_at,
            replayed_at,
            discarded_at,
        ) = row
        return DLQEntry(
            task_id=task_id,
            tenant_id=tenant_id,
            payload=payload,
            error=error,
            reason=reason,
            workflow_id=workflow_id,
            gate_id=gate_id,
            status=status,
            created_at=created_at.isoformat(),
            replayed_at=replayed_at.isoformat() if replayed_at else None,
            discarded_at=discarded_at.isoformat() if discarded_at else None,
        )

    def enqueue(
        self,
        payload: Any,
        error: str,
        tenant_id: str,
        task_id: Optional[str] = None,
        reason: Optional[str] = None,
        workflow_id: Optional[str] = None,
        gate_id: Optional[str] = None,
    ) -> DLQEntry:
        """Add a failed task to the DLQ.

        Idempotent on task_id: ON CONFLICT DO NOTHING means a caller that
        retries enqueue() for the same logical failure (e.g. Temporal
        retrying the activity that calls enqueue(), before that activity
        returns successfully) gets back the original entry instead of a
        duplicate row — protects every caller, not just ones that happen
        to pass a stable task_id themselves.

        workflow_id/gate_id, when provided, mean this entry can be
        resumed (not just replayed-from-scratch) — see
        runtime/workflows/base_workflow.py's run_with_recoverable_step and
        runtime/temporal_replay.py, which signals the live workflow at
        workflow_id rather than restarting a terminated one.
        """
        entry = DLQEntry(
            task_id=task_id or str(uuid.uuid4()),
            tenant_id=tenant_id,
            payload=payload,
            error=error,
            reason=reason,
            workflow_id=workflow_id,
            gate_id=gate_id,
        )
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO dlq_entries (task_id, tenant_id, payload, error, reason, workflow_id, gate_id)
                    VALUES (%s, %s, %s::jsonb, %s, %s, %s, %s)
                    ON CONFLICT (task_id) DO NOTHING
                    RETURNING task_id
                    """,
                    (
                        entry.task_id,
                        entry.tenant_id,
                        json.dumps(payload, default=str),
                        error,
                        reason,
                        workflow_id,
                        gate_id,
                    ),
                )
                inserted = cur.fetchone() is not None
        finally:
            conn.close()
        if inserted:
            _notify(tenant_id, entry.task_id, reason, error)
        return entry if inserted else (self._get(entry.task_id) or entry)

    def list(
        self,
        tenant_id: Optional[str] = None,
        limit: int = 100,
        status: str = "pending",
    ) -> List[DLQEntry]:
        """List DLQ entries, optionally filtered by tenant."""
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                if tenant_id:
                    cur.execute(
                        """
                        SELECT task_id, tenant_id, payload, error, reason, workflow_id, gate_id,
                               status, created_at, replayed_at, discarded_at
                        FROM dlq_entries WHERE status = %s AND tenant_id = %s
                        ORDER BY created_at DESC LIMIT %s
                        """,
                        (status, tenant_id, limit),
                    )
                else:
                    cur.execute(
                        """
                        SELECT task_id, tenant_id, payload, error, reason, workflow_id, gate_id,
                               status, created_at, replayed_at, discarded_at
                        FROM dlq_entries WHERE status = %s
                        ORDER BY created_at DESC LIMIT %s
                        """,
                        (status, limit),
                    )
                return [self._row_to_entry(row) for row in cur.fetchall()]
        finally:
            conn.close()

    def _get(self, task_id: str) -> Optional[DLQEntry]:
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT task_id, tenant_id, payload, error, reason, workflow_id, gate_id,
                           status, created_at, replayed_at, discarded_at
                    FROM dlq_entries WHERE task_id = %s
                    """,
                    (task_id,),
                )
                row = cur.fetchone()
                return self._row_to_entry(row) if row else None
        finally:
            conn.close()

    def replay(self, task_id: str, override_payload: Optional[Any] = None) -> None:
        """Re-submit a failed task to the workflow engine.

        Calls the constructor-supplied `replay_handler(entry)` if one was
        provided — that's where actual Temporal/Celery re-enqueueing
        happens, since this module has no opinion on which workflow engine
        a tenant uses. Without a handler, this only marks the entry
        replayed and logs the attempt — it does NOT re-run anything.

        override_payload: the CRM-style "operator edited the JSON in the
        Ops Portal" case — when provided, the entry passed to
        replay_handler carries this payload instead of the original
        failing one, and the DB row's payload column is updated to match
        (so the audit trail shows what was actually sent on replay, not
        just what originally failed).
        """
        entry = self._get(task_id)
        if entry is None:
            raise KeyError(f"No DLQ entry with task_id={task_id!r}")

        if override_payload is not None:
            entry.payload = override_payload

        if self._replay_handler is not None:
            self._replay_handler(entry)
        else:
            logger.warning(
                "DeadLetterQueue.replay(%s) called with no replay_handler configured — "
                "marking replayed without re-enqueueing to any workflow engine.",
                task_id,
            )

        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                if override_payload is not None:
                    cur.execute(
                        "UPDATE dlq_entries SET status = 'replayed', replayed_at = now(), payload = %s::jsonb WHERE task_id = %s",
                        (json.dumps(override_payload, default=str), task_id),
                    )
                else:
                    cur.execute(
                        "UPDATE dlq_entries SET status = 'replayed', replayed_at = now() WHERE task_id = %s",
                        (task_id,),
                    )
        finally:
            conn.close()

    def discard(self, task_id: str) -> None:
        """Mark a DLQ entry as resolved and remove it from the active queue."""
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE dlq_entries SET status = 'discarded', discarded_at = now() WHERE task_id = %s",
                    (task_id,),
                )
                if cur.rowcount == 0:
                    raise KeyError(f"No DLQ entry with task_id={task_id!r}")
        finally:
            conn.close()
