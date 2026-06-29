"""
runtime/workflows/base_workflow.py — Reference Temporal workflow base class.

Demonstrates the durable-execution pattern described in SPECS.md §25:
  - Activities call runtime/llm_gateway.py (never cost_router.py)
  - HITL pause/resume via workflow signal, with a timeout that routes to the DLQ
  - A generalized recoverable-step pattern (run_with_recoverable_step) where
    ANY activity failure — not just an explicit "needs review" gate — parks
    the workflow alive and waits for a human to edit the failing payload in
    the Ops Portal and replay it, e.g. a tool call that hallucinated a field
    name ({"account_status": "active"} when the schema expects "status")
    gets corrected and resumed in place, instead of failing the request and
    dead-lettering a payload nothing can act on
  - All spans carry tenant.id, workflow.id, workflow.run_id

This is a PATTERN, not a deployable workflow. Tenant repos copy and adapt this
shape into their own workflow files — see examples/oil-price-agent/workflows/
for a concrete domain example built on top of it. Framework workflows are
never deployed directly as tenant production code (§25).

Requires: pip install temporalio
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, Optional

try:
    from temporalio import activity, workflow
    from temporalio.common import RetryPolicy
    _HAS_TEMPORAL = True
except ImportError:
    _HAS_TEMPORAL = False
    RetryPolicy = None  # type: ignore

    class _Workflow:
        """No-op stand-in so this module is importable without temporalio installed."""
        def signal(self, fn=None, **_k):
            return fn if fn is not None else (lambda f: f)

    workflow = _Workflow()  # type: ignore
    activity = None  # type: ignore


HITL_SIGNAL_TIMEOUT = timedelta(hours=24)

# Bounds run_with_recoverable_step's retry loop — a human repeatedly
# submitting a fix that still fails (or fixing the wrong field) shouldn't
# keep a workflow parked forever; after this many failed attempts it
# dead-letters terminally even if a human is still actively trying.
RECOVERABLE_STEP_MAX_ATTEMPTS = 5


if _HAS_TEMPORAL:
    @activity.defn
    async def dlq_enqueue_activity(input: dict) -> dict:
        """Generic DLQ enqueue activity — wraps DeadLetterQueue.enqueue() so
        workflow code (which must stay deterministic/side-effect-free) never
        touches Postgres directly. Shared by run_with_recoverable_step;
        domain-specific dead-letter activities (e.g.
        examples/oil-price-agent/workflows/activities.py's
        dead_letter_activity) remain separate since they may carry
        domain-specific payload shaping the generic version shouldn't
        assume.
        """
        from dead_letter import DeadLetterQueue  # type: ignore

        dlq = DeadLetterQueue()
        entry = dlq.enqueue(
            payload=input["payload"],
            error=input["error"],
            tenant_id=input["tenant_id"],
            reason=input.get("reason"),
            workflow_id=input.get("workflow_id"),
            gate_id=input.get("gate_id"),
        )
        return {"task_id": entry.task_id}


@dataclass
class AgentWorkflowInput:
    tenant_id: str
    task: str
    spec: str
    workflow_run_id: str


@dataclass
class AgentWorkflowResult:
    status: str  # "success" | "failed" | "dead_letter"
    plan: str = ""
    code: str = ""
    validation: str = ""


class BaseAgentWorkflow:
    """
    Reference three-node pattern: Architect -> Developer -> Validator, with an
    optional HITL pause before a destructive/low-confidence step.

    Subclass and override `activities()` to bind domain-specific activity
    functions (e.g. IngestionActivity, PredictionActivity, DecisionActivity
    for the oil-price example) while keeping the HITL/DLQ control flow here.
    """

    def __init__(self) -> None:
        self._hitl_approved: Optional[bool] = None
        self._gate_fixes: Dict[str, Any] = {}

    if _HAS_TEMPORAL:
        @workflow.signal
        def hitl_approved(self, approved: bool) -> None:
            """External signal fired by the Phoenix annotation -> Ops Portal bridge on HITL review."""
            self._hitl_approved = approved

        @workflow.signal
        def human_fix_payload(self, gate_id: str, fix: Any) -> None:
            """Fired by the Ops Portal's DLQ "Replay with edits" action
            (via the tenant's own replay-webhook receiver — see
            runtime/temporal_replay.py) when a human corrects a failing
            payload and clicks Replay: the CRM example's
            {"account_status": "active"} -> {"status": "active"} fix.
            Keyed by gate_id, not a single shared field, so multiple
            recoverable steps in one workflow (sequential or concurrent)
            don't clobber each other's pending fix.
            """
            self._gate_fixes[gate_id] = fix

    async def run_with_hitl_gate(
        self,
        gate_activity_name: str,
        gate_input: Any,
        resume_activity_name: str,
        resume_input: Any,
        dead_letter_activity_name: str,
    ) -> AgentWorkflowResult:
        """
        Run `gate_activity_name`; if it requests human review, wait on the
        `hitl_approved` signal up to HITL_SIGNAL_TIMEOUT. On timeout, route to
        the dead-letter activity instead of blocking the workflow forever.
        """
        if not _HAS_TEMPORAL:
            raise RuntimeError(
                "temporalio is not installed. Run: pip install temporalio. "
                "See SPECS.md §25 for the production runtime spec."
            )

        gate_result = await workflow.execute_activity(
            gate_activity_name,
            gate_input,
            start_to_close_timeout=timedelta(minutes=10),
        )

        if not gate_result.get("needs_hitl"):
            return await workflow.execute_activity(
                resume_activity_name,
                resume_input,
                start_to_close_timeout=timedelta(minutes=10),
            )

        try:
            await workflow.wait_condition(
                lambda: self._hitl_approved is not None,
                timeout=HITL_SIGNAL_TIMEOUT,
            )
        except TimeoutError:
            await workflow.execute_activity(
                dead_letter_activity_name,
                {**gate_input, "error": "hitl_timeout"},
                start_to_close_timeout=timedelta(minutes=5),
            )
            return AgentWorkflowResult(status="dead_letter")

        if not self._hitl_approved:
            return AgentWorkflowResult(status="failed")

        return await workflow.execute_activity(
            resume_activity_name,
            resume_input,
            start_to_close_timeout=timedelta(minutes=10),
        )

    async def run_with_recoverable_step(
        self,
        activity_name: str,
        payload: Any,
        tenant_id: str,
        gate_id: str,
        reason: str = "validation_error",
        timeout: timedelta = HITL_SIGNAL_TIMEOUT,
        max_attempts: int = RECOVERABLE_STEP_MAX_ATTEMPTS,
    ) -> Any:
        """
        Run `activity_name` with `payload`. On ANY activity failure — not
        just an explicit "needs review" gate, e.g. a tool call that
        hallucinated a field name the way the CRM example does
        ({"account_status": "active"} where the schema expects "status")
        — this workflow stays ALIVE (it does not return/terminate) and
        parks on a per-gate signal up to `timeout`, waiting for a human to
        fix the payload via the Ops Portal's editable DLQ view.

        On a human_fix_payload signal for this gate_id: retries
        `activity_name` with the corrected payload. If that also fails,
        loops (new DLQ entry, waits again) up to `max_attempts` before
        giving up and dead-lettering terminally — bounds how long a
        workflow stays parked if a human keeps submitting fixes that
        don't actually fix the problem.

        On timeout with no fix at all: dead-letters terminally, same
        fallback behavior as run_with_hitl_gate.
        """
        if not _HAS_TEMPORAL:
            raise RuntimeError(
                "temporalio is not installed. Run: pip install temporalio. "
                "See SPECS.md §25 for the production runtime spec."
            )

        workflow_id = workflow.info().workflow_id
        current_payload = payload

        for attempt in range(max_attempts):
            try:
                return await workflow.execute_activity(
                    activity_name,
                    current_payload,
                    start_to_close_timeout=timedelta(minutes=10),
                    # maximum_attempts=1: Temporal's default retry policy
                    # retries indefinitely (with backoff) until
                    # start_to_close_timeout — pointless and slow for a
                    # validation/tool-call error, which won't succeed on
                    # retry without a different payload. THIS method's own
                    # for-loop is the retry mechanism (only after a human
                    # supplies a corrected payload), not Temporal's.
                    retry_policy=RetryPolicy(maximum_attempts=1),
                )
            except Exception as exc:
                error_text = str(exc)[:500]

            self._gate_fixes.pop(gate_id, None)
            await workflow.execute_activity(
                dlq_enqueue_activity,
                {
                    "payload": current_payload,
                    "error": error_text,
                    "tenant_id": tenant_id,
                    "reason": reason,
                    "workflow_id": workflow_id,
                    "gate_id": gate_id,
                },
                start_to_close_timeout=timedelta(minutes=5),
            )

            if attempt == max_attempts - 1:
                break

            try:
                await workflow.wait_condition(lambda: gate_id in self._gate_fixes, timeout=timeout)
            except TimeoutError:
                return AgentWorkflowResult(status="dead_letter")

            current_payload = self._gate_fixes.pop(gate_id)

        return AgentWorkflowResult(status="dead_letter")
