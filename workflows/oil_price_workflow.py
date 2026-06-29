"""
examples/oil-price-agent/workflows/oil_price_workflow.py — Temporal reference
workflow for the oil price prediction pipeline.

Pipeline: IngestionAgent -> PredictionAgent -> DecisionAgent.

HITL triggers (see examples/oil-price-agent/agents/README.md):
  - Price anomaly > 3 standard deviations: pause workflow, alert ops team
  - Model confidence < 0.6: pause for human review before order

Actually subclasses runtime/workflows/base_workflow.py's BaseAgentWorkflow
(it didn't, for a while — this file used to reimplement the same
hitl_approved signal + wait_condition inline, which worked but meant this
"reference example" never actually demonstrated the base class it claimed
to be built on, and never exercised run_with_recoverable_step at all).
Two of the framework's patterns are demonstrated here, deliberately kept
distinct because they answer different questions:
  - run_with_hitl_gate-style approve/reject (still hand-rolled below using
    the INHERITED self._hitl_approved signal, not run_with_hitl_gate
    itself — that method's resume_input is a fixed value decided before
    the gate runs, but this pipeline's resume step needs the gate's OWN
    output (the prediction) as its input, a shape run_with_hitl_gate
    doesn't support) for "should this prediction be acted on at all."
  - run_with_recoverable_step for "decide_action_activity rejected this
    specific payload" — e.g. a malformed action shape — demonstrating the
    CRM-style edit-and-replay pattern (FIXES_AND_CLEANUP.md's HITL/DLQ
    redesign) on the order-placement step specifically, since that's the
    step a downstream system could plausibly reject on a bad field.

This is a reference implementation: copy this directory into your own
tenant repository rather than deploying it from AgentSmith/examples
(§28).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

# runtime/ and runtime/workflows/ must already be on sys.path before this
# module is imported — done by the importer (worker.py for real deployment;
# test fixtures the same way), deliberately NOT here. This module defines
# the workflow class, which Temporal's sandbox re-imports for determinism
# validation — a sys.path.insert(..., Path(...).resolve()...) at this
# module's top level trips the sandbox's restriction on
# pathlib.Path.resolve() (confirmed by running it, not a guess).
from base_workflow import (
    BaseAgentWorkflow,
    _HAS_TEMPORAL,
    workflow,
)  # type: ignore

HITL_SIGNAL_TIMEOUT = timedelta(hours=24)


@dataclass
class OilPriceWorkflowInput:
    tenant_id: str
    workflow_run_id: str
    price_series: list


@workflow.defn
class OilPricePredictionWorkflow(BaseAgentWorkflow):
    """
    Scheduled via `.agenticframework/schedules.yaml` (cron: "0 6 * * *", §25).
    Task queue: f"agent-tasks-{tenant_id}" (shared pool, partitioned by tenant.id).
    """

    def __init__(self) -> None:
        super().__init__()

    @workflow.run
    async def run(self, input: OilPriceWorkflowInput) -> dict:
        if not _HAS_TEMPORAL:
            raise RuntimeError(
                "temporalio is not installed. Run: pip install temporalio. "
                "See SPECS.md §25 for the production runtime spec."
            )

        ingestion = await workflow.execute_activity(
            "fetch_oil_price_activity",
            {"tenant_id": input.tenant_id, "price_series": input.price_series},
            start_to_close_timeout=timedelta(minutes=5),
        )

        prediction = await workflow.execute_activity(
            "run_prediction_activity",
            {**ingestion, "workflow_run_id": input.workflow_run_id},
            start_to_close_timeout=timedelta(minutes=10),
        )

        if not prediction.get("needs_hitl"):
            return await self._decide(input.tenant_id, prediction)

        # Approve/reject gate — inherited signal (self._hitl_approved), see
        # this file's module docstring for why run_with_hitl_gate itself
        # isn't used here.
        try:
            await workflow.wait_condition(
                lambda: self._hitl_approved is not None,
                timeout=HITL_SIGNAL_TIMEOUT,
            )
        except TimeoutError:
            return await workflow.execute_activity(
                "dead_letter_activity",
                {**prediction, "error": "hitl_timeout"},
                start_to_close_timeout=timedelta(minutes=5),
            )

        if not self._hitl_approved:
            return {"status": "rejected", **prediction}

        return await self._decide(input.tenant_id, prediction)

    async def _decide(self, tenant_id: str, prediction: dict) -> dict:
        """Edit-and-resume gate — decide_action_activity validates its
        payload shape (see activities.py) and can be corrected in place
        via the Ops Portal's DLQ view if a downstream order-system schema
        rejects it, rather than dead-lettering the whole prediction."""
        result = await self.run_with_recoverable_step(
            "decide_action_activity",
            prediction,
            tenant_id=tenant_id,
            gate_id="decide-action-gate",
            timeout=HITL_SIGNAL_TIMEOUT,
        )
        return (
            result
            if isinstance(result, dict)
            else {"status": "dead_letter", **prediction}
        )
