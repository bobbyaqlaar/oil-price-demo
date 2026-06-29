"""
examples/oil-price-agent/workflows/activities.py — Domain activities for the
oil price prediction workflow.

Pipeline: IngestionAgent -> PredictionAgent -> DecisionAgent (see
examples/oil-price-agent/agents/README.md).

All LLM calls route through runtime/llm_gateway.py — never cost_router.py
(§25, §29). This file is a reference; tenant repos vendor or pip-install the
framework's runtime/ package rather than reaching into a sibling checkout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from temporalio import activity
except ImportError:

    class _NoopActivity:
        def defn(self, fn=None, **_k):
            return fn if fn is not None else (lambda f: f)

    activity = _NoopActivity()  # type: ignore


def _ensure_runtime_on_path() -> None:
    import os

    # Prefer AGENTSMITH_DIR (set in ~/.zshrc, or baked into a container
    # image's ENV) — works regardless of where this file lives (inside the
    # framework tree, copied to a tenant repo, or vendored into a container
    # at a shallower depth than the framework tree, e.g. /app/workflows/).
    agentsmith_dir = os.environ.get("AGENTSMITH_DIR")
    here = Path(__file__).resolve()
    candidates = []
    if agentsmith_dir:
        candidates.append(Path(agentsmith_dir) / "runtime")
    # Fallback 1: walk up from this file (works when inside framework tree:
    # examples/oil-price-agent/workflows/ → parents[3] = framework root).
    # Guarded — a container layout (/app/workflows/activities.py) has fewer
    # than 4 parents and would otherwise raise IndexError before ever
    # reaching the AGENTSMITH_DIR candidate above.
    if len(here.parents) > 3:
        candidates.append(here.parents[3] / "runtime")
    # Fallback 2: vendored install location
    candidates.append(Path.home() / ".agent-framework" / "runtime")
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if str(candidate) not in sys.path:
            sys.path.insert(
                0, str(candidate)
            )  # bare imports, e.g. `from dead_letter import ...`
        if str(candidate.parent) not in sys.path:
            sys.path.insert(
                0, str(candidate.parent)
            )  # package imports, e.g. `import runtime.llm_gateway`
        return


_ensure_runtime_on_path()


# ── HITL thresholds (examples/oil-price-agent/agents/README.md) ──────────────

ANOMALY_STD_DEV_THRESHOLD = 3.0
CONFIDENCE_HITL_THRESHOLD = 0.6


@activity.defn
async def fetch_oil_price_activity(payload: dict) -> dict:
    """IngestionAgent: fetch latest price data. Stub data source for the reference example."""
    # TODO (tenant implementation): replace with a real price feed API call.
    return {
        "price_series": payload.get("price_series", []),
        "tenant_id": payload["tenant_id"],
    }


@activity.defn
async def run_prediction_activity(payload: dict) -> dict:
    """
    PredictionAgent: forecast next price point. Flags for HITL review when the
    observed price deviates more than ANOMALY_STD_DEV_THRESHOLD standard
    deviations from the trailing series, or when model confidence is low.
    """
    from llm_gateway import LLMGateway  # type: ignore
    from idempotency import make_key  # type: ignore

    series = payload.get("price_series", [])
    tenant_id = payload["tenant_id"]

    mean = sum(series) / len(series) if series else 0.0
    variance = sum((p - mean) ** 2 for p in series) / len(series) if series else 0.0
    std_dev = variance**0.5
    latest = series[-1] if series else 0.0
    is_anomaly = (
        std_dev > 0 and abs(latest - mean) > ANOMALY_STD_DEV_THRESHOLD * std_dev
    )

    # Keyed on the workflow run + activity + its actual input, not just the
    # workflow run id alone — a Temporal activity retry of THIS SPECIFIC call
    # must dedupe against itself, but a different activity in the same
    # workflow run (or the same activity called again with different input)
    # must not collide with it (FIXES_AND_CLEANUP.md P0).
    idempotency_key = make_key(
        {
            "activity": "run_prediction_activity",
            "workflow_run_id": payload.get("workflow_run_id"),
            "price_series": series,
        }
    )

    gateway = LLMGateway(tenant_id=tenant_id)
    try:
        result = await gateway.complete(
            prompt=f"Given recent oil prices {series}, predict the next price point and a "
            f'confidence score (0-1) as JSON: {{"prediction": <float>, "confidence": <float>}}.',
            model_hint="validator",  # cheap/fast tier for a bounded forecasting prompt
            workflow_id=payload.get("workflow_run_id"),
            idempotency_key=idempotency_key,
        )
    except Exception as exc:
        # Surface a short, actionable message — full traceback is in worker logs.
        raise RuntimeError(f"LLM call failed: {exc}") from None

    try:
        parsed = json.loads(result.text)
        prediction = float(parsed["prediction"])
        confidence = float(parsed["confidence"])
    except Exception:
        prediction, confidence = latest, 0.0

    needs_hitl = is_anomaly or confidence < CONFIDENCE_HITL_THRESHOLD
    return {
        "tenant_id": tenant_id,
        "prediction": prediction,
        "confidence": confidence,
        "is_anomaly": is_anomaly,
        "needs_hitl": needs_hitl,
        "cost_usd": result.cost_usd,
    }


@activity.defn
async def decide_action_activity(payload: dict) -> dict:
    """DecisionAgent: act on an approved (or non-flagged) prediction.

    Validates the shape a real downstream order-placement API would
    require before calling it — wrapped by
    oil_price_workflow.py's run_with_recoverable_step, so a payload this
    rejects parks the workflow alive for a human to correct via the Ops
    Portal's DLQ view (FIXES_AND_CLEANUP.md's HITL/DLQ redesign) instead
    of failing the run. This is the exact CRM-example shape: a malformed
    field is the kind of error a human edit fixes, not a retry.
    """
    if "prediction" not in payload or not isinstance(
        payload.get("confidence"), (int, float)
    ):
        bad_keys = [
            k
            for k in payload
            if k not in ("tenant_id", "is_anomaly", "needs_hitl", "cost_usd")
        ]
        raise ValueError(
            f"decide_action_activity payload missing required prediction/confidence fields "
            f"(got keys: {bad_keys}) — downstream order API requires {{'prediction': <float>, 'confidence': <float>}}"
        )

    # TODO (tenant implementation): place the order / fire the alert.
    return {
        "status": "success",
        "prediction": payload.get("prediction"),
        "confidence": payload.get("confidence"),
    }


@activity.defn
async def dead_letter_activity(payload: dict) -> dict:
    """Routed to when the HITL signal times out (24h) — see base_workflow.py."""
    from dead_letter import DeadLetterQueue  # type: ignore

    # Deterministic task_id (not a fresh uuid per call): if Temporal retries
    # THIS activity itself (e.g. a transient DB blip inside enqueue(), before
    # it returns successfully), the retry computes the same task_id, and
    # DeadLetterQueue.enqueue()'s ON CONFLICT DO NOTHING dedupes it into one
    # DLQ row instead of one per retry attempt.
    workflow_run_id = payload.get("workflow_run_id", "unknown")
    task_id = f"hitl-timeout-{workflow_run_id}"

    try:
        dlq = DeadLetterQueue()
        dlq.enqueue(
            payload=payload,
            error=payload.get("error", "unknown"),
            tenant_id=payload["tenant_id"],
            task_id=task_id,
        )
    except Exception as exc:
        raise RuntimeError(f"Dead-letter enqueue failed: {exc}") from None
    return {"status": "dead_letter"}
