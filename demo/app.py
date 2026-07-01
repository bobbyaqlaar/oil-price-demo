"""
demo/app.py — Streamlit GUI for the oil-price-demo pipeline.

Connects to the live Temporal server, starts OilPricePredictionWorkflow,
polls for status, and lets the operator approve or reject the HITL gate —
the same operations trigger_workflow.py and resolve_hitl.py do from the CLI.

Requires the Temporal worker (worker.py) to be running.

Environment variables (same as worker.py / trigger_workflow.py):
    TEMPORAL_ADDRESS   Temporal frontend host:port  (default: localhost:7233)
    TEMPORAL_TLS       "true" to enable TLS         (default: false)
    TENANT_ID          tenant identifier             (default: oil-price-demo)
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from typing import Optional

import streamlit as st

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Oil Price Agent — AgentSmith",
    page_icon="\U0001f6e2\ufe0f",
    layout="wide",
)

TEMPORAL_ADDRESS = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
TEMPORAL_TLS = os.environ.get("TEMPORAL_TLS", "false").lower() == "true"
TENANT_ID = os.environ.get("TENANT_ID", "oil-price-demo")
TASK_QUEUE = f"agent-tasks-{TENANT_ID}"


@dataclass
class OilPriceWorkflowInput:
    tenant_id: str
    workflow_run_id: str
    price_series: list


# ── Temporal helpers ──────────────────────────────────────────────────────────


def _run(coro):
    """Run a coroutine from synchronous Streamlit context.

    Uses asyncio.run() so it works in Streamlit's non-main threads
    (Python 3.10+ deprecated get_event_loop() in non-main threads).
    """
    return asyncio.run(coro)


async def _connect():
    """Return a cached Temporal client, creating one per session if needed."""
    from temporalio.client import Client

    if (
        "temporal_client" not in st.session_state
        or st.session_state.temporal_client is None
    ):
        st.session_state.temporal_client = await Client.connect(
            TEMPORAL_ADDRESS, tls=TEMPORAL_TLS
        )
    return st.session_state.temporal_client


async def _start_workflow(series: list[float], workflow_id: str) -> str:
    client = await _connect()
    await client.start_workflow(
        "OilPricePredictionWorkflow",
        OilPriceWorkflowInput(
            tenant_id=TENANT_ID,
            workflow_run_id=workflow_id,
            price_series=series,
        ),
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )
    return workflow_id


async def _get_status(workflow_id: str) -> dict:
    """Return a status dict without blocking on result()."""
    try:
        client = await _connect()
        handle = client.get_workflow_handle(workflow_id)
        desc = await handle.describe()
        status = str(desc.status).split(".")[
            -1
        ]  # e.g. "RUNNING", "COMPLETED", "FAILED"
        return {"status": status, "id": workflow_id}
    except (
        Exception
    ) as exc:  # fail-open: network/RPC errors become ERROR status, not a crash
        return {"status": "ERROR", "error": str(exc)}


async def _send_signal(workflow_id: str, approve: bool) -> None:
    client = await _connect()
    handle = client.get_workflow_handle(workflow_id)
    await handle.signal("hitl_approved", approve)


async def _get_result(workflow_id: str) -> Optional[dict]:
    """Non-blocking: return result if workflow is complete, else None."""
    from temporalio.client import WorkflowExecutionStatus

    client = await _connect()
    handle = client.get_workflow_handle(workflow_id)
    desc = await handle.describe()
    if desc.status == WorkflowExecutionStatus.COMPLETED:
        return await handle.result()
    return None


async def _terminate(workflow_id: str) -> None:
    client = await _connect()
    handle = client.get_workflow_handle(workflow_id)
    await handle.terminate(reason="Cancelled via demo UI")


# ── session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("workflow_id", None),
    ("last_status", None),
    ("result", None),
    ("history", []),
    ("hitl_triggered", False),
    ("error", None),
    ("temporal_client", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("\U0001f6e2\ufe0f Oil Price Agent")
    st.caption(f"Tenant: `{TENANT_ID}`")
    st.caption(f"Temporal: `{TEMPORAL_ADDRESS}`")

    st.markdown("---")
    st.subheader("Price series")
    preset = st.selectbox(
        "Preset",
        [
            "Normal run (no HITL)",
            "HITL — price spike",
            "HITL — low confidence override",
            "Custom",
        ],
    )
    presets = {
        "Normal run (no HITL)": "70.0, 71.0, 69.5, 70.2, 70.8, 71.0, 70.5",
        "HITL — price spike": "70.0, 70.1, 69.9, 70.0, 70.1, 70.0, 70.2, 69.8, 70.1, 70.0, 110.0",
        "HITL — low confidence override": "70.0, 71.0, 69.5, 70.2, 70.8",
        "Custom": "70.0, 71.0, 69.5, 70.2, 70.8",
    }
    series_input = st.text_area(
        "Prices (comma-separated, USD/bbl)", value=presets[preset], height=80
    )

    st.markdown("---")
    run_btn = st.button(
        "\u25b6\ufe0f  Start workflow",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.workflow_id is not None,
    )
    cancel_btn = st.button(
        "\u23f9\ufe0f  Cancel",
        use_container_width=True,
        disabled=st.session_state.workflow_id is None,
    )
    refresh_btn = st.button("\U0001f504  Refresh status", use_container_width=True)

# ── main ───────────────────────────────────────────────────────────────────────
st.title("Oil Price Prediction Pipeline")
st.caption(
    "Ingestion \u2192 Prediction \u2192 HITL gate \u2192 Decision · backed by live Temporal worker"
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 1 \xb7 IngestionAgent")
    st.info(
        "Validates the price series.\n\n_Calls `fetch_oil_price_activity` on the worker._"
    )
with col2:
    st.markdown("### 2 \xb7 PredictionAgent")
    st.info(
        "Anomaly detection (>3\u03c3) + LLM forecast.\n\n_Calls `run_prediction_activity` via LLMGateway._"
    )
with col3:
    st.markdown("### 3 \xb7 DecisionAgent")
    st.info(
        "Places order or routes to DLQ.\n\n_Calls `decide_action_activity`; HITL gate pauses here._"
    )

st.markdown("---")

# ── start ──────────────────────────────────────────────────────────────────────
if run_btn:
    try:
        series = [float(x.strip()) for x in series_input.split(",") if x.strip()]
    except ValueError:
        st.error("Invalid series — enter comma-separated numbers.")
        st.stop()
    if len(series) < 3:
        st.error("Need at least 3 price points.")
        st.stop()

    wf_id = f"oil-price-demo-{uuid.uuid4().hex[:8]}"
    try:
        _run(_start_workflow(series, wf_id))
        st.session_state.workflow_id = wf_id
        st.session_state.last_status = "RUNNING"
        st.session_state.result = None
        st.session_state.hitl_triggered = False
        st.session_state.error = None
        st.rerun()
    except Exception as exc:
        st.error(f"Failed to start workflow: {exc}")

# ── cancel ─────────────────────────────────────────────────────────────────────
if cancel_btn and st.session_state.workflow_id:
    try:
        _run(_terminate(st.session_state.workflow_id))
    except Exception:  # fail-open: cancel is best-effort; clear local state regardless
        pass
    st.session_state.workflow_id = None
    st.session_state.last_status = None
    st.rerun()

# ── status polling ─────────────────────────────────────────────────────────────
if st.session_state.workflow_id:
    wf_id = st.session_state.workflow_id

    try:
        status_dict = _run(_get_status(wf_id))
        status = status_dict.get("status", "UNKNOWN")
        st.session_state.last_status = status
    except Exception as exc:
        status = "ERROR"
        st.session_state.error = str(exc)

    st.markdown(f"**Workflow:** `{wf_id}`")

    status_map = {
        "RUNNING": ("\U0001f7e1 Running", "warning"),
        "COMPLETED": ("\u2705 Completed", "success"),
        "FAILED": ("\u274c Failed", "error"),
        "TERMINATED": ("\u23f9\ufe0f Terminated", "warning"),
        "ERROR": ("\u274c Error connecting to Temporal", "error"),
    }
    label, kind = status_map.get(status, (f"\u2753 {status}", "info"))
    getattr(st, kind)(label)

    if status == "RUNNING":
        # Heuristic: if workflow has been running > ~5 s, likely at HITL gate
        st.info(
            "Workflow is running on the Temporal worker. "
            "If the price series triggered an anomaly or low confidence, "
            "the HITL gate below will be active."
        )
        # Show HITL panel — the operator signals regardless; the workflow
        # ignores the signal if it hasn't reached the gate yet.
        st.markdown("---")
        st.markdown("## \U0001f6d1 Human-in-the-Loop Gate")
        st.warning(
            "If the prediction agent flagged this run (anomaly >3\u03c3 or "
            "confidence <0.6), the workflow is paused here waiting for your signal."
        )
        if st.session_state.hitl_triggered:
            st.info("Signal already sent — waiting for workflow to transition.")
        else:
            ca, cb, _ = st.columns([2, 2, 4])
            with ca:
                if st.button(
                    "\u2705 Approve", type="primary", use_container_width=True
                ):
                    try:
                        _run(_send_signal(wf_id, approve=True))
                        st.session_state.hitl_triggered = True
                        st.success("Approval signal sent.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Signal failed: {exc}")
            with cb:
                if st.button("\u274c Reject", use_container_width=True):
                    try:
                        _run(_send_signal(wf_id, approve=False))
                        st.session_state.hitl_triggered = True
                        st.error(
                            "Rejection signal sent — workflow routes to Dead-Letter Queue."
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Signal failed: {exc}")

        st.markdown(
            "_Click \U0001f504 Refresh status in the sidebar to check for completion._"
        )

    elif status == "COMPLETED":
        try:
            result = _run(_get_result(wf_id))
            if result:
                st.session_state.result = result
                st.session_state.history.append(
                    {"workflow_id": wf_id, "result": result}
                )
                st.markdown("### Result")
                st.json(result)
            else:
                st.warning("Result not yet available — click 🔄 Refresh status.")
        except Exception as exc:
            st.warning(f"Workflow completed but could not fetch result: {exc}")

        if st.button("Start new run"):
            st.session_state.workflow_id = None
            st.session_state.last_status = None
            st.rerun()

    elif status in ("FAILED", "TERMINATED", "ERROR"):
        if st.session_state.error:
            st.error(st.session_state.error)
        if st.button("Start new run"):
            st.session_state.workflow_id = None
            st.session_state.last_status = None
            st.rerun()

elif not run_btn:
    st.info(
        "Configure the price series in the sidebar and click \u25b6\ufe0f Start workflow."
    )

# ── run history ────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("## Run history")
    rows = [
        {
            "Workflow ID": h["workflow_id"],
            "Status": h.get("result", {}).get("status", "—"),
            "Prediction ($)": h.get("result", {}).get("prediction", "—"),
            "Confidence": h.get("result", {}).get("confidence", "—"),
        }
        for h in reversed(st.session_state.history)
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

# ── footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "AgentSmith framework \xb7 oil-price-demo \xb7 "
    "[github.com/bobbyaqlaar/oil-price-demo](https://github.com/bobbyaqlaar/oil-price-demo)"
)
