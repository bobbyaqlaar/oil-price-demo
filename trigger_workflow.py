"""
examples/oil-price-agent/trigger_workflow.py — Submit a workflow run and wait
for its result. Run from the tenant project root (oil-price-demo/).

Usage:
    python3 trigger_workflow.py [--price-series 70 71 69 72 95]

The default price series includes 95 as a deliberate outlier — it is >3
standard deviations from the rest, which trips the HITL gate. The workflow
will pause and wait for an approval signal (see resolve_hitl.py).
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# ── Runtime path setup ────────────────────────────────────────────────────────
# Prefer AGENTSMITH_DIR (set in ~/.zshrc); fall back to relative path from
# inside the framework tree (examples/oil-price-agent/ → ../../runtime).
_agentsmith_dir = os.environ.get("AGENTSMITH_DIR")
if _agentsmith_dir:
    _runtime_root = Path(_agentsmith_dir) / "runtime"
else:
    _runtime_root = Path(__file__).resolve().parent.parent.parent / "runtime"

sys.path.insert(0, str(Path(__file__).resolve().parent))           # tenant root
sys.path.insert(0, str(Path(__file__).resolve().parent / "workflows"))  # bare workflow/activity imports
sys.path.insert(0, str(_runtime_root))                             # framework runtime
sys.path.insert(0, str(_runtime_root / "workflows"))               # base_workflow etc.
# ─────────────────────────────────────────────────────────────────────────────

try:
    from temporalio.client import Client
    from temporalio.service import RPCError
except ImportError:
    print("ERROR: temporalio not installed. Run: pip install temporalio", file=sys.stderr)
    sys.exit(1)

# Define the input dataclass locally so this script doesn't need to import
# oil_price_workflow.py (which Temporal's sandbox re-imports under strict rules).
@dataclass
class OilPriceWorkflowInput:
    tenant_id: str
    workflow_run_id: str
    price_series: list


TENANT_ID       = os.environ.get("TENANT_ID", "oil-price-demo")
TEMPORAL_ADDRESS = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
WORKFLOW_ID     = "oil-price-demo-run-1"
TASK_QUEUE      = f"agent-tasks-{TENANT_ID}"
DEFAULT_SERIES  = [70, 71, 69, 72, 95]   # 95 is the deliberate HITL-triggering outlier


async def main() -> None:
    price_series = DEFAULT_SERIES
    if "--price-series" in sys.argv:
        idx = sys.argv.index("--price-series")
        price_series = [float(x) for x in sys.argv[idx + 1:]]

    use_tls = os.environ.get("TEMPORAL_TLS", "false").lower() == "true"
    print(f"Connecting to Temporal at {TEMPORAL_ADDRESS} (tls={use_tls}) …")
    client = await Client.connect(TEMPORAL_ADDRESS, tls=use_tls)

    print(f"Starting workflow {WORKFLOW_ID} on queue {TASK_QUEUE} …")
    print(f"Price series: {price_series}")
    print("(If 95 is in the series, the HITL gate will trip — run resolve_hitl.py to approve.)\n")

    try:
        handle = await client.start_workflow(
            "OilPricePredictionWorkflow",
            OilPriceWorkflowInput(
                tenant_id=TENANT_ID,
                workflow_run_id=WORKFLOW_ID,
                price_series=price_series,
            ),
            id=WORKFLOW_ID,
            task_queue=TASK_QUEUE,
        )
    except RPCError as exc:
        if "already running" in str(exc).lower():
            print(
                f"\nWorkflow {WORKFLOW_ID!r} is already running in Temporal.\n"
                "Terminate it first, then retry:\n\n"
                f"    temporal workflow terminate --workflow-id {WORKFLOW_ID}\n",
                file=sys.stderr,
            )
            sys.exit(1)
        raise

    print(f"Workflow started. Waiting for result (may pause at HITL gate) …")
    result = await handle.result()
    print(f"\nResult: {result}")


if __name__ == "__main__":
    asyncio.run(main())
