"""
examples/oil-price-agent/worker.py — Temporal worker entrypoint for this tenant.

Tenant repos register their OWN workflows/activities here — this is what
that registration looks like, copied out of runtime/worker.py's generic
NotImplementedError stub and made concrete for one domain (§25).

Usage:
    export TENANT_ID=oil-price-demo
    export TEMPORAL_ADDRESS=localhost:7233
    python3 worker.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "workflows"))

# Locate the AgentSmith runtime — prefer AGENTSMITH_DIR env var (set in ~/.zshrc)
# so this worker can run from any directory, not just from inside the framework tree.
_agentsmith_dir = os.environ.get("AGENTSMITH_DIR")
if _agentsmith_dir:
    _runtime_root = Path(_agentsmith_dir) / "runtime"
else:
    # Fallback: assume worker.py is still inside examples/oil-price-agent/ (3 levels deep)
    _runtime_root = Path(__file__).resolve().parent.parent.parent / "runtime"

# oil_price_workflow.py subclasses BaseAgentWorkflow (runtime/workflows/) —
# added here, not inside oil_price_workflow.py itself, because that file
# defines the workflow class and gets re-imported by Temporal's sandbox
# for determinism validation; a sys.path.insert(..., Path(...).resolve()...)
# at THAT module's top level trips the sandbox's restriction on
# pathlib.Path.resolve() (confirmed by running it — not a guess).
sys.path.insert(0, str(_runtime_root))
sys.path.insert(0, str(_runtime_root / "workflows"))

from oil_price_workflow import OilPricePredictionWorkflow  # type: ignore
from activities import (  # type: ignore
    fetch_oil_price_activity,
    run_prediction_activity,
    decide_action_activity,
    dead_letter_activity,
)


async def _run_health_server() -> None:
    """Minimal GET /healthz -> 200 server, run alongside the Temporal poller.

    Cloud Run (and most container platforms) expect an HTTP listener to
    health-check; a pure Temporal worker has no HTTP surface at all. This
    does NOT report Temporal connectivity — it's a liveness probe for the
    process itself (so a hung/crashed process gets recycled), not a
    readiness probe for "is this worker actually polling its task queue."
    Port via $PORT (Cloud Run's convention), default 8080 for local runs.
    """
    from aiohttp import web

    async def healthz(_request: web.Request) -> web.Response:
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/healthz", healthz)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", "8080"))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    print(f"[worker] Health server listening on :{port}/healthz")


async def _run_worker_with_retry(tenant_id: str) -> None:
    """Connect to Temporal and run the worker, retrying the connect step on
    failure instead of raising — a transient Temporal outage must not crash
    the whole process (and take the health server down with it). The health
    server is a liveness probe for the process, not Temporal connectivity;
    this keeps that distinction real instead of accidental."""
    from temporalio.client import Client
    from temporalio.worker import Worker

    address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    # TEMPORAL_TLS=true for cloud-hosted Temporal behind a TLS-terminating
    # ingress (e.g. Cloud Run, which only exposes 443 — there is no
    # plaintext gRPC port reachable from outside the container).
    use_tls = os.environ.get("TEMPORAL_TLS", "false").lower() == "true"
    delay = 2.0
    while True:
        try:
            client = await Client.connect(address, tls=use_tls)
            break
        except Exception as exc:
            print(f"[worker] Temporal connect failed ({exc}); retrying in {delay:.0f}s", file=sys.stderr)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)

    worker = Worker(
        client,
        task_queue=f"agent-tasks-{tenant_id}",
        workflows=[OilPricePredictionWorkflow],
        activities=[
            fetch_oil_price_activity,
            run_prediction_activity,
            decide_action_activity,
            dead_letter_activity,
        ],
    )
    print(f"[worker] Listening on task queue agent-tasks-{tenant_id} @ {address}")
    await worker.run()


async def main() -> None:
    tenant_id = os.environ.get("TENANT_ID", "")
    if not tenant_id:
        print("ERROR: TENANT_ID environment variable is required.", file=sys.stderr)
        sys.exit(1)

    try:
        import temporalio  # noqa: F401
    except ImportError:
        print("ERROR: temporalio is not installed. Run: pip install temporalio", file=sys.stderr)
        sys.exit(1)

    await _run_health_server()
    await _run_worker_with_retry(tenant_id)


if __name__ == "__main__":
    asyncio.run(main())
