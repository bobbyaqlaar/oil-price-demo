"""
runtime/worker.py — Production worker entrypoint.

Starts a Temporal or Celery worker partitioned by tenant.id.
All LLM calls route through llm_gateway.py — cost_router.py is NOT used here.
All spans carry tenant.id, workflow.id, workflow.run_id.

This module intentionally has no domain-specific workflows/activities of its
own — per SPECS.md §25/§28, "framework workflows are never deployed directly
as tenant production code." Tenant repos copy this file's shape and bind
their own workflows/activities, the same way
examples/oil-price-agent/worker.py does (a complete, working reference).

TENANT_WORKER_MODULE is an alternative to copying this file outright: if
set, runtime/worker.py becomes a thin dispatcher that imports that module
and calls its `start_temporal_worker(tenant_id)` / `start_celery_worker(tenant_id)`
function instead of raising — useful for a tenant repo that wants to keep
using runtime/worker.py as its actual entrypoint script (e.g. as a
container CMD) while supplying its own workflow registration as an
importable module rather than a full copy of this file. Without
TENANT_WORKER_MODULE set, the behavior is unchanged: this module cannot run
anything by itself and says so loudly rather than pretending to.

See SPECS.md §25 for the full production runtime specification.
"""

from __future__ import annotations

import importlib
import os
import sys


def main() -> None:
    """
    Worker entrypoint. Reads WORKER_BACKEND from environment:
      - 'temporal' (default): start Temporal worker
      - 'celery':             start Celery worker

    TENANT_ID must be set for production workers.
    """
    backend = os.environ.get("WORKER_BACKEND", "temporal").lower()
    tenant_id = os.environ.get("TENANT_ID", "")

    if not tenant_id:
        print(
            "[worker] ERROR: TENANT_ID environment variable is required.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[worker] Starting {backend} worker for tenant={tenant_id}")

    if backend == "temporal":
        _start_temporal_worker(tenant_id)
    elif backend == "celery":
        _start_celery_worker(tenant_id)
    else:
        print(f"[worker] ERROR: Unknown WORKER_BACKEND={backend!r}", file=sys.stderr)
        sys.exit(1)


def _load_tenant_worker_module():
    module_name = os.environ.get("TENANT_WORKER_MODULE", "")
    if not module_name:
        return None
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        print(
            f"[worker] ERROR: TENANT_WORKER_MODULE={module_name!r} could not be imported: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


def _start_temporal_worker(tenant_id: str) -> None:
    """
    Delegates to TENANT_WORKER_MODULE.start_temporal_worker(tenant_id) if
    that env var is set. Without it, this module has no workflows/
    activities to register and cannot start anything — see module
    docstring for the two ways to supply them.

    Reference pattern (what examples/oil-price-agent/worker.py implements
    concretely):
      async with Client.connect(os.environ["TEMPORAL_ADDRESS"]) as client:
          worker = Worker(
              client,
              task_queue=f"agent-tasks-{tenant_id}",
              workflows=[AgentWorkflow],
              activities=[architect_activity, developer_activity, validator_activity],
          )
          await worker.run()
    """
    module = _load_tenant_worker_module()
    if module is None:
        raise NotImplementedError(
            "No workflows/activities registered. Either copy this file's shape into your "
            "tenant repo (see examples/oil-price-agent/worker.py for a complete example) "
            "or set TENANT_WORKER_MODULE=your_module (exposing start_temporal_worker(tenant_id)). "
            "See SPECS.md §25 and runtime/workflows/."
        )
    if not hasattr(module, "start_temporal_worker"):
        raise AttributeError(
            f"TENANT_WORKER_MODULE={module.__name__!r} has no start_temporal_worker(tenant_id) function."
        )
    module.start_temporal_worker(tenant_id)


def _start_celery_worker(tenant_id: str) -> None:
    """
    Delegates to TENANT_WORKER_MODULE.start_celery_worker(tenant_id) if that
    env var is set — same rationale as _start_temporal_worker above.

    Reference pattern:
      app = Celery("agent_worker", broker=os.environ["REDIS_URL"])
      app.conf.task_routes = {
          "runtime.tasks.*": {"queue": f"agent-{tenant_id}"}
      }
      app.worker_main(argv=["worker", "--loglevel=info"])
    """
    module = _load_tenant_worker_module()
    if module is None:
        raise NotImplementedError(
            "No tasks registered. Either copy this file's shape into your tenant repo "
            "or set TENANT_WORKER_MODULE=your_module (exposing start_celery_worker(tenant_id)). "
            "See SPECS.md §25."
        )
    if not hasattr(module, "start_celery_worker"):
        raise AttributeError(
            f"TENANT_WORKER_MODULE={module.__name__!r} has no start_celery_worker(tenant_id) function."
        )
    module.start_celery_worker(tenant_id)


if __name__ == "__main__":
    main()
