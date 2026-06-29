"""
examples/oil-price-agent/resolve_hitl.py — Send the hitl_approved signal to a
waiting workflow. Run from a second terminal while trigger_workflow.py is
still waiting.

Usage:
    python3 resolve_hitl.py [--approve | --reject]

Default: --approve
"""

from __future__ import annotations

import asyncio
import os
import sys

try:
    from temporalio.client import Client
except ImportError:
    print("ERROR: temporalio not installed. Run: pip install temporalio", file=sys.stderr)
    sys.exit(1)

TEMPORAL_ADDRESS = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
WORKFLOW_ID      = "oil-price-demo-run-1"


async def main() -> None:
    approve = "--reject" not in sys.argv

    print(f"Connecting to Temporal at {TEMPORAL_ADDRESS} …")
    client = await Client.connect(TEMPORAL_ADDRESS)

    handle = client.get_workflow_handle(WORKFLOW_ID)
    await handle.signal("hitl_approved", approve)

    decision = "APPROVED" if approve else "REJECTED"
    print(f"Signal sent: hitl_approved={approve} ({decision})")
    print("trigger_workflow.py should now complete.")


if __name__ == "__main__":
    asyncio.run(main())
