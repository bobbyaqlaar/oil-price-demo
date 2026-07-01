"""
runtime/replay_webhook_server.py — reference HTTP receiver for the Ops
Portal's "Replay with edits" DLQ action.

Why this exists: the Ops Portal (Next.js) has no Temporal client and
never will — runtime/dead_letter.py's replay_handler is deliberately
engine-agnostic, and the portal is meant to stay backend/orchestrator-
agnostic too (a tenant could run Celery, not Temporal). So when a human
edits a failing payload in the portal's DLQ view and clicks Replay, the
portal can't signal a live workflow directly — it POSTs the edit to
THIS tenant-run receiver instead, which DOES have a Temporal client
(it runs alongside the worker) and calls DeadLetterQueue.replay() for real.

This also keeps HITL routing tenant-specific by construction: each tenant
configures their OWN replay_webhook_url (synced from
.agenticframework/tenant.yaml, see OPERATIONS.md), so a human-in-the-loop
fix for tenant A's DLQ entry is delivered to tenant A's own
receiver/team — never a shared, cross-tenant endpoint.

This is a PATTERN, not a hardened production server — it's deliberately
built on the stdlib (http.server) so it has no new dependency beyond
what's already in requirements.txt, the same "reference, not
prescription" posture as base_workflow.py and worker.py. Tenants are
expected to copy/adapt this into their actual web framework (FastAPI,
Flask, etc.) — see worker.py's TENANT_WORKER_MODULE for the equivalent
pattern on the worker side.

Required env vars:
  DATABASE_URL              — same Postgres the worker's DeadLetterQueue uses
  REPLAY_WEBHOOK_SECRET      — shared secret; must match what's configured
                               in the portal for this tenant (sent back via
                               .agenticframework/tenant.yaml -> sync, see
                               OPERATIONS.md "Wire your platform")
  TEMPORAL_ADDRESS           — e.g. "localhost:7233" (default if unset)

Run: python3 replay_webhook_server.py [port, default 8090]
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger(__name__)


def _verify_signature(secret: str, body: bytes, signature_header: str) -> bool:
    if not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    provided = signature_header[len("sha256=") :]
    return hmac.compare_digest(expected, provided)


class ReplayWebhookHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 (http.server's required method name)
        if self.path != "/replay":
            self.send_response(404)
            self.end_headers()
            return

        secret = os.environ.get("REPLAY_WEBHOOK_SECRET", "")
        if not secret:
            logger.error("REPLAY_WEBHOOK_SECRET not set — refusing all requests")
            self.send_response(503)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        signature = self.headers.get("X-Replay-Signature", "")
        if not _verify_signature(secret, body, signature):
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'{"error":"invalid signature"}')
            return

        try:
            data = json.loads(body)
            task_id = data["taskId"]
            payload = data["payload"]
        except (json.JSONDecodeError, KeyError) as exc:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"bad request: {exc}"}).encode())
            return

        try:
            self._replay(task_id, payload)
        except Exception as exc:
            logger.exception("Replay failed for task_id=%s", task_id)
            self.send_response(500)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(exc)}).encode())
            return

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def _replay(self, task_id: str, payload: dict) -> None:
        sys.path.insert(
            0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/runtime"
        )
        from dead_letter import DeadLetterQueue  # type: ignore
        from temporal_replay import make_temporal_replay_handler  # type: ignore
        from temporalio.client import Client  # type: ignore

        async def _connect():
            # Bounded, not indefinite: without this, a Temporal server
            # that's down/unreachable hangs this request for the OS TCP
            # connect timeout (often 2+ minutes) before failing — long
            # enough to look like the portal itself is stuck, not "the
            # tenant's Temporal is unreachable."
            return await asyncio.wait_for(
                Client.connect(os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")),
                timeout=10.0,
            )

        client = asyncio.run(_connect())
        dlq = DeadLetterQueue(replay_handler=make_temporal_replay_handler(client))
        dlq.replay(task_id, override_payload=payload)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        logger.info("%s - %s", self.address_string(), format % args)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
    server = ThreadingHTTPServer(("0.0.0.0", port), ReplayWebhookHandler)
    logger.info("replay_webhook_server listening on :%d (POST /replay)", port)
    server.serve_forever()


if __name__ == "__main__":
    main()
