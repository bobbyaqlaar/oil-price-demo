"""
notifier.py — Cross-platform desktop notification dispatcher.

Primary:  plyer.notification (macOS / Linux / Windows)
Enhanced: osascript on macOS for richer alerts with action buttons.
Async:    Webhook delivery in a background thread (Slack / Teams / custom).

All notifications are non-blocking — the agent is never stalled waiting
for a desktop alert to be acknowledged.
"""

from __future__ import annotations

import os
import subprocess
import threading
from typing import Literal, Optional

# ── Config ────────────────────────────────────────────────────────────────────

APP_NAME = "AgentSmith"
ICON_PATH = os.environ.get("AGENT_NOTIFY_ICON", "")
WEBHOOK_URL = os.environ.get("AGENT_NOTIFY_WEBHOOK", "")  # Slack / Teams / custom
NOTIFY_SOUND = os.environ.get("AGENT_NOTIFY_SOUND", "Ping")  # macOS sound name

Urgency = Literal["low", "normal", "critical"]


# ── Primary: plyer ─────────────────────────────────────────────────────────────


def _notify_plyer(title: str, message: str, timeout: int = 8) -> bool:
    try:
        from plyer import notification as plyer_notif

        kwargs: dict = {
            "title": title,
            "message": message,
            "app_name": APP_NAME,
            "timeout": timeout,
        }
        if ICON_PATH and os.path.exists(ICON_PATH):
            kwargs["app_icon"] = ICON_PATH
        plyer_notif.notify(**kwargs)
        return True
    except Exception:
        return False


# ── Enhancement: osascript on macOS ───────────────────────────────────────────


# The notification text is DATA, read out of `argv` — never spliced into the
# script source.
#
# This used to f-string `message` and `title` straight into
# `display notification "{message}" with title "..."`. AppleScript has no
# escaping there, so a `"` in the text closes the literal and everything after
# it is executed as AppleScript — including `do shell script`, which runs an
# arbitrary command as whoever the agent is running as. Confirmed by running
# it, not inferred: a crafted message wrote a file to /tmp.
#
# The reachable source matters more than the shape. `notify_hitl_required` is
# called by scripts/multi_agent_system.py's `hitl_node` with
# `detail="\n".join(state["issues"])` — the Validator AGENT's own model output.
# So a model that echoes an injected instruction, or merely quotes the code it
# is reviewing, reaches a shell on the operator's machine. Model output is
# untrusted input; this is the receiving side of that boundary, and it is the
# side that has to hold (review-levers 2.7).
_OSASCRIPT_NOTIFY = """on run argv
  display notification (item 1 of argv) with title (item 2 of argv) \
    subtitle (item 3 of argv) sound name (item 4 of argv)
end run"""


def _notify_osascript(title: str, message: str) -> bool:
    """Display a macOS system notification with sound via osascript.

    Returns whether osascript actually accepted it. It used to return True
    unconditionally: `subprocess.run` without `check=` does not raise on a
    non-zero exit, so a script osascript rejected reported as delivered. This
    is the FALLBACK path — plyer has already failed by the time it runs — so a
    false "delivered" meant the last chance to reach a human was spent and
    nothing said otherwise.
    """
    try:
        import platform

        if platform.system() != "Darwin":
            return False
        proc = subprocess.run(
            [
                "osascript",
                "-e",
                _OSASCRIPT_NOTIFY,
                str(message),
                APP_NAME,
                str(title),
                NOTIFY_SOUND,
            ],
            capture_output=True,
            timeout=5,
        )
        return proc.returncode == 0
    except Exception:
        return False


# ── Webhook: Slack / Teams / custom ───────────────────────────────────────────


def _send_webhook(title: str, message: str) -> None:
    """POST to AGENT_NOTIFY_WEBHOOK in a daemon thread (fire-and-forget)."""
    if not WEBHOOK_URL:
        return

    def _post() -> None:
        try:
            import json
            import urllib.request

            payload = json.dumps(
                {"text": f"*{APP_NAME}* — *{title}*\n{message}"}
            ).encode()
            req = urllib.request.Request(
                WEBHOOK_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:  # fail-open: fire-and-forget webhook on a daemon thread; nothing waits on or checks this
            pass

    threading.Thread(target=_post, daemon=True, name="notifier-webhook").start()


# ── Public API ────────────────────────────────────────────────────────────────


def send_notification(
    title: str,
    message: str,
    urgency: Urgency = "normal",
    timeout: int = 8,
    webhook: bool = True,
) -> None:
    """
    Send a desktop notification (non-blocking).

    Tries plyer first; falls back to osascript on macOS if plyer is not
    available or fails.  Simultaneously dispatches a webhook if configured.

    Args:
        title:   Short heading shown in the notification banner.
        message: Body text.
        urgency: "low" | "normal" | "critical" — critical uses a longer timeout.
        timeout: How long the banner stays visible (seconds).
        webhook: Whether to also fire the webhook URL (default True).
    """
    if urgency == "critical":
        timeout = max(timeout, 15)

    def _dispatch() -> None:
        delivered = _notify_plyer(title, message, timeout=timeout)
        if not delivered:
            _notify_osascript(title, message)
        if webhook:
            _send_webhook(title, message)

    # Always non-blocking
    threading.Thread(target=_dispatch, daemon=True, name="notifier-dispatch").start()


def notify_hitl_required(
    agent: str,
    event: str,
    detail: str,
    project: Optional[str] = None,
) -> None:
    """Convenience wrapper for MAJOR/CRITICAL HITL escalation alerts."""
    project_tag = f" [{project}]" if project else ""
    send_notification(
        title=f"🔴 HITL Required{project_tag}",
        message=f"{agent}: {event}\n{detail}",
        urgency="critical",
    )


def notify_circuit_breaker(tier: str, detail: str) -> None:
    """Convenience wrapper for circuit breaker alerts."""
    send_notification(
        title=f"🚨 Circuit Breaker ({tier})",
        message=detail,
        urgency="critical",
    )


def notify_eval_result(
    score: float, threshold: float, project: Optional[str] = None
) -> None:
    """Convenience wrapper for post-eval summary."""
    emoji = "✅" if score >= threshold else "❌"
    project_tag = f" [{project}]" if project else ""
    send_notification(
        title=f"{emoji} Eval Result{project_tag}",
        message=f"Score: {score:.2f} (threshold: {threshold:.2f})",
        urgency="normal" if score >= threshold else "critical",
    )


# ── CLI smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    msg = " ".join(sys.argv[1:]) or "AgentSmith notification test"
    send_notification("Test Notification", msg, urgency="normal")
    import time

    time.sleep(1)  # let the daemon thread fire
    print("✅ Notification dispatched")
