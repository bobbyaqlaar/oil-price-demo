"""
network_watchdog.py — Network connectivity probe with automatic offline fallback.

Pings 1.1.1.1:53 (Cloudflare DNS). On failure, switches the active LLM
endpoint to the local Ollama instance and notifies the agent.

Used by cost_router.py and local_agent_stack.py to decide which model
tier to use before every LLM call.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

PROBE_HOST = "1.1.1.1"
PROBE_PORT = 53
PROBE_TIMEOUT = 2.0  # seconds

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "ollama")
CLOUD_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
CLOUD_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# How often to re-probe when in offline mode (seconds)
RECHECK_INTERVAL = 30

# ── State ─────────────────────────────────────────────────────────────────────

_lock = threading.Lock()
_online: Optional[bool] = None  # None = not yet probed
_last_check: float = 0.0


# ── Core probe ────────────────────────────────────────────────────────────────


def is_online(force: bool = False) -> bool:
    """
    Return True if the machine has internet connectivity.
    Result is cached for RECHECK_INTERVAL seconds.
    Pass force=True to bypass the cache and re-probe immediately.
    """
    global _online, _last_check
    with _lock:
        now = time.time()
        if not force and _online is not None and (now - _last_check) < RECHECK_INTERVAL:
            return _online

        try:
            sock = socket.create_connection(
                (PROBE_HOST, PROBE_PORT), timeout=PROBE_TIMEOUT
            )
            sock.close()
            was_online = _online
            _online = True
            _last_check = now
            if was_online is False:
                # Recovered from offline — notify
                _notify_recovery()
            return True
        except OSError:
            was_online = _online
            _online = False
            _last_check = now
            if was_online is not False:
                # Just went offline — notify
                _notify_offline()
            return False


def get_llm_endpoint() -> dict:
    """
    Return {"base_url": ..., "api_key": ...} for the currently active
    LLM backend — cloud when online, Ollama when offline.
    """
    if is_online():
        return {"base_url": CLOUD_BASE_URL, "api_key": CLOUD_API_KEY}
    return {"base_url": OLLAMA_BASE_URL, "api_key": OLLAMA_API_KEY}


def require_online(feature: str = "this feature") -> None:
    """
    Raise a RuntimeError if offline. Use before any operation that
    strictly requires internet access (e.g. fetching remote data).
    """
    if not is_online(force=True):
        raise RuntimeError(
            f"Network offline — {feature} requires internet connectivity. "
            f"Start local Ollama: `ollama serve`"
        )


# ── Background keepalive thread ───────────────────────────────────────────────

_watcher_thread: Optional[threading.Thread] = None


def start_background_watcher(interval: float = RECHECK_INTERVAL) -> None:
    """
    Spawn a daemon thread that continuously monitors connectivity.
    Call once at agent startup to enable proactive offline detection.
    """
    global _watcher_thread
    if _watcher_thread and _watcher_thread.is_alive():
        return

    def _run() -> None:
        while True:
            is_online(force=True)
            time.sleep(interval)

    _watcher_thread = threading.Thread(
        target=_run, daemon=True, name="network-watchdog"
    )
    _watcher_thread.start()


def stop_background_watcher() -> None:
    # Daemon thread — exits automatically with main process.
    # This is a no-op kept for API symmetry.
    pass


# ── Notifications ─────────────────────────────────────────────────────────────


def _notify_offline() -> None:
    try:
        from notifier import send_notification

        send_notification(
            "📡 Network Offline",
            "AgentSmith switched to LOCAL mode (Ollama).",
            urgency="normal",
        )
    except Exception:
        import sys

        print("[network_watchdog] OFFLINE — falling back to Ollama", file=sys.stderr)


def _notify_recovery() -> None:
    try:
        from notifier import send_notification

        send_notification(
            "✅ Network Restored",
            "AgentSmith is back online — cloud models available.",
            urgency="low",
        )
    except Exception:
        import sys

        print("[network_watchdog] ONLINE — cloud models available", file=sys.stderr)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    status = is_online(force=True)
    endpoint = get_llm_endpoint()
    print(
        json.dumps(
            {
                "online": status,
                "active_endpoint": endpoint["base_url"],
            },
            indent=2,
        )
    )
