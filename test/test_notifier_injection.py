"""
test/test_notifier_injection.py — the notification text must reach
osascript as DATA, never as script source.

`_notify_osascript` built its AppleScript by f-string:

    f'display notification "{message}" with title "{APP_NAME}" ...'

AppleScript does no escaping there, so a `"` in `message` closes the string
literal and everything after it is executed. `do shell script` is a normal
AppleScript command, so the payload gets a shell — as whoever the agent runs
as. That was confirmed by running it, not inferred: a crafted message wrote a
file under /tmp.

The reachable source is what makes it a finding rather than a shape.
`notify_hitl_required` is called from scripts/multi_agent_system.py's
`hitl_node` with `detail="\n".join(state["issues"])` — the Validator AGENT's
own model output. A model that echoes an injected instruction, or merely
quotes the code under review, reaches that sink.

These tests do not need a Mac: they patch `platform.system` and capture the
argv, which is where the property lives.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

import pytest

# conftest.py already puts scripts/ on sys.path

import notifier  # noqa: E402


# A message shaped like the exploit: it closes the literal, runs a command, and
# reopens the literal so the surrounding script still parses.
_PAYLOAD = '" & (do shell script "touch /tmp/agentsmith-notifier-pwned") & "'


@pytest.fixture()
def captured(monkeypatch):
    """Pretend to be macOS and record the argv instead of running osascript."""
    calls: list[list[str]] = []

    class _Proc:
        returncode = 0

    def _fake_run(argv, **_kwargs):
        calls.append(list(argv))
        return _Proc()

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(subprocess, "run", _fake_run)
    return calls


def test_message_is_never_spliced_into_the_script_source(captured):
    notifier._notify_osascript("Validator", _PAYLOAD)

    argv = captured[0]
    script = argv[argv.index("-e") + 1]
    assert _PAYLOAD not in script, (
        "the notification text was interpolated into the AppleScript source; "
        f"osascript would execute it:\n{script}"
    )
    assert "do shell script" not in script
    # It must still be delivered — as an argument, which `on run argv` reads.
    assert _PAYLOAD in argv[argv.index("-e") + 2 :]


def test_title_is_never_spliced_into_the_script_source(captured):
    """Same boundary, the other parameter. `title` is caller-built too —
    notify_hitl_required interpolates `project` into it."""
    notifier._notify_osascript(_PAYLOAD, "body")

    argv = captured[0]
    script = argv[argv.index("-e") + 1]
    assert _PAYLOAD not in script
    assert _PAYLOAD in argv[argv.index("-e") + 2 :]


def test_the_hitl_wrapper_carries_model_output_through_unspliced(monkeypatch, captured):
    """End to end from the API `multi_agent_system.hitl_node` actually calls.

    Its `detail` is the Validator model's `issues`, joined. Dispatch is
    threaded and fire-and-forget, so the thread is run inline here — the
    property under test is the argv, not the concurrency.
    """
    monkeypatch.setattr(notifier, "_notify_plyer", lambda *a, **k: False)
    monkeypatch.setattr(notifier, "_send_webhook", lambda *a, **k: None)

    class _InlineThread:
        def __init__(self, target=None, **_kwargs):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(notifier.threading, "Thread", _InlineThread)

    notifier.notify_hitl_required(
        agent="Validator",
        event="max_revisions_exceeded",
        detail=_PAYLOAD,
        project="demo",
    )

    assert captured, "osascript was never reached"
    argv = captured[0]
    assert _PAYLOAD not in argv[argv.index("-e") + 1]


def test_a_rejected_script_is_not_reported_as_delivered(monkeypatch):
    """`subprocess.run` without `check=` does not raise on a non-zero exit, so
    this returned True however osascript answered.

    It is the FALLBACK — plyer has already failed by the time it runs — so a
    false "delivered" spends the last chance to reach a human and says nothing.
    """

    class _Failed:
        returncode = 1

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Failed())

    assert notifier._notify_osascript("t", "m") is False


def test_a_non_mac_host_declines(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    assert notifier._notify_osascript("t", "m") is False
