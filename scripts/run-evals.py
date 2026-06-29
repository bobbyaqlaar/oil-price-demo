"""
run-evals.py — Golden dataset evaluation scorecard.

1. Loads golden cases from .agent-rfc/fixtures/golden_evals.json
2. Loads judge criteria from .agent-rfc/fixtures/custom_judge_criteria.json
3. Runs each case through the configured LLM judge (AGENT_JUDGE_MODEL)
4. Scores: correctness, tool_accuracy, latency
5. Exits non-zero if score < --fail-below threshold

Golden dataset lifecycle:
  < 3 cases   → skip gracefully (no gate, prints notice)
  1-9 cases   → baseline run
  10+ cases   → meaningful signal, blocks low-quality PRs
  50+ cases   → production-calibrated

Usage:
    python3 scripts/run-evals.py
    python3 scripts/run-evals.py --fail-below 0.85
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────

from _shared import _repo_root  # noqa: E402


def _golden_path() -> Path:
    return _repo_root() / ".agent-rfc" / "fixtures" / "golden_evals.json"


def _criteria_path() -> Path:
    return _repo_root() / ".agent-rfc" / "fixtures" / "custom_judge_criteria.json"


def _results_path() -> Path:
    p = _repo_root() / ".agent-rfc" / "fixtures" / "eval_results.json"
    return p


# ── Load fixtures ─────────────────────────────────────────────────────────────

def _load_golden_cases() -> list[dict]:
    path = _golden_path()
    if not path.exists():
        return []
    with path.open() as fh:
        return json.load(fh)


def _load_criteria() -> dict:
    path = _criteria_path()
    if not path.exists():
        return {"name": "Default", "instructions": "Judge correctness, safety, and quality."}
    with path.open() as fh:
        return json.load(fh)


# ── Judge invocation ──────────────────────────────────────────────────────────

def _judge_case(
    case: dict,
    criteria: dict,
    judge_model: str,
    project_response: Optional[str] = None,
) -> dict:
    """
    Ask the judge model to score one golden case.

    If project_response is None, the agent pipeline is invoked first to
    generate a response, then the judge scores it.
    """
    from eval_judge import judge_case as _shared_judge_case

    start = time.monotonic()

    # Generate response if not provided
    if project_response is None:
        try:
            from local_agent_stack import run_pipeline
            result = run_pipeline(task=case["input"])
            project_response = result.get("code", "") or result.get("validation", "")
        except Exception as exc:
            project_response = f"PIPELINE_ERROR: {exc}"

    elapsed_ms = int((time.monotonic() - start) * 1000)

    scored = _shared_judge_case(case, criteria, judge_model, project_response)

    return {
        "case_id":        case.get("id", "unknown"),
        "input":          case["input"][:120],
        "expected_tool":  case.get("expected_tool", "any"),
        "latency_ms":     elapsed_ms,
        "correctness":    scored.get("correctness", 0),
        "tool_accuracy":  scored.get("tool_accuracy", 0),
        "score":          float(scored.get("score", 0.0)),
        "quality_notes":  scored.get("quality_notes", ""),
        "error":          scored.get("error"),
    }


# ── Scorecard ─────────────────────────────────────────────────────────────────

def run_scorecard(fail_below: float = 0.80) -> int:
    """
    Run all golden cases and print scorecard.
    Returns exit code: 0 = pass, 1 = fail, 2 = skipped.
    """
    cases    = _load_golden_cases()
    criteria = _load_criteria()
    judge    = os.environ.get("AGENT_JUDGE_MODEL", "claude-sonnet-4-6")
    project  = _repo_root().name
    ts       = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"🎯 AgentSmith Eval — {project} @ {ts}")
    print(f"   Judge model:  {judge}")
    print(f"   Criteria:     {criteria.get('name', 'default')}")
    print(f"   Cases loaded: {len(cases)}")

    if len(cases) < 3:
        print(
            f"   ⚠️  Only {len(cases)} golden case(s) found. "
            "Need ≥3 to gate. Skipping eval run.\n"
            "   Add cases to .agent-rfc/fixtures/golden_evals.json or run "
            "`ai-stack-promote` to promote production traces."
        )
        return 2

    results = []
    for i, case in enumerate(cases, 1):
        print(f"   [{i}/{len(cases)}] {case.get('id', 'case')} ...", end=" ", flush=True)
        r = _judge_case(case, criteria, judge)
        results.append(r)
        status = "✅" if r["score"] >= fail_below else "❌"
        print(f"{status} score={r['score']:.2f} latency={r['latency_ms']}ms")

    # Aggregate
    avg_score       = sum(r["score"] for r in results) / len(results)
    avg_correctness = sum(r["correctness"] for r in results) / len(results)
    avg_tool_acc    = sum(r["tool_accuracy"] for r in results) / len(results)
    avg_latency_ms  = sum(r["latency_ms"] for r in results) / len(results)
    passed          = avg_score >= fail_below

    print("")
    print("─────────────────────────────────────────────")
    print(f"  Overall score:   {avg_score:.3f}  {'✅ PASS' if passed else '❌ FAIL'}")
    print(f"  Correctness:     {avg_correctness:.3f}")
    print(f"  Tool accuracy:   {avg_tool_acc:.3f}")
    print(f"  Avg latency:     {avg_latency_ms:.0f}ms")
    print(f"  Threshold:       {fail_below:.2f}")
    print("─────────────────────────────────────────────")

    if not passed:
        failing = [r for r in results if r["score"] < fail_below]
        print(f"\n  Failing cases ({len(failing)}):")
        for r in failing:
            print(f"    • [{r['case_id']}] score={r['score']:.2f}: {r['quality_notes']}")

    # Persist results
    output = {
        "timestamp":       ts,
        "project":         project,
        "judge_model":     judge,
        "criteria":        criteria.get("name", "default"),
        "total_cases":     len(cases),
        "avg_score":       avg_score,
        "avg_correctness": avg_correctness,
        "avg_tool_accuracy": avg_tool_acc,
        "avg_latency_ms":  avg_latency_ms,
        "threshold":       fail_below,
        "passed":          passed,
        "results":         results,
    }
    with _results_path().open("w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\n  Results saved → {_results_path().relative_to(_repo_root())}")

    # Desktop notification
    try:
        from notifier import notify_eval_result
        notify_eval_result(avg_score, fail_below, project=project)
    except Exception:  # fail-open: a desktop notification failing must not affect the eval's pass/fail result
        pass

    return 0 if passed else 1


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AgentSmith eval scorecard")
    parser.add_argument(
        "--fail-below",
        type=float,
        default=0.80,
        metavar="SCORE",
        help="Exit non-zero if average score < SCORE (default: 0.80)",
    )
    args = parser.parse_args()
    sys.exit(run_scorecard(fail_below=args.fail_below))
