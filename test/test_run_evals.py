"""
test/test_run_evals.py — Unit tests for scripts/run-evals.py.

Covers the key exit-code contracts:
  exit 2 — < 3 golden cases (skip gracefully)
  exit 0 — ALL cases fail due to pipeline/API errors (Groq 429 etc.) — skip gracefully
  exit 0 — avg score >= threshold (pass)
  exit 1 — avg score < threshold (fail)
  exit 1 — mixed pipeline errors + low scores (partial error, not a skip)

run-evals.py has a hyphen in its filename, so it must be loaded via importlib.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import mock_open, patch

_repo = Path(__file__).resolve().parent.parent
_script = _repo / "scripts" / "run-evals.py"


def _load_run_evals():
    spec = importlib.util.spec_from_file_location("run_evals", _script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("run_evals", mod)
    spec.loader.exec_module(mod)
    return mod


run_evals = _load_run_evals()


# ── helpers ───────────────────────────────────────────────────────────────────


def _case(id_: str = "c1") -> dict:
    return {"id": id_, "input": "oil price?", "expected_output": "~$75"}


def _result(score: float = 0.9, pipeline_error: bool = False) -> dict:
    return {
        "case_id": "c1",
        "input": "oil price?",
        "expected_tool": "any",
        "latency_ms": 50,
        "correctness": score,
        "tool_accuracy": score,
        "score": score,
        "quality_notes": "",
        "error": None,
        "pipeline_error": pipeline_error,
    }


# ── tests ─────────────────────────────────────────────────────────────────────


def test_skip_fewer_than_3_cases(monkeypatch):
    monkeypatch.setattr(run_evals, "_load_golden_cases", lambda: [_case()])
    monkeypatch.setattr(run_evals, "_load_criteria", lambda: {})
    assert run_evals.run_scorecard() == 2


def test_skip_exactly_2_cases(monkeypatch):
    monkeypatch.setattr(
        run_evals, "_load_golden_cases", lambda: [_case("c1"), _case("c2")]
    )
    monkeypatch.setattr(run_evals, "_load_criteria", lambda: {})
    assert run_evals.run_scorecard() == 2


def test_skip_when_all_pipeline_errors(monkeypatch):
    """All 3 cases fail with PIPELINE_ERROR (e.g. Groq 429) → exit 0 (skip gracefully)."""
    monkeypatch.setattr(
        run_evals, "_load_golden_cases", lambda: [_case("c1"), _case("c2"), _case("c3")]
    )
    monkeypatch.setattr(run_evals, "_load_criteria", lambda: {})
    monkeypatch.setattr(
        run_evals,
        "_judge_case",
        lambda case, criteria, judge: _result(score=0.0, pipeline_error=True),
    )
    assert run_evals.run_scorecard() == 0


def test_pass_above_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_evals, "_load_golden_cases", lambda: [_case("c1"), _case("c2"), _case("c3")]
    )
    monkeypatch.setattr(run_evals, "_load_criteria", lambda: {})
    monkeypatch.setattr(
        run_evals, "_judge_case", lambda case, criteria, judge: _result(score=0.9)
    )
    monkeypatch.setattr(run_evals, "_results_path", lambda: tmp_path / "r.json")
    # Suppress desktop notification
    monkeypatch.setattr(
        run_evals, "notify_eval_result", lambda *a, **kw: None, raising=False
    )
    with patch("builtins.open", mock_open()):
        code = run_evals.run_scorecard(fail_below=0.80)
    assert code == 0


def test_fail_below_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_evals, "_load_golden_cases", lambda: [_case("c1"), _case("c2"), _case("c3")]
    )
    monkeypatch.setattr(run_evals, "_load_criteria", lambda: {})
    monkeypatch.setattr(
        run_evals, "_judge_case", lambda case, criteria, judge: _result(score=0.5)
    )
    monkeypatch.setattr(run_evals, "_results_path", lambda: tmp_path / "r.json")
    monkeypatch.setattr(
        run_evals, "notify_eval_result", lambda *a, **kw: None, raising=False
    )
    with patch("builtins.open", mock_open()):
        code = run_evals.run_scorecard(fail_below=0.80)
    assert code == 1


def test_mixed_errors_and_scores_not_skipped(tmp_path, monkeypatch):
    """1 pipeline error + 2 real scores → NOT a full skip → fails on low avg."""
    monkeypatch.setattr(
        run_evals, "_load_golden_cases", lambda: [_case("c1"), _case("c2"), _case("c3")]
    )
    monkeypatch.setattr(run_evals, "_load_criteria", lambda: {})
    results = [
        _result(score=0.0, pipeline_error=True),
        _result(score=0.9),
        _result(score=0.8),
    ]
    it = iter(results)
    monkeypatch.setattr(
        run_evals, "_judge_case", lambda case, criteria, judge: next(it)
    )
    monkeypatch.setattr(run_evals, "_results_path", lambda: tmp_path / "r.json")
    monkeypatch.setattr(
        run_evals, "notify_eval_result", lambda *a, **kw: None, raising=False
    )
    with patch("builtins.open", mock_open()):
        code = run_evals.run_scorecard(fail_below=0.80)
    # avg = (0 + 0.9 + 0.8) / 3 = 0.567 < 0.80
    assert code == 1
