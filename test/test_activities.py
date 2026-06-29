"""
test/test_activities.py — Unit tests for the oil-price-demo activity layer.

Tests pure logic (anomaly detection, HITL thresholds, payload validation)
without hitting any real LLM or Temporal runtime. The LLMGateway is mocked
so these pass in CI with no API keys set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── sys.path: runtime/ (for llm_gateway, idempotency etc.) + workflows/ ──────
_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo / "runtime"))
sys.path.insert(0, str(_repo / "workflows"))

from activities import (  # noqa: E402
    CONFIDENCE_HITL_THRESHOLD,
    decide_action_activity,
    fetch_oil_price_activity,
    run_prediction_activity,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _llm_result(prediction: float, confidence: float) -> SimpleNamespace:
    return SimpleNamespace(
        text=json.dumps({"prediction": prediction, "confidence": confidence}),
        cost_usd=0.001,
    )


def _mock_gw(prediction: float = 75.0, confidence: float = 0.9):
    gw = MagicMock()
    gw.complete = AsyncMock(return_value=_llm_result(prediction, confidence))
    return gw


# ── fetch_oil_price_activity ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_passthrough():
    """Activity echoes price_series and tenant_id unchanged."""
    series = [70.0, 71.0, 72.0]
    result = await fetch_oil_price_activity({"tenant_id": "t1", "price_series": series})
    assert result["price_series"] == series
    assert result["tenant_id"] == "t1"


@pytest.mark.asyncio
async def test_fetch_missing_series_defaults_empty():
    result = await fetch_oil_price_activity({"tenant_id": "t1"})
    assert result["price_series"] == []


# ── run_prediction_activity ───────────────────────────────────────────────────

# Patch target: the class in the module where it's defined (imported inside
# the function body via `from llm_gateway import LLMGateway`).

@pytest.mark.asyncio
async def test_stable_prices_no_hitl():
    """Stable series + high confidence → needs_hitl=False, is_anomaly=False."""
    series = [70.0, 71.0, 70.5, 71.5, 70.8]
    with patch("llm_gateway.LLMGateway", return_value=_mock_gw(72.0, 0.95)):
        r = await run_prediction_activity(
            {"tenant_id": "t1", "price_series": series, "workflow_run_id": "wf-1"}
        )
    assert r["needs_hitl"] is False
    assert r["is_anomaly"] is False


@pytest.mark.asyncio
async def test_price_spike_anomaly_triggers_hitl():
    """Latest price > 3σ → is_anomaly=True, needs_hitl=True."""
    # Tight cluster at 70, then a spike to 100 — well beyond 3σ
    series = [70.0, 70.1, 69.9, 70.0, 70.2, 100.0]
    with patch("llm_gateway.LLMGateway", return_value=_mock_gw(95.0, 0.95)):
        r = await run_prediction_activity(
            {"tenant_id": "t1", "price_series": series, "workflow_run_id": "wf-2"}
        )
    assert r["is_anomaly"] is True
    assert r["needs_hitl"] is True


@pytest.mark.asyncio
async def test_low_confidence_triggers_hitl():
    """Confidence below CONFIDENCE_HITL_THRESHOLD → needs_hitl=True."""
    series = [70.0, 71.0, 70.5]
    low = CONFIDENCE_HITL_THRESHOLD - 0.1
    with patch("llm_gateway.LLMGateway", return_value=_mock_gw(71.0, low)):
        r = await run_prediction_activity(
            {"tenant_id": "t1", "price_series": series, "workflow_run_id": "wf-3"}
        )
    assert r["needs_hitl"] is True
    assert r["confidence"] < CONFIDENCE_HITL_THRESHOLD


@pytest.mark.asyncio
async def test_confidence_at_threshold_no_hitl():
    """Confidence exactly at threshold is not below it → needs_hitl=False."""
    series = [70.0, 71.0, 70.5]
    with patch("llm_gateway.LLMGateway", return_value=_mock_gw(71.0, CONFIDENCE_HITL_THRESHOLD)):
        r = await run_prediction_activity(
            {"tenant_id": "t1", "price_series": series, "workflow_run_id": "wf-4"}
        )
    assert r["needs_hitl"] is False


@pytest.mark.asyncio
async def test_llm_parse_failure_falls_back_to_latest_price():
    """Unparseable LLM text → prediction=latest price, confidence=0, needs_hitl=True."""
    series = [70.0, 71.0, 72.0]
    bad = SimpleNamespace(text="not valid json", cost_usd=0.001)
    gw = MagicMock()
    gw.complete = AsyncMock(return_value=bad)
    with patch("llm_gateway.LLMGateway", return_value=gw):
        r = await run_prediction_activity(
            {"tenant_id": "t1", "price_series": series, "workflow_run_id": "wf-5"}
        )
    assert r["prediction"] == 72.0
    assert r["confidence"] == 0.0
    assert r["needs_hitl"] is True   # confidence=0 < threshold


@pytest.mark.asyncio
async def test_empty_series_no_anomaly():
    """Empty price series → std_dev=0 → no anomaly."""
    with patch("llm_gateway.LLMGateway", return_value=_mock_gw(0.0, 0.85)):
        r = await run_prediction_activity(
            {"tenant_id": "t1", "price_series": [], "workflow_run_id": "wf-6"}
        )
    assert r["is_anomaly"] is False


# ── decide_action_activity ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_decide_valid_payload():
    r = await decide_action_activity(
        {"tenant_id": "t1", "prediction": 75.0, "confidence": 0.9}
    )
    assert r["status"] == "success"
    assert r["prediction"] == 75.0
    assert r["confidence"] == 0.9


@pytest.mark.asyncio
async def test_decide_missing_prediction_raises():
    with pytest.raises(ValueError, match="missing required prediction/confidence"):
        await decide_action_activity({"tenant_id": "t1", "confidence": 0.9})


@pytest.mark.asyncio
async def test_decide_missing_confidence_raises():
    with pytest.raises(ValueError, match="missing required prediction/confidence"):
        await decide_action_activity({"tenant_id": "t1", "prediction": 75.0})


@pytest.mark.asyncio
async def test_decide_non_numeric_confidence_raises():
    with pytest.raises((ValueError, TypeError)):
        await decide_action_activity(
            {"tenant_id": "t1", "prediction": 75.0, "confidence": "high"}
        )
