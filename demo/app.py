"""
demo/app.py — Streamlit demo for the oil-price-demo AgentSmith tenant.

Demonstrates the three-agent pipeline without requiring the full framework
runtime (Temporal worker, Postgres, Redis) — the agent logic is reproduced
inline using the same thresholds and formulas as workflows/activities.py.

Run locally:
    pip install streamlit
    streamlit run demo/app.py

Deploy to Streamlit Cloud:
    Push this repo to GitHub; connect at share.streamlit.io.
    No secrets required for the simulation mode.
    Set GROQ_API_KEY (or OPENAI_API_KEY / ANTHROPIC_API_KEY) to enable
    real LLM calls via the "Live LLM" toggle in the sidebar.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Optional

import streamlit as st

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Oil Price Agent — AgentSmith Demo",
    page_icon="\U0001f6e2️",
    layout="wide",
)

# ── thresholds (must match workflows/activities.py) ────────────────────────────
ANOMALY_STD_DEV_THRESHOLD = 3.0
CONFIDENCE_HITL_THRESHOLD = 0.6


# ── inline agent logic (no framework runtime needed) ──────────────────────────

def _stats(series: list[float]) -> tuple[float, float, float]:
    n = len(series)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(series) / n
    std_dev = (sum((p - mean) ** 2 for p in series) / n) ** 0.5
    return mean, std_dev, series[-1]


def ingest(price_series: list[float], tenant_id: str) -> dict:
    """IngestionAgent — pass-through; real impl calls a price-feed API."""
    return {"price_series": price_series, "tenant_id": tenant_id}


def predict(payload: dict, llm_response: Optional[str] = None) -> dict:
    """PredictionAgent — anomaly detection + LLM (or simulated) forecast."""
    series = payload.get("price_series", [])
    mean, std_dev, latest = _stats(series)
    is_anomaly = std_dev > 0 and abs(latest - mean) > ANOMALY_STD_DEV_THRESHOLD * std_dev

    if llm_response:
        try:
            parsed = json.loads(llm_response)
            prediction = float(parsed["prediction"])
            confidence = float(parsed["confidence"])
        except Exception:
            prediction, confidence = latest, 0.0
    else:
        # Simulated forecast: small random drift; confidence drops on anomaly
        prediction = round(latest * random.uniform(0.98, 1.02), 2)
        base_conf = random.uniform(0.45, 0.75) if is_anomaly else random.uniform(0.65, 0.97)
        confidence = round(base_conf, 2)

    needs_hitl = is_anomaly or confidence < CONFIDENCE_HITL_THRESHOLD
    return {
        "tenant_id": payload["tenant_id"],
        "prediction": prediction,
        "confidence": confidence,
        "is_anomaly": is_anomaly,
        "needs_hitl": needs_hitl,
        "mean": round(mean, 2),
        "std_dev": round(std_dev, 2),
        "latest": latest,
    }


def _call_llm(series: list[float]) -> Optional[str]:
    """Call a real LLM if any key is configured; return raw JSON string or None."""
    prompt = (
        f"Given recent oil prices {series}, predict the next price point and a "
        f'confidence score (0-1). Reply with ONLY valid JSON: '
        f'{{"prediction": <float>, "confidence": <float>}}'
    )
    try:
        groq_key = os.environ.get("GROQ_API_KEY", "")
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

        if groq_key:
            from openai import OpenAI  # groq is openai-compatible
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=64,
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        elif openai_key:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=64,
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        elif anthropic_key:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=64,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
    except Exception as exc:
        st.warning(f"LLM call failed ({exc}) — falling back to simulation.")
    return None


def decide(payload: dict) -> dict:
    """DecisionAgent — validate payload and place order / send alert."""
    if "prediction" not in payload or not isinstance(payload.get("confidence"), (int, float)):
        raise ValueError("Missing prediction/confidence fields")
    action = "ORDER_PLACED" if payload.get("confidence", 0) >= CONFIDENCE_HITL_THRESHOLD else "ALERT_SENT"
    return {
        "status": "success",
        "prediction": payload.get("prediction"),
        "confidence": payload.get("confidence"),
        "action": action,
    }


# ── session state ──────────────────────────────────────────────────────────────
for key, default in [("history", []), ("hitl_pending", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("\U0001f6e2️ Oil Price Agent")
    st.caption("AgentSmith framework · tenant demo")
    st.markdown("---")

    st.subheader("Settings")
    tenant_id = st.text_input("Tenant ID", value="oil-price-demo")

    live_llm = st.toggle(
        "Live LLM calls",
        value=False,
        help="Uses GROQ_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY if set. Falls back to simulation.",
    )

    st.markdown("---")
    st.subheader("Price series")
    preset = st.selectbox(
        "Preset scenario",
        ["Stable market", "Price spike (anomaly)", "Low confidence", "Custom"],
    )

    presets = {
        "Stable market":          "80.1, 80.5, 79.8, 80.2, 80.0, 79.9, 80.3",
        "Price spike (anomaly)":  "80.0, 80.2, 79.8, 80.1, 80.0, 79.9, 120.5",
        "Low confidence":         "80.0, 80.1, 79.9, 80.2, 80.0",
        "Custom":                 "80.0, 80.5, 79.8, 80.2, 80.0",
    }
    series_input = st.text_area(
        "Prices (comma-separated, USD/bbl)", value=presets[preset], height=80
    )

    manual_llm = st.text_area(
        "Override LLM response (optional JSON)",
        placeholder='{"prediction": 81.5, "confidence": 0.87}',
        height=68,
        help="Paste any JSON here to bypass the LLM and test a specific forecast.",
    )

    run_btn = st.button("▶️  Run pipeline", type="primary", use_container_width=True)

# ── header ──────────────────────────────────────────────────────────────────────
st.title("Oil Price Prediction Pipeline")
st.caption(
    "Three-agent pipeline: **Ingestion → Prediction → Decision** "
    "with Human-in-the-Loop (HITL) gate on anomaly or low confidence."
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### 1 \xb7 IngestionAgent")
    st.info("Fetches and validates the price series.\n\n_Real impl: calls a live price-feed API._")
with c2:
    st.markdown("### 2 \xb7 PredictionAgent")
    st.info(
        "Detects price anomalies (>3σ) and calls an LLM to forecast "
        "the next price point + confidence score."
    )
with c3:
    st.markdown("### 3 \xb7 DecisionAgent")
    st.info(
        "Places the order (or sends alert) on approved predictions. "
        "Routes anomalies / low-confidence runs to the HITL queue first."
    )

st.markdown("---")

# ── pipeline run ───────────────────────────────────────────────────────────────
if run_btn:
    try:
        series = [float(x.strip()) for x in series_input.split(",") if x.strip()]
    except ValueError:
        st.error("Invalid price series — enter comma-separated numbers.")
        st.stop()
    if len(series) < 3:
        st.error("Need at least 3 price points.")
        st.stop()

    with st.status("Running pipeline…", expanded=True) as status:
        # Step 1
        st.write("**Step 1 — IngestionAgent**")
        ingested = ingest(series, tenant_id)
        time.sleep(0.3)
        st.write(f" Loaded {len(series)} price points for tenant `{tenant_id}`")

        # Step 2
        st.write("**Step 2 — PredictionAgent**")
        override = manual_llm.strip() or None
        if not override and live_llm:
            with st.spinner("Calling LLM…"):
                override = _call_llm(series)
        pred = predict(ingested, llm_response=override)
        time.sleep(0.2)

        a_icon = "\U0001f6a8" if pred["is_anomaly"] else "✅"
        c_icon = "⚠️" if pred["confidence"] < CONFIDENCE_HITL_THRESHOLD else "✅"
        st.write(
            f" Anomaly: {a_icon}  |  Confidence: {c_icon} {pred['confidence']:.2f}  |  "
            f"Forecast: **${pred['prediction']}**"
        )
        st.write(
            f" _(latest ${pred['latest']}, mean ${pred['mean']}, σ {pred['std_dev']})_"
        )

        # Step 3
        if pred["needs_hitl"]:
            reasons = []
            if pred["is_anomaly"]:
                reasons.append(
                    f"price spike >3σ "
                    f"(|{pred['latest']} − {pred['mean']}| > "
                    f"{ANOMALY_STD_DEV_THRESHOLD}\xd7{pred['std_dev']})"
                )
            if pred["confidence"] < CONFIDENCE_HITL_THRESHOLD:
                reasons.append(f"low confidence ({pred['confidence']:.2f} < {CONFIDENCE_HITL_THRESHOLD})")
            st.write(f"**Step 3 — HITL gate triggered:** {'; '.join(reasons)}")
            st.session_state.hitl_pending = pred
            status.update(label="⏸️  Paused — awaiting human review", state="running")
        else:
            st.write("**Step 3 — DecisionAgent**")
            result = decide(pred)
            time.sleep(0.2)
            st.write(f" Action: **{result['action']}** at ${result['prediction']} (conf {result['confidence']:.2f})")
            st.session_state.hitl_pending = None
            st.session_state.history.append({**pred, "action": result["action"], "hitl": False})
            status.update(label="✅  Pipeline complete", state="complete")

# ── HITL review panel ─────────────────────────────────────────────────────────
if st.session_state.hitl_pending:
    p = st.session_state.hitl_pending
    st.markdown("---")
    st.markdown("## \U0001f6d1 Human-in-the-Loop Review")
    st.warning(
        f"Pipeline paused — prediction **${p['prediction']}** "
        f"(confidence {p['confidence']:.2f}) requires human approval before the order is placed."
    )

    reasons = []
    if p.get("is_anomaly"):
        reasons.append(
            f"**Price anomaly:** |{p['latest']} − {p['mean']}| > "
            f"{ANOMALY_STD_DEV_THRESHOLD}\xd7{p['std_dev']} standard deviations"
        )
    if p.get("confidence", 1) < CONFIDENCE_HITL_THRESHOLD:
        reasons.append(f"**Low model confidence:** {p['confidence']:.2f} < threshold {CONFIDENCE_HITL_THRESHOLD}")
    for r in reasons:
        st.markdown(f"- {r}")

    ca, cb, _ = st.columns([2, 2, 4])
    with ca:
        if st.button("✅ Approve", type="primary", use_container_width=True):
            result = decide(p)
            st.session_state.history.append({**p, "action": result["action"], "hitl": True, "hitl_decision": "approved"})
            st.session_state.hitl_pending = None
            st.success(f"Approved → **{result['action']}** at ${result['prediction']}")
            st.rerun()
    with cb:
        if st.button("❌ Reject", use_container_width=True):
            st.session_state.history.append({**p, "action": "REJECTED", "hitl": True, "hitl_decision": "rejected"})
            st.session_state.hitl_pending = None
            st.error("Rejected — prediction discarded, routed to Dead-Letter Queue.")
            st.rerun()

# ── run history ───────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("## Run history")
    h = st.session_state.history
    m1, m2, m3 = st.columns(3)
    m1.metric("Total runs", len(h))
    m2.metric("HITL-flagged", sum(1 for r in h if r.get("hitl")))
    m3.metric("Avg confidence", f"{sum(r['confidence'] for r in h) / len(h):.2f}")

    rows = [
        {
            "Prediction ($)": r.get("prediction"),
            "Confidence": f"{r.get('confidence', 0):.2f}",
            "Anomaly": "\U0001f6a8" if r.get("is_anomaly") else "—",
            "HITL": (
                "✅ approved" if r.get("hitl_decision") == "approved"
                else ("❌ rejected" if r.get("hitl_decision") == "rejected"
                      else "—")
            ),
            "Action": r.get("action", "—"),
        }
        for r in reversed(h)
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

# ── footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "AgentSmith framework \xb7 oil-price-demo \xb7 "
    "[github.com/bobbyaqlaar/oil-price-demo](https://github.com/bobbyaqlaar/oil-price-demo)"
)
