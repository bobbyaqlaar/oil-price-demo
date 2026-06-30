# Demo app

Interactive Streamlit demo for the oil-price-demo AgentSmith tenant.
Demonstrates the three-agent pipeline without requiring the full framework
runtime (Temporal, Postgres, Redis).

## Run locally

```bash
pip install streamlit openai anthropic
streamlit run demo/app.py
```

Open http://localhost:8501.

## Scenarios to try

| Preset | What it demonstrates |
|---|---|
| **Stable market** | Normal pipeline run — all three agents complete without HITL |
| **Price spike (anomaly)** | Last price is >3σ from the mean → HITL gate fires |
| **Low confidence** | Simulated low-confidence LLM response → HITL gate fires |

## Enable real LLM calls

Set any one of these env vars before starting:

```bash
export GROQ_API_KEY=gsk_...        # free tier, fastest
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

Without any key the app simulates the LLM response (same pipeline logic, random forecast).

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Point at `demo/app.py`
4. Optionally add API key secrets in the app settings
