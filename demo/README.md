# Demo app

Streamlit GUI for the oil-price-demo AgentSmith tenant.

Connects to the **live Temporal server** and running **worker** — it is a frontend
for the real pipeline, not a simulation.

## Prerequisites

The full stack must be running before you start the demo app:
- Temporal server (default: `localhost:7233`)
- `worker.py` registered on the `agent-tasks-oil-price-demo` task queue
- Postgres (for idempotency / DLQ, used by the worker)

See [OPERATIONS.md §D](../OPERATIONS.md) and the root README for setup.

## Run locally

```bash
pip install streamlit temporalio openai anthropic
streamlit run demo/app.py
```

Opens at http://localhost:8501.

Set env vars if your stack is not on localhost:

```bash
export TEMPORAL_ADDRESS=my-temporal-host:7233
export TENANT_ID=oil-price-demo
streamlit run demo/app.py
```

## Deploy to Cloud Run

The demo UI can run on Cloud Run while the Temporal worker runs elsewhere
(Cloud Run, GKE, Compute Engine — anything reachable from the demo service).

```bash
gcloud run deploy oil-price-demo-ui \
  --source . \
  --dockerfile demo/Dockerfile \
  --region us-central1 \
  --project $GCP_PROJECT_ID \
  --allow-unauthenticated \
  --port 8080 \
  --set-env-vars "TEMPORAL_ADDRESS=<your-temporal-host>:7233,TENANT_ID=oil-price-demo"
```

The CI workflow `.github/workflows/cd-demo-ui.yml` deploys automatically
on every push to `develop` or `main` that touches `demo/**`.

## Scenarios

| Preset | What it demonstrates |
|---|---|
| Normal run | All three agents complete, no HITL gate — result shows immediately |
| Price spike | Anomaly >3σ triggers HITL gate — Approve/Reject buttons send signal to Temporal |
| Custom | Enter any price series |

The HITL Approve/Reject buttons call `handle.signal("hitl_approved", True/False)` on
the Temporal workflow — exactly what `resolve_hitl.py` does from the CLI.
