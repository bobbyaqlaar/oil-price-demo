# Oil Price Agent — Domain Agents

## Architecture

```
IngestionAgent  →  PredictionAgent  →  DecisionAgent
      │                  │                  │
   Fetches            Runs ML            Places
   price data         model              order /
   from APIs          forecast           alerts
```

## HITL Triggers

- Price anomaly > 3 standard deviations: pause workflow, alert ops team
- Model confidence < 0.6: pause for human review before order

## Temporal Workflow

See `../workflows/oil_price_workflow.py` and `../workflows/activities.py`.
`OilPricePredictionWorkflow` actually subclasses
`runtime/workflows/base_workflow.py`'s `BaseAgentWorkflow` (§25) — not just
"follows the same shape," it inherits the `hitl_approved` signal directly
and demonstrates both of the framework's HITL patterns:
- The price-anomaly/low-confidence gate above uses the inherited
  `self._hitl_approved` signal (approve/reject — `run_with_hitl_gate`
  itself isn't used here since its `resume_input` is fixed before the gate
  runs, but this pipeline's resume step needs the gate's own prediction
  output as input).
- The order-placement step (`decide_action_activity`) is wrapped in
  `run_with_recoverable_step` — a malformed payload (missing/wrong-typed
  `prediction`/`confidence` fields, the same class of error as the
  framework's CRM hallucinated-field-name example) parks the workflow
  alive for a human to correct via the Ops Portal's DLQ view, rather than
  failing the run.

Worker bootstrap: `../worker.py`. Cron schedule: `../.agenticframework/schedules.yaml`.

## Status

Phase 2 — reference Temporal workflow implemented (ingestion/prediction/decision
activities, HITL approve/reject gate + edit-and-resume gate, DLQ timeout).
Domain logic (real price feed, real forecasting model, real order placement)
is left as TODOs for the tenant fork — see inline
`# TODO (tenant implementation)` markers in `../workflows/activities.py`.
