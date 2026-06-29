"""
runtime/llm_gateway.py — Production LLM gateway.

Centralised routing for all production agent LLM calls.
Replaces cost_router.py for production use.

Responsibilities:
  - Accurate per-model pricing (from models.yaml, not blended estimates)
  - Per-tenant budget enforcement (reads from budget store — Postgres or Redis)
  - Degrade ladder on budget/quota breach
  - Audit trail: every call recorded as span attributes
  - Idempotency: duplicate calls short-circuited via idempotency.py

Degrade ladder (on budget breach or provider throttle):
  1. Throttle   — exponential backoff on request rate
  2. Downgrade  — route to cheaper tier in models.yaml
  3. Queue      — delay tasks with exponential backoff
  4. Local      — switch to Ollama if OLLAMA_BASE_URL is configured
  5. Alert      — Ops Portal + Slack/Teams

Workers MUST NOT import cost_router.py directly.

See SPECS.md §29 for full specification.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CompletionResult:
    """Result of a gateway-routed completion."""

    text: str
    model_used: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    degrade_tier: Optional[str] = None  # None = nominal; "downgrade" | "local" | etc.


class BudgetExceededError(RuntimeError):
    """Raised when the degrade ladder is exhausted (halt + alert tier, §29)."""


# ── Model registry (§29 Model Registry) ───────────────────────────────────────


def _repo_root() -> Path:
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / ".git").exists():
            return parent
    return cwd


_FRAMEWORK_MODELS_YAML = Path(__file__).resolve().parent / "models.yaml"


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore

        with path.open() as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def load_model_registry() -> dict:
    """
    Load the model registry: framework defaults from runtime/models.yaml,
    overridden by a tenant repo's own models.yaml (if present), overridden
    again by `.agenticframework/tenant.yaml` -> gateway.routing_overrides
    (a per-role model id shorthand, §29).
    """
    registry: dict = {}
    for role, cfg in _load_yaml(_FRAMEWORK_MODELS_YAML).get("models", {}).items():
        registry[role] = dict(cfg)

    root = _repo_root()
    tenant_models_path = root / "models.yaml"
    if tenant_models_path.exists():
        for role, cfg in _load_yaml(tenant_models_path).get("models", {}).items():
            registry[role] = {**registry.get(role, {}), **cfg}

    tenant_yaml_path = root / ".agenticframework" / "tenant.yaml"
    if tenant_yaml_path.exists():
        tenant_cfg = _load_yaml(tenant_yaml_path)
        overrides = (tenant_cfg.get("gateway") or {}).get("routing_overrides") or {}
        for role, model_id in overrides.items():
            registry.setdefault(role, {})
            registry[role]["id"] = model_id

    return registry


# ── Budget store (Postgres or Redis backend, §25/§29) ─────────────────────────


def _current_period() -> str:
    """ "YYYY-MM" in UTC — pinned explicitly rather than via bare
    time.strftime("%Y-%m"), which uses the server's LOCAL timezone.
    portal/lib/cost.ts derives the same period via
    `new Date().toISOString().slice(0, 7)`, which is always UTC; a worker
    running in a non-UTC server timezone could otherwise disagree with the
    portal for several hours around a month boundary, putting spend in the
    "wrong" month from the portal's point of view (FIXES_AND_CLEANUP.md 4.15).
    """
    return time.strftime("%Y-%m", time.gmtime())


@dataclass
class BudgetStatus:
    tenant_id: str
    spent_usd: float
    cap_usd: float
    period_start: str

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self.spent_usd)

    @property
    def breached(self) -> bool:
        return self.spent_usd >= self.cap_usd


class _BudgetBackend:
    def get_spend(self, tenant_id: str) -> float:
        raise NotImplementedError

    def add_spend(self, tenant_id: str, amount_usd: float) -> float:
        raise NotImplementedError

    def try_reserve(self, tenant_id: str, amount_usd: float, cap_usd: float) -> bool:
        """Atomically add amount_usd to spend IF the result would not exceed
        cap_usd, in one indivisible operation. Returns True if reserved (the
        amount has already been added to spend) or False if it would have
        breached the cap (nothing was added).

        This exists because the old pattern — a separate get_budget_status()
        read, then an add_spend() write only after the (slow, variable-cost)
        LLM call returns — left a window where N concurrent calls for the
        same tenant could all read "not breached" before any of them
        recorded spend, letting the combined cost of every in-flight call
        blow through the monthly cap (FIXES_AND_CLEANUP.md 2.1). Callers
        reserve an upper-bound cost estimate via try_reserve() before
        invoking the provider, then reconcile the estimate vs. actual cost
        afterward via add_spend()'s signed delta.
        """
        raise NotImplementedError


class _MemoryBudgetBackend(_BudgetBackend):
    """Single-process budget tracking. Suitable for dev/CI; not for multi-worker prod fleets."""

    def __init__(self) -> None:
        self._spend: dict[str, float] = {}
        self._lock = threading.Lock()

    def get_spend(self, tenant_id: str) -> float:
        with self._lock:
            return self._spend.get(tenant_id, 0.0)

    def add_spend(self, tenant_id: str, amount_usd: float) -> float:
        with self._lock:
            self._spend[tenant_id] = self._spend.get(tenant_id, 0.0) + amount_usd
            return self._spend[tenant_id]

    def try_reserve(self, tenant_id: str, amount_usd: float, cap_usd: float) -> bool:
        with self._lock:
            current = self._spend.get(tenant_id, 0.0)
            if current + amount_usd > cap_usd:
                return False
            self._spend[tenant_id] = current + amount_usd
            return True


class _RedisBudgetBackend(_BudgetBackend):
    def __init__(self) -> None:
        import redis  # type: ignore

        self._client = redis.from_url(os.environ["REDIS_URL"])

    def _key(self, tenant_id: str) -> str:
        period = _current_period()
        return f"agenticframework:budget:{tenant_id}:{period}"

    def get_spend(self, tenant_id: str) -> float:
        val = self._client.get(self._key(tenant_id))
        return float(val) if val else 0.0

    def add_spend(self, tenant_id: str, amount_usd: float) -> float:
        key = self._key(tenant_id)
        new_total = self._client.incrbyfloat(key, amount_usd)
        self._client.expire(key, 40 * 86400)  # budgets are monthly; expire stale keys
        return float(new_total)

    def try_reserve(self, tenant_id: str, amount_usd: float, cap_usd: float) -> bool:
        # INCRBYFLOAT is atomic, but "increment, then check, then maybe
        # undo" is not a single atomic step — between two concurrent
        # INCRBYFLOATs both can observe a total over cap and both roll back,
        # or (the actually dangerous case) both can observe under cap before
        # either's increment is visible to the other... except INCRBYFLOAT
        # itself serializes on the key (Redis commands on one key run one at
        # a time), so the *increment* ordering is always consistent — the
        # post-hoc compensating DECRBYFLOAT below is what makes the
        # reserve-or-release atomic *with respect to the cap*, not the
        # increment itself.
        key = self._key(tenant_id)
        new_total = float(self._client.incrbyfloat(key, amount_usd))
        if new_total > cap_usd:
            self._client.incrbyfloat(key, -amount_usd)
            return False
        self._client.expire(key, 40 * 86400)
        return True


class _PostgresBudgetBackend(_BudgetBackend):
    def __init__(self) -> None:
        self._dsn = os.environ["DATABASE_URL"]
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS llm_gateway_budget (
                        tenant_id TEXT NOT NULL,
                        period TEXT NOT NULL,
                        spent_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
                        PRIMARY KEY (tenant_id, period)
                    )
                    """
                )
        finally:
            conn.close()

    def _connect(self):
        import psycopg2  # type: ignore

        return psycopg2.connect(self._dsn)

    def _period(self) -> str:
        return _current_period()

    def get_spend(self, tenant_id: str) -> float:
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT spent_usd FROM llm_gateway_budget WHERE tenant_id = %s AND period = %s",
                    (tenant_id, self._period()),
                )
                row = cur.fetchone()
                return float(row[0]) if row else 0.0
        finally:
            conn.close()

    def add_spend(self, tenant_id: str, amount_usd: float) -> float:
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO llm_gateway_budget (tenant_id, period, spent_usd)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (tenant_id, period)
                    DO UPDATE SET spent_usd = llm_gateway_budget.spent_usd + EXCLUDED.spent_usd
                    RETURNING spent_usd
                    """,
                    (tenant_id, self._period(), amount_usd),
                )
                new_total = cur.fetchone()[0]
                return float(new_total)
        finally:
            conn.close()

    def try_reserve(self, tenant_id: str, amount_usd: float, cap_usd: float) -> bool:
        # Single atomic statement: the row is inserted/updated and the cap
        # check happens in the same WHERE clause Postgres evaluates under
        # the row lock taken for the UPDATE, so no other transaction's
        # concurrent reserve on the same (tenant_id, period) can interleave
        # between "check" and "act" the way the old read-then-write did.
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO llm_gateway_budget (tenant_id, period, spent_usd)
                    VALUES (%s, %s, 0)
                    ON CONFLICT (tenant_id, period) DO NOTHING
                    """,
                    (tenant_id, self._period()),
                )
                cur.execute(
                    """
                    UPDATE llm_gateway_budget
                    SET spent_usd = spent_usd + %s
                    WHERE tenant_id = %s AND period = %s AND spent_usd + %s <= %s
                    RETURNING spent_usd
                    """,
                    (amount_usd, tenant_id, self._period(), amount_usd, cap_usd),
                )
                row = cur.fetchone()
                return row is not None
        finally:
            conn.close()


def _make_budget_backend() -> _BudgetBackend:
    backend = os.environ.get("BUDGET_BACKEND", "memory").lower()
    if backend == "redis":
        return _RedisBudgetBackend()
    if backend == "postgres":
        return _PostgresBudgetBackend()
    if backend == "memory":
        return _MemoryBudgetBackend()
    raise ValueError(
        f"Unknown BUDGET_BACKEND={backend!r}. Use 'memory', 'redis', or 'postgres'."
    )


# ── Gateway ───────────────────────────────────────────────────────────────────


class LLMGateway:
    """
    Production LLM gateway. Instantiate once per worker process.

    Usage:
        gateway = LLMGateway(tenant_id="acme")
        result = await gateway.complete(
            prompt=messages,
            model_hint="developer",
            workflow_id="wf-oil-0042",
            idempotency_key="sha256:...",
        )
    """

    def __init__(self, tenant_id: str, budget_cap_usd: Optional[float] = None) -> None:
        self.tenant_id = tenant_id
        self.models = load_model_registry()
        self.budget_cap_usd = (
            budget_cap_usd
            if budget_cap_usd is not None
            else float(os.environ.get("AGENT_MONTHLY_USD_CAP", "150.0"))
        )
        self._budget = _make_budget_backend()
        self._idempotency = self._make_idempotency_store()

    @staticmethod
    def _make_idempotency_store():
        try:
            from idempotency import IdempotencyStore  # type: ignore

            return IdempotencyStore()
        except Exception as exc:
            # Missing REDIS_URL/DATABASE_URL, backend lib not installed, etc.
            # Degrades gracefully to "no idempotency" (duplicate-call
            # suppression simply doesn't happen) rather than failing gateway
            # construction — but logged, not silently invisible, since this
            # now means a real backend failed to initialize, not "not
            # implemented yet".
            logger.warning(
                "idempotency store unavailable, duplicate-call suppression disabled: %s",
                exc,
            )
            return None

    # ── Budget ────────────────────────────────────────────────────────────────

    def get_budget_status(self) -> BudgetStatus:
        """Return current budget status for this tenant."""
        spent = self._budget.get_spend(self.tenant_id)
        return BudgetStatus(
            tenant_id=self.tenant_id,
            spent_usd=spent,
            cap_usd=self.budget_cap_usd,
            period_start=f"{_current_period()}-01",
        )

    # ── Degrade ladder (§29) ─────────────────────────────────────────────────

    def _degrade_chain(self, model_hint: str) -> list[str]:
        """[model_hint, ...downgrade targets...] following models.yaml degrade_to links."""
        chain = [model_hint]
        current = model_hint
        seen = {model_hint}
        while True:
            nxt = self.models.get(current, {}).get("degrade_to")
            if not nxt or nxt in seen:
                break
            chain.append(nxt)
            seen.add(nxt)
            current = nxt
        return chain

    @staticmethod
    def _is_free_tier(cfg: dict) -> bool:
        return (
            cfg.get("provider") == "ollama" or cfg.get("cost_per_input_token", 1) == 0
        )

    @staticmethod
    def _is_provider_exhausted(exc: Exception) -> bool:
        """True when the provider itself is unavailable for this key/tier — no point
        retrying; degrade to the next tier instead.  Covers billing, quota, and
        auth errors that will not resolve on their own."""
        msg = str(exc).lower()
        return any(
            k in msg
            for k in (
                "credit balance is too low",
                "insufficient_quota",
                "rate limit",
                "billing",
                "payment required",
                "429",
                "overloaded",
            )
        )

    def _resolve_role(
        self, model_hint: str, budget: BudgetStatus
    ) -> tuple[str, Optional[str]]:
        """Walk the degrade ladder. Returns (role, degrade_tier); degrade_tier is None at full strength."""
        if not budget.breached:
            return model_hint, None

        hint_cfg = self.models.get(model_hint, {})
        if self._is_free_tier(hint_cfg):
            # Already on the cheapest/local tier — nothing left to degrade to, and
            # using it doesn't add spend, so the breach doesn't block it.
            return model_hint, None

        chain = self._degrade_chain(model_hint)
        if len(chain) < 2:
            raise BudgetExceededError(
                f"tenant={self.tenant_id} budget exhausted (${budget.spent_usd:.2f}/${budget.cap_usd:.2f}) "
                f"and no cheaper tier available below {model_hint!r}. Halting (alert tier)."
            )

        next_role = chain[1]
        next_cfg = self.models.get(next_role, {})
        tier = "local" if self._is_free_tier(next_cfg) else "downgrade"
        return next_role, tier

    # ── Run status reporting (Ops Portal, FIXES_AND_CLEANUP.md P2a) ─────────

    def _report_run_status(
        self,
        run_id: str,
        status: str,
        workflow_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        error_summary: Optional[str] = None,
    ) -> None:
        """Best-effort POST to the Ops Portal's run-status ingest endpoint —
        gated on OPS_PORTAL_URL being set, fails open (logs, never raises)
        on any error. Same philosophy as every other runtime/-to-portal
        call in this codebase (e.g. _ai_audit_log_event in
        install-ai-stack.sh): never let optional observability infra block
        or fail the actual LLM call.

        workflow_id is the grouping key portal/lib/runStatus.ts uses to
        aggregate multiple gateway calls within one workflow run (it was
        previously dropped here despite being accepted by the ingest route
        and the agent_runs.workflow_id column — every row landed with
        workflow_id=NULL regardless of what the caller passed to
        complete()).
        """
        ops_portal_url = os.environ.get("OPS_PORTAL_URL")
        sync_token = os.environ.get("OPS_PORTAL_SYNC_TOKEN")
        if not ops_portal_url or not sync_token:
            return
        try:
            import httpx

            httpx.post(
                f"{ops_portal_url.rstrip('/')}/api/runs/ingest",
                json={
                    "tenantId": self.tenant_id,
                    "runId": run_id,
                    "workflowId": workflow_id,
                    "status": status,
                    "traceId": trace_id,
                    "errorSummary": error_summary,
                },
                headers={"Authorization": f"Bearer {sync_token}"},
                timeout=5.0,
            )
        except Exception as exc:
            logger.debug(
                "run-status report failed tenant=%s run_id=%s: %s",
                self.tenant_id,
                run_id,
                exc,
            )

    # ── Span attribute recording (§15, §29) ──────────────────────────────────

    def _record_span_attributes(
        self,
        role: str,
        model_id: str,
        degrade_tier: Optional[str],
        workflow_id: Optional[str],
        cost_usd: float,
    ) -> None:
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            if span is None:
                return
            span.set_attribute("tenant.id", self.tenant_id)
            span.set_attribute("llm.model_name", model_id)
            span.set_attribute("llm.gateway.tier", role)
            span.set_attribute("llm.gateway.cost_usd", cost_usd)
            if degrade_tier:
                span.set_attribute("llm.gateway.degrade_reason", degrade_tier)
            if workflow_id:
                span.set_attribute("workflow.id", workflow_id)
        except Exception:  # fail-open: tracing must never break the actual LLM call
            pass

    # ── Completion ────────────────────────────────────────────────────────────

    async def complete(
        self,
        prompt: Any,
        model_hint: str = "developer",
        workflow_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> CompletionResult:
        """
        Route a completion request with per-tenant budget enforcement.

        model_hint options: "architect" | "developer" | "validator" | "fast"
        """
        if idempotency_key and self._idempotency is not None:
            try:
                cached = self._idempotency.get(idempotency_key)
                if cached is not None:
                    logger.info(
                        "idempotency cache hit tenant=%s key=%s",
                        self.tenant_id,
                        idempotency_key,
                    )
                    return CompletionResult(**cached)
                logger.debug(
                    "idempotency cache miss tenant=%s key=%s",
                    self.tenant_id,
                    idempotency_key,
                )
            except Exception as exc:
                # Now that the backends are real (Postgres/Redis), a failure
                # here is a live infra error (DB down, bad creds), not the
                # old "backend not implemented" case — log it instead of
                # silently treating every failure as a cache miss.
                logger.error(
                    "idempotency lookup failed tenant=%s key=%s: %s",
                    self.tenant_id,
                    idempotency_key,
                    exc,
                )

        budget = self.get_budget_status()
        role, degrade_tier = self._resolve_role(model_hint, budget)

        cfg = self.models.get(role)
        if not cfg:
            raise ValueError(
                f"No model registered for role {role!r}. Check models.yaml."
            )

        model_id = cfg["id"]
        messages = self._coerce_messages(prompt)

        # Reserve an upper-bound cost estimate atomically before the call,
        # not after — closes the check-then-act race where concurrent calls
        # could all observe "not breached" before any of them recorded
        # spend (FIXES_AND_CLEANUP.md 2.1). max_tokens bounds output cost
        # exactly; input cost is bounded by the same max_tokens too since we
        # don't know the actual prompt token count until the provider
        # responds — this overestimates input cost, which only makes the
        # gateway degrade *earlier* under contention, never later.
        estimated_cost_usd = max_tokens * (
            cfg.get("cost_per_input_token", 0) + cfg.get("cost_per_output_token", 0)
        )
        reserved = True
        if estimated_cost_usd and not self._is_free_tier(cfg):
            reserved = self._budget.try_reserve(
                self.tenant_id, estimated_cost_usd, self.budget_cap_usd
            )
            if not reserved:
                raise BudgetExceededError(
                    f"tenant={self.tenant_id} budget reservation of ${estimated_cost_usd:.4f} for "
                    f"model_hint={model_hint!r} would exceed cap (${budget.spent_usd:.2f}/${budget.cap_usd:.2f}). "
                    "Concurrent in-flight calls already reserved the remaining budget."
                )

        # run_id is always unique per CALL, never reused across multiple
        # gateway.complete() calls within one workflow run — a workflow
        # that makes 2+ calls (the expected shape for multi-agent/
        # multi-LLM tenant apps, not just the single-call oil-price
        # example) would otherwise have call #2's "running" report
        # re-upsert call #1's already-"success" agent_runs row, resetting
        # finished_at to NULL and making the widget show "running" for a
        # workflow that's actually done. workflow_id is reported
        # separately (see _report_run_status) purely as a grouping key —
        # portal/lib/runStatus.ts aggregates all calls sharing a
        # workflow_id (including concurrent/parallel ones, e.g. fan-out to
        # multiple LLMs) into one widget status rather than relying on
        # row identity to do that grouping.
        run_id = (
            f"{workflow_id}-{uuid.uuid4().hex[:8]}"
            if workflow_id
            else f"{self.tenant_id}-{uuid.uuid4().hex[:12]}"
        )
        self._report_run_status(run_id, "running", workflow_id=workflow_id)

        # Try the chosen tier; on provider-level exhaustion (billing, quota,
        # overload) walk the degrade_to chain rather than failing immediately.
        degrade_chain = self._degrade_chain(role)
        tried: list[str] = []
        text = in_tok = out_tok = None
        last_exc: Exception | None = None
        for attempt_role in degrade_chain:
            attempt_cfg = self.models.get(attempt_role)
            if not attempt_cfg:
                continue
            try:
                text, in_tok, out_tok = await self._invoke(
                    attempt_cfg, messages, max_tokens, temperature
                )
                if attempt_role != role:
                    # Record the tier we actually used
                    degrade_tier = (
                        "local" if self._is_free_tier(attempt_cfg) else "downgrade"
                    )
                    cfg = attempt_cfg
                    model_id = attempt_cfg["id"]
                    logger.warning(
                        "Degraded from %r to %r due to provider error: %s",
                        role,
                        attempt_role,
                        last_exc,
                    )
                last_exc = None
                break
            except Exception as exc:
                tried.append(attempt_role)
                last_exc = exc
                if self._is_provider_exhausted(exc):
                    logger.warning(
                        "Provider exhausted for role=%r model=%r: %s — trying next tier",
                        attempt_role,
                        attempt_cfg.get("id"),
                        exc,
                    )
                    continue
                # Non-exhaustion error (bad prompt, network timeout, etc.) — fail fast
                break

        if last_exc is not None:
            if reserved and estimated_cost_usd:
                self._budget.add_spend(
                    self.tenant_id, -estimated_cost_usd
                )  # release the reservation
            self._report_run_status(
                run_id,
                "failed",
                workflow_id=workflow_id,
                error_summary=str(last_exc)[:500],
            )
            if tried:
                raise RuntimeError(
                    f"All model tiers exhausted (tried: {tried}). Last error: {last_exc}"
                ) from last_exc
            raise last_exc

        cost_usd = in_tok * cfg.get("cost_per_input_token", 0) + out_tok * cfg.get(
            "cost_per_output_token", 0
        )
        if reserved and estimated_cost_usd:
            # Reconcile: replace the (conservative) reservation with the
            # actual cost. The delta can be negative (actual < estimate,
            # the common case) or positive (rare, e.g. provider returned
            # more output tokens than max_tokens would suggest) — add_spend
            # accepts a signed amount either way.
            delta = cost_usd - estimated_cost_usd
            if delta:
                self._budget.add_spend(self.tenant_id, delta)
        elif cost_usd:
            self._budget.add_spend(self.tenant_id, cost_usd)

        self._record_span_attributes(
            role, model_id, degrade_tier, workflow_id, cost_usd
        )
        self._report_run_status(
            run_id, "degraded" if degrade_tier else "success", workflow_id=workflow_id
        )

        result = CompletionResult(
            text=text,
            model_used=model_id,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost_usd,
            degrade_tier=degrade_tier,
        )

        if idempotency_key and self._idempotency is not None:
            try:
                self._idempotency.set(idempotency_key, result.__dict__)
            except Exception as exc:
                logger.error(
                    "idempotency write failed tenant=%s key=%s: %s",
                    self.tenant_id,
                    idempotency_key,
                    exc,
                )

        return result

    @staticmethod
    def _coerce_messages(prompt: Any) -> list[dict]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        if isinstance(prompt, list):
            return prompt
        raise TypeError(f"prompt must be str or list[dict], got {type(prompt)}")

    @staticmethod
    def _is_retryable_provider_error(exc: BaseException) -> bool:
        """Transient-only: connection/timeout issues, 429 (rate limit), and
        5xx (provider-side fault) — never 4xx other than 429, since a bad
        request/auth/model-id error will fail identically on retry and
        retrying it just burns the attempt budget for no benefit."""
        import httpx

        if isinstance(exc, httpx.TransportError):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            return status == 429 or status >= 500
        return False

    async def _invoke(
        self, cfg: dict, messages: list[dict], max_tokens: int, temperature: float
    ) -> tuple[str, int, int]:
        """Call the provider for this model config. Returns (text, input_tokens, output_tokens).

        Request building / response parsing delegated to
        runtime/provider_dispatch.py, shared with scripts/cost_router.py
        (FIXES_AND_CLEANUP.md 4.3) — only the base_url/api_key resolution
        below (which legitimately differs: this is the production path with
        its own model registry, cost_router.py has its own env-var-driven
        route table) stays local to this method.

        Retries transient failures with exponential backoff (this module's
        own docstring has documented a "Throttle: exponential backoff on
        request rate" degrade-ladder step from the start — `tenacity` was
        already a required dependency for exactly this, but nothing in the
        codebase actually called it until now). Non-transient errors (bad
        request, auth failure, unknown model) raise immediately — retrying
        those would just waste the attempt budget on a failure that can't
        succeed differently the second time.
        """
        import httpx
        from tenacity import (
            retry,
            retry_if_exception,
            stop_after_attempt,
            wait_exponential,
        )

        try:
            from runtime.provider_dispatch import (
                build_cloud_request,
                build_request,
                is_cloud_provider,
                parse_cloud_response,
                parse_response,
            )
        except ImportError:
            from provider_dispatch import (  # type: ignore
                build_cloud_request,
                build_request,
                is_cloud_provider,
                parse_cloud_response,
                parse_response,
            )

        provider = cfg.get("provider", "openai")
        model_id = cfg["id"]

        if is_cloud_provider(provider):
            # Cloud-native providers (vertex_ai/azure_openai/bedrock/
            # huawei_modelarts) need their own auth scheme and URL/envelope
            # shape, not just a different host — provider_dispatch.py's
            # CloudProviderAdapter owns that, and returns a full URL rather
            # than a path since project/region/deployment/endpoint-id are
            # baked into the URL itself.
            url, headers, body = build_cloud_request(
                provider, model_id, messages, cfg, max_tokens, temperature
            )
        else:
            # base_url/api_key_env are config-driven (models.yaml `endpoint` /
            # `api_key_env` fields) so a tenant can point a provider at a proxy,
            # a region-pinned host, or a differently-named API key env var
            # (e.g. per-tenant keys) without editing this code. The literals
            # below are fallbacks for the common case only — direct Anthropic/
            # OpenAI calls — not a ceiling on what's supported.
            if provider == "anthropic":
                api_key_env = cfg.get("api_key_env", "ANTHROPIC_API_KEY")
                api_key = os.environ.get(api_key_env, "")
                base_url = cfg.get("endpoint") or "https://api.anthropic.com"
            elif provider == "ollama":
                base_url = os.path.expandvars(
                    cfg.get("endpoint", "${OLLAMA_BASE_URL}/v1")
                )
                # expandvars leaves unset variables as literal "${VAR}" — not a valid URL.
                if not base_url.startswith("http"):
                    base_url = (
                        os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                        + "/v1"
                    )
                api_key = "ollama"
            elif provider == "groq":
                # Groq's API is OpenAI-compatible (same request/response shape,
                # parse_response's non-anthropic branch handles it) — only the
                # host and API key env var differ from direct OpenAI, same as
                # every other "openai_compatible" provider in this codebase.
                api_key_env = cfg.get("api_key_env", "GROQ_API_KEY")
                api_key = os.environ.get(api_key_env, "")
                base_url = cfg.get("endpoint") or "https://api.groq.com/openai/v1"
            else:
                api_key_env = cfg.get("api_key_env", "OPENAI_API_KEY")
                api_key = os.environ.get(api_key_env, "")
                base_url = cfg.get("endpoint") or "https://api.openai.com/v1"
            base_url = os.path.expandvars(base_url)

            path, headers, body = build_request(
                provider, model_id, messages, api_key, max_tokens, temperature
            )
            url = base_url.rstrip("/") + path

        @retry(
            retry=retry_if_exception(self._is_retryable_provider_error),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )
        async def _post_with_retry() -> dict:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, json=body, headers=headers)
                if not resp.is_success:
                    # Try to surface a human-readable message from the provider
                    # before falling back to a raw HTTP error.
                    try:
                        err_body = resp.json()
                        err_msg = (
                            err_body.get("error", {}).get("message")
                            or err_body.get("message")
                            or resp.text[:400]
                        )
                    except Exception:
                        err_msg = resp.text[:400]
                    raise RuntimeError(
                        f"LLM API error {resp.status_code} (model={model_id!r}): {err_msg}"
                    )
                return resp.json()

        data = await _post_with_retry()
        if is_cloud_provider(provider):
            return parse_cloud_response(provider, data)
        return parse_response(provider, data)
