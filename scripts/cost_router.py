"""
cost_router.py — Prompt complexity analyser → cheapest capable model selector.

Routing logic:
  1. Token count (via tiktoken)
  2. Semantic keyword analysis
  3. Network availability (via network_watchdog)

Route table (overridable via env vars):
  AGENT_MODEL_ARCHITECT   default: claude-sonnet-4-6
  AGENT_MODEL_COMPLEX     default: gpt-4o
  AGENT_MODEL_STANDARD    default: llama-3.3-70b-versatile (Groq / Ollama)
  AGENT_MODEL_FAST        default: gemma2 (Ollama local)
  AGENT_MODEL_LOCAL       default: llama3 (Ollama fallback)

Escalation policy: only escalate to a frontier model after two consecutive
failures on the cheaper tier.
"""

from __future__ import annotations

import os
from typing import Optional

# ── Model config ──────────────────────────────────────────────────────────────

MODEL_ARCHITECT = os.environ.get("AGENT_MODEL_ARCHITECT", "claude-sonnet-4-6")
MODEL_COMPLEX = os.environ.get("AGENT_MODEL_COMPLEX", "gpt-4o")
MODEL_STANDARD = os.environ.get(
    "AGENT_MODEL_STANDARD", "llama-3.3-70b-versatile"
)  # Groq id
MODEL_FAST = os.environ.get("AGENT_MODEL_FAST", "gemma2")  # Ollama

# GitHub Models (https://docs.github.com/en/github-models) — free-tier
# OpenAI-compatible inference using a GitHub token instead of a billed
# OPENAI_API_KEY. GITHUB_TOKEN is the automatically-provided token in
# every GitHub Actions run (no extra secret needed there); GITHUB_MODELS_TOKEN
# is the override for local dev (e.g. `export GITHUB_MODELS_TOKEN=$(gh auth token)`).
GITHUB_MODELS_TOKEN = os.environ.get("GITHUB_MODELS_TOKEN") or os.environ.get(
    "GITHUB_TOKEN", ""
)
MODEL_LOCAL = os.environ.get("AGENT_MODEL_LOCAL", "llama3")  # Ollama fallback

# Token thresholds
TOKEN_TIER_HIGH = 8_000
TOKEN_TIER_MEDIUM = 3_000

# Keywords that force a higher-capability model regardless of token count
ARCHITECT_KEYWORDS: list[str] = [
    "architect",
    "system design",
    "rfc",
    "design decision",
    "migration",
    "race condition",
    "security",
    "cryptography",
    "ast",
    "parser",
    "distributed",
]

COMPLEX_KEYWORDS: list[str] = [
    "refactor",
    "optimise",
    "optimize",
    "performance",
    "concurrency",
    "async",
    "deadlock",
    "memory leak",
    "dependency injection",
    "interface design",
]

# ── Token counter ─────────────────────────────────────────────────────────────


def _count_tokens(text: str) -> int:
    """Use tiktoken if available; fall back to word-count heuristic."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text.split()) * 4 // 3)


# ── Keyword scorer ────────────────────────────────────────────────────────────


def _keyword_tier(prompt: str) -> Optional[str]:
    lower = prompt.lower()
    for kw in ARCHITECT_KEYWORDS:
        if kw in lower:
            return "architect"
    for kw in COMPLEX_KEYWORDS:
        if kw in lower:
            return "complex"
    return None


# ── Failure tracker (session-scoped) ─────────────────────────────────────────

_consecutive_failures: dict[str, int] = {}

# Module-level dict keyed by model name, only shrinks via record_success for
# individual models — fine for this file's actual usage (dev-mode, one
# process per session, a handful of model names), but unbounded if ever used
# in a long-running process with many distinct/dynamic model ids
# (FIXES_AND_CLEANUP.md 4.4). This is a cheap upper bound, not an LRU — if it
# ever fires, dropping the whole dict just means the escalation counters
# reset to 0, which is the same as every model's first call ever.
_MAX_TRACKED_MODELS = 256


def record_failure(model: str) -> int:
    """Increment failure counter for model. Returns new count."""
    if (
        len(_consecutive_failures) >= _MAX_TRACKED_MODELS
        and model not in _consecutive_failures
    ):
        _consecutive_failures.clear()
    _consecutive_failures[model] = _consecutive_failures.get(model, 0) + 1
    return _consecutive_failures[model]


def record_success(model: str) -> None:
    """Reset failure counter after a successful call."""
    _consecutive_failures.pop(model, None)


def _should_escalate(model: str) -> bool:
    return _consecutive_failures.get(model, 0) >= 2


# ── Main router ───────────────────────────────────────────────────────────────


class ModelRoute:
    """
    Holds the chosen model and all parameters needed to invoke the LLM API.

    Attributes:
        model:      Model identifier string.
        base_url:   API base URL.
        api_key:    API key (may be empty for Ollama).
        tier:       "architect" | "complex" | "standard" | "fast" | "local"
        is_local:   True if routing to a local Ollama model.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        tier: str,
        is_local: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.tier = tier
        self.is_local = is_local

    def __repr__(self) -> str:
        return f"<ModelRoute model={self.model!r} tier={self.tier!r} local={self.is_local}>"


def route(
    prompt: str,
    task_type: Optional[str] = None,
    force_local: bool = False,
) -> ModelRoute:
    """
    Analyse prompt and return the cheapest capable ModelRoute.

    Args:
        prompt:     The full prompt string (system + user).
        task_type:  Optional hint: "architect" | "code" | "format" | "review".
        force_local: Override to always route to local Ollama.
    """
    # Check network availability
    try:
        from network_watchdog import is_online

        online = is_online() and not force_local
    except Exception:
        online = False

    if not online:
        return _local_route()

    token_count = _count_tokens(prompt)
    kw_tier = _keyword_tier(prompt)

    # Explicit task type overrides
    if task_type == "architect":
        tier = "architect"
    elif task_type == "format":
        tier = "fast"
    elif task_type == "code":
        tier = "standard"
    elif kw_tier:
        tier = kw_tier
    elif token_count > TOKEN_TIER_HIGH:
        tier = "architect"
    elif token_count > TOKEN_TIER_MEDIUM:
        tier = "complex"
    else:
        tier = "standard"

    # Escalation: if the standard/fast model has failed twice, bump up
    if tier == "standard" and _should_escalate(MODEL_STANDARD):
        tier = "complex"
    if tier == "fast" and _should_escalate(MODEL_FAST):
        tier = "standard"

    return _build_cloud_route(tier)


def _build_cloud_route(tier: str) -> ModelRoute:
    groq_key = os.environ.get("GROQ_API_KEY", "")

    if tier == "architect":
        # Provider inferred from AGENT_MODEL_ARCHITECT's actual value (via
        # _route_for_model) rather than hardcoded to Anthropic — previously
        # this tier always posted to api.anthropic.com regardless of what
        # AGENT_MODEL_ARCHITECT was set to, so overriding that env var to a
        # non-Anthropic model id silently sent it to the wrong host with
        # the wrong key. route.tier is relabelled "architect" below since
        # _route_for_model's own generic "forced" tier label is for the
        # force_model param's callers (eval_judge.py), not this one.
        r = _route_for_model(MODEL_ARCHITECT)
        r.tier = "architect"
        return r
    elif tier == "complex":
        r = _route_for_model(MODEL_COMPLEX)
        r.tier = "complex"
        return r
    elif tier == "standard":
        if groq_key:
            return ModelRoute(
                model=MODEL_STANDARD,
                base_url="https://api.groq.com/openai/v1",
                api_key=groq_key,
                tier="standard",
            )
        # Fallback to local Ollama
        return _local_route(model=MODEL_LOCAL)
    else:  # fast
        return _local_route(model=MODEL_FAST)


def _local_route(model: Optional[str] = None) -> ModelRoute:
    return ModelRoute(
        model=model or MODEL_LOCAL,
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key="ollama",
        tier="local",
        is_local=True,
    )


def _route_for_model(model: str) -> ModelRoute:
    """Build a route for an EXACT model id, bypassing route()'s complexity
    heuristics entirely — for callers (e.g. eval_judge.py's judge model)
    that need a specific, caller-chosen model rather than "whichever tier
    this prompt's length/keywords land on." Provider is inferred from the
    model id's naming convention, same substring-based approach
    infer_provider() already uses elsewhere in this codebase for base_url
    strings."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    groq_key = os.environ.get("GROQ_API_KEY", "")
    lower = model.lower()

    if "claude" in lower:
        return ModelRoute(
            model=model,
            base_url="https://api.anthropic.com/v1",
            api_key=anthropic_key,
            tier="forced",
        )
    if "gpt" in lower or lower.startswith("o1") or lower.startswith("o3"):
        if GITHUB_MODELS_TOKEN:
            # Free-tier, no OpenAI billing required — prefer this over a
            # possibly-unfunded OPENAI_API_KEY. GitHub Models namespaces
            # OpenAI model ids under "openai/" (confirmed against the live
            # API: bare "gpt-4o" 404s, "openai/gpt-4o" succeeds).
            gh_model = model if "/" in model else f"openai/{model}"
            return ModelRoute(
                model=gh_model,
                base_url="https://models.github.ai/inference",
                api_key=GITHUB_MODELS_TOKEN,
                tier="forced",
            )
        return ModelRoute(
            model=model,
            base_url="https://api.openai.com/v1",
            api_key=openai_key,
            tier="forced",
        )
    if groq_key and ("llama" in lower or "mixtral" in lower or "gemma" in lower):
        return ModelRoute(
            model=model,
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key,
            tier="forced",
        )
    return _local_route(model=model)


# ── Convenience: call via OpenAI-compatible API ───────────────────────────────


def call(
    prompt: str,
    system: str = "",
    task_type: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    force_model: Optional[str] = None,
) -> str:
    """
    Route and invoke the model. Returns the response text.
    Records token usage for circuit breaker.

    force_model: bypass route()'s complexity-tier heuristics and use this
    exact model id (e.g. a configured eval judge model that must not be
    silently swapped for whatever tier the prompt's length/keywords land
    on — see eval_judge.py's run_judge()).
    """
    route_result = (
        _route_for_model(force_model)
        if force_model
        else route(prompt, task_type=task_type)
    )

    # Build messages
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        import httpx

        try:
            from runtime.provider_dispatch import (
                build_request,
                infer_provider,
                parse_response,
            )
        except ImportError:
            import sys as _sys
            from pathlib import Path as _Path

            _sys.path.insert(
                0, str(_Path(__file__).resolve().parent.parent / "runtime")
            )
            from provider_dispatch import build_request, infer_provider, parse_response  # type: ignore

        # Request building / response parsing shared with
        # runtime/llm_gateway.py via runtime/provider_dispatch.py — this
        # used to independently re-derive "is this Anthropic" from the
        # base_url string and build/parse bodies inline, drifting from
        # llm_gateway.py's own copy of the same logic (FIXES_AND_CLEANUP.md 4.3).
        provider = infer_provider(route_result.base_url)
        path_suffix, headers, body = build_request(
            provider,
            route_result.model,
            messages,
            route_result.api_key,
            max_tokens,
            temperature,
        )
        # cost_router's Anthropic base_url has no /v1 segment (unlike
        # llm_gateway.py's), so its messages endpoint is base_url + "/messages",
        # not base_url + "/v1/messages" — preserve that pre-existing URL shape
        # exactly rather than switching it to provider_dispatch's path.
        url = route_result.base_url.rstrip("/") + (
            "/messages" if provider == "anthropic" else path_suffix
        )

        resp = httpx.post(url, json=body, headers=headers, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()

        text, in_tok, out_tok = parse_response(provider, data)

        # Record token usage for circuit breaker
        try:
            from circuit_breaker import audit_token_velocity_circuit

            audit_token_velocity_circuit(in_tok, out_tok)
        except Exception:  # fail-open: circuit breaker is a side-effect check after a successful call; the call's own errors are handled by the outer except below, not this one
            pass

        record_success(route_result.model)
        return text

    except Exception as exc:
        record_failure(route_result.model)
        raise RuntimeError(
            f"LLM call failed [{route_result.tier} / {route_result.model}]: {exc}"
        ) from exc


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json as _json

    prompt = " ".join(sys.argv[1:]) or "Write a hello world function in Python."
    r = route(prompt)
    print(
        _json.dumps(
            {
                "model": r.model,
                "tier": r.tier,
                "base_url": r.base_url,
                "is_local": r.is_local,
                "estimated_tokens": _count_tokens(prompt),
            },
            indent=2,
        )
    )
