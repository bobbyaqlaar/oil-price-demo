"""
runtime/trace_redactor.py — Environment-aware OTLP span scrubbing.

Acts as an OpenTelemetry SpanProcessor. Intercepts spans before export
and applies the active redaction profile based on $ENVIRONMENT.

Redaction profiles (see SPECS.md §27):
  development  — full capture (up to 1,000 chars)
  staging      — PII/secret patterns stripped; structure preserved; hashed identifiers
  production   — minimal: hashed/truncated to 50 chars; full payload in encrypted HITL blob only

Usage:
    from runtime.trace_redactor import TraceRedactor
    provider = TracerProvider()
    provider.add_span_processor(TraceRedactor())   # reads ENVIRONMENT from env

Note on mutating spans in on_end(): the OTel SDK's `ReadableSpan` exposes
attributes as a read-only mapping by contract, but the only point at which a
processor can intercept a span before export is `on_end`. This processor
mutates the span's internal `_attributes` dict directly — the standard
workaround used by redaction/scrubbing processors in the OTel Python
ecosystem, since there is no public "rewrite before export" hook.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor as _OTelSpanProcessor
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False
    _OTelSpanProcessor = object  # type: ignore

try:
    # Normal case: repo root on sys.path, runtime/ is a package (has __init__.py).
    from runtime.environment import get_environment
except ImportError:
    # scripts/verify_system.py imports this module with only runtime/ itself
    # (not its parent) on sys.path — fall back to the flat sibling import.
    from environment import get_environment  # type: ignore

# The span attribute set by runtime/llm_gateway.py and the agent scripts on
# every span — the authoritative per-span tenant identity.
_TENANT_ATTRIBUTE = "tenant.id"


# ── Default secret/PII pattern library (§27) ──────────────────────────────────

_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9\-]{20,}"),                       # Anthropic API keys
    re.compile(r"sk-[A-Za-z0-9]{20,}"),                             # OpenAI API keys
    re.compile(r"Bearer\s+[A-Za-z0-9\-_.~+/]+=*"),                  # Bearer tokens
    re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),  # Email addresses
]

# Candidate credit-card-shaped digit runs; validated with Luhn before redaction.
_CARD_CANDIDATE = re.compile(r"(?:\d[ -]?){13,19}")

# Disabled by default in staging per §27 ("optional — disabled by default in staging").
_IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

_REDACTED_MARKER = "[REDACTED]"

# Attributes that carry free-text payloads worth redacting/truncating.
_PAYLOAD_ATTRIBUTES = {"input.value", "output.value"}


def _luhn_valid(digits: str) -> bool:
    digits = re.sub(r"[ -]", "", digits)
    if not digits.isdigit() or not (13 <= len(digits) <= 19):
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _redact_credit_cards(text: str) -> str:
    def _sub(match: "re.Match") -> str:
        return _REDACTED_MARKER if _luhn_valid(match.group(0)) else match.group(0)
    return _CARD_CANDIDATE.sub(_sub, text)


def _hash8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:8]


def _load_extra_patterns() -> list:
    """Tenant repos extend the pattern library via .agenticframework/redaction-patterns.yaml (§27)."""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / ".agenticframework" / "redaction-patterns.yaml"
        if candidate.exists():
            try:
                import yaml  # type: ignore
                data = yaml.safe_load(candidate.read_text()) or {}
                return [re.compile(p) for p in data.get("patterns", [])]
            except Exception:
                return []
        if (parent / ".git").exists():
            break
    return []


# ── Encrypted HITL blob storage (§27) ─────────────────────────────────────────

class HITLBlobStore:
    """
    Stores the full, unredacted payload for production spans flagged for HITL
    review. Encrypted with AES-256-GCM using a per-tenant key.

    Storage backend: local filesystem by default (HITL_BLOB_DIR, default
    runtime/.hitl_blobs/); set HITL_BLOB_S3_BUCKET to write to S3 instead.

    TTL is recorded in blob metadata (default 90 days, §27) — actual expiry
    is enforced by an external lifecycle job (S3 lifecycle rule, or a cron
    that purges local blobs older than their recorded TTL), not by this class.
    """

    def __init__(self, tenant_id: str) -> None:
        self.tenant_id = tenant_id

    def _key(self) -> bytes:
        raw = (
            os.environ.get(f"HITL_ENCRYPTION_KEY_{self.tenant_id.upper()}")
            or os.environ.get("HITL_ENCRYPTION_KEY", "")
        )
        if not raw:
            raise RuntimeError(
                f"No HITL encryption key configured for tenant={self.tenant_id!r}. "
                f"Set HITL_ENCRYPTION_KEY_{self.tenant_id.upper()} or HITL_ENCRYPTION_KEY."
            )
        # Derive a 32-byte key regardless of input length/encoding.
        return hashlib.sha256(raw.encode("utf-8")).digest()

    def put(self, ref: str, plaintext: str, ttl_days: int = 90) -> str:
        """Encrypt and store plaintext under ref. Returns the blob reference.

        Raises RuntimeError immediately for configuration errors (missing
        encryption key) — silently swallowing that case used to leave a span
        with an `hitl_blob_ref` pointing at a blob that was never written,
        defeating the "full payload preserved for compliance review"
        guarantee with zero error or alert (FIXES_AND_CLEANUP.md 2.3).
        Storage I/O errors (disk full, S3 unreachable) are logged at ERROR
        and swallowed — a transient storage outage still shouldn't break
        trace export, but it must not be invisible either.
        """
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        import os as _os
        import json
        import time

        key = self._key()  # raises RuntimeError if unconfigured — let it propagate
        nonce = _os.urandom(12)
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)

        bucket = os.environ.get("HITL_BLOB_S3_BUCKET")
        blob = {
            "nonce": nonce.hex(),
            "ciphertext": ciphertext.hex(),
            "tenant_id": self.tenant_id,
            "created_at": time.time(),
            "ttl_days": ttl_days,
        }
        try:
            if bucket:
                import boto3  # type: ignore
                s3 = boto3.client("s3")
                s3.put_object(Bucket=bucket, Key=f"hitl/{self.tenant_id}/{ref}.json", Body=json.dumps(blob))
            else:
                blob_dir = Path(os.environ.get("HITL_BLOB_DIR", str(Path(__file__).resolve().parent / ".hitl_blobs")))
                blob_dir = blob_dir / self.tenant_id
                blob_dir.mkdir(parents=True, exist_ok=True)
                (blob_dir / f"{ref}.json").write_text(json.dumps(blob))
            return ref
        except OSError as exc:
            logger.error("HITL blob persistence failed for tenant=%s ref=%s: %s", self.tenant_id, ref, exc)
            return ref
        except Exception as exc:  # e.g. boto3 ClientError — not importable to catch by name unconditionally
            logger.error("HITL blob persistence failed for tenant=%s ref=%s: %s", self.tenant_id, ref, exc)
            return ref


def _make_blob_ref(trace_id: str, span_id: str, attr_key: str) -> str:
    # span_id is required: a single trace with multiple independently
    # HITL-flagged sibling spans (e.g. Architect/Developer/Validator) would
    # otherwise all compute the same ref `{trace_id}.{attr_key}` and the last
    # write wins, permanently overwriting the earlier spans' encrypted
    # payloads before anyone reviews them (FIXES_AND_CLEANUP.md 2.2).
    return f"{trace_id}.{span_id}.{attr_key}"


# ── Span processor ────────────────────────────────────────────────────────────

class TraceRedactor(_OTelSpanProcessor):
    """
    SpanProcessor that scrubs sensitive data before export.

    Inherits from opentelemetry.sdk.trace.SpanProcessor (when available) so
    that newer SDK lifecycle hooks (e.g. `_on_ending`, added after this
    interface was first stabilised) get a safe no-op default instead of
    raising AttributeError when the SDK's TracerProvider calls them.

    profile:
      "none"       — no scrubbing (development)
      "staging"    — strip patterns; preserve structure; hash flagged identifiers
      "production" — strip patterns, truncate to 50 chars; full payload stashed
                      in an encrypted HITL blob
    """

    def __init__(self, profile: Optional[str] = None, tenant_id: Optional[str] = None) -> None:
        env = profile or get_environment()
        self.profile = {
            "development": "none",
            "staging": "staging",
            "production": "production",
        }.get(env, "production")  # fail closed: unrecognized -> strictest profile
        # Fallback only — used when a span carries no tenant.id attribute at
        # all. The authoritative source is per-span, resolved in on_end()
        # below; binding tenant_id once here (at __init__/process-construction
        # time) was the actual cross-tenant leak: on a shared worker pool
        # processing spans for multiple tenants in one process, every
        # HITL-flagged span got encrypted with whichever tenant's key the
        # processor happened to be constructed with (FIXES_AND_CLEANUP.md 1.2).
        self.default_tenant_id = tenant_id or os.environ.get("TENANT_ID", "unknown")
        self.enable_ip_redaction = os.environ.get("ENABLE_IP_REDACTION", "false").lower() == "true"
        self._extra_patterns = _load_extra_patterns()
        self._blob_stores: dict[str, HITLBlobStore] = {}

    def _blob_store_for(self, tenant_id: str) -> HITLBlobStore:
        store = self._blob_stores.get(tenant_id)
        if store is None:
            store = HITLBlobStore(tenant_id)
            self._blob_stores[tenant_id] = store
        return store

    def on_start(self, span: "Span", parent_context=None) -> None:  # type: ignore[override]
        pass  # No action on start — scrubbing happens once the span's final attributes are known.

    def on_end(self, span: "ReadableSpan") -> None:  # type: ignore[override]
        if self.profile == "none" or not _HAS_OTEL:
            return

        attributes = getattr(span, "_attributes", None)
        if not attributes:
            return

        trace_id = format(span.context.trace_id, "032x") if getattr(span, "context", None) else "unknown"
        span_id = format(span.context.span_id, "016x") if getattr(span, "context", None) else "unknown"
        tenant_id = attributes.get(_TENANT_ATTRIBUTE) or self.default_tenant_id

        for key, value in list(attributes.items()):
            if not isinstance(value, str):
                continue

            if self.profile == "staging":
                attributes[key] = self._scrub(value, hash_identifiers=True)
            elif self.profile == "production":
                scrubbed = self._scrub(value, hash_identifiers=False)
                if key in _PAYLOAD_ATTRIBUTES and len(scrubbed) > 50:
                    ref = _make_blob_ref(trace_id, span_id, key)
                    try:
                        self._blob_store_for(tenant_id).put(ref, value)
                        attributes[f"{key}.hitl_blob_ref"] = ref
                    except RuntimeError as exc:
                        # Missing HITL_ENCRYPTION_KEY for this tenant — log
                        # loudly rather than silently truncating the payload
                        # with a dangling blob ref nothing ever wrote to.
                        logger.error(
                            "HITL blob NOT written for tenant=%s ref=%s: %s — payload truncated without compliance backup.",
                            tenant_id, ref, exc,
                        )
                attributes[key] = self._truncate(scrubbed, max_chars=50)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _scrub(self, text: str, hash_identifiers: bool) -> str:
        """Apply the pattern library to a string. hash_identifiers=True replaces
        matches with a short hash (staging — structure preserved, identifiers
        recoverable for correlation); False replaces with a flat marker
        (production — no information retained outside the encrypted blob)."""
        marker_fn = (lambda m: f"[REDACTED:{_hash8(m.group(0))}]") if hash_identifiers else (lambda m: _REDACTED_MARKER)

        for pattern in (*_SECRET_PATTERNS, *self._extra_patterns):
            text = pattern.sub(marker_fn, text)

        text = _redact_credit_cards(text) if not hash_identifiers else _CARD_CANDIDATE.sub(
            lambda m: marker_fn(m) if _luhn_valid(m.group(0)) else m.group(0), text
        )

        if self.enable_ip_redaction:
            text = _IP_PATTERN.sub(marker_fn, text)

        return text

    def _truncate(self, text: str, max_chars: int = 50) -> str:
        if len(text) > max_chars:
            return text[:max_chars] + "…[truncated]"
        return text
