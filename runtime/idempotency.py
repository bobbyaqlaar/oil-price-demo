"""
runtime/idempotency.py — Idempotency key store and deduplication.

Every LLM Gateway activity is assigned an idempotency key derived from
a hash of its input parameters. Duplicate submissions (on retry after crash)
are detected and short-circuited — the cached result is returned immediately.

Backend: Redis (default) or Postgres. Configurable via IDEMPOTENCY_BACKEND env var.

Usage:
    store = IdempotencyStore()
    result = store.get("sha256:abc123")
    if result is not None:
        return result  # cached
    # ... do the work ...
    store.set("sha256:abc123", result, ttl_seconds=86400)

See SPECS.md §25 for the full specification.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Optional


def make_key(payload: Any) -> str:
    """Derive a stable idempotency key from any JSON-serialisable payload."""
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


class IdempotencyStore:
    """
    Idempotency key store.

    Instantiate once per worker process; share across gateway calls.
    """

    def __init__(self) -> None:
        backend = os.environ.get("IDEMPOTENCY_BACKEND", "redis").lower()
        if backend == "redis":
            self._backend: Any = _RedisBackend()
        elif backend == "postgres":
            self._backend = _PostgresBackend()
        else:
            raise ValueError(f"Unknown IDEMPOTENCY_BACKEND={backend!r}. Use 'redis' or 'postgres'.")

    def get(self, key: str) -> Optional[Any]:
        """Return cached result for key, or None if not found / expired."""
        return self._backend.get(key)

    def set(self, key: str, value: Any, ttl_seconds: int = 86400) -> None:
        """Store result for key with TTL."""
        self._backend.set(key, value, ttl_seconds)


class _RedisBackend:
    def __init__(self) -> None:
        import redis  # type: ignore
        self._client = redis.from_url(os.environ["REDIS_URL"])

    def get(self, key: str) -> Optional[Any]:
        raw = self._client.get(f"agenticframework:idempotency:{key}")
        if raw is None:
            return None
        return json.loads(raw)

    def set(self, key: str, value: Any, ttl_seconds: int = 86400) -> None:
        self._client.set(
            f"agenticframework:idempotency:{key}",
            json.dumps(value, default=str),
            ex=ttl_seconds,
        )


class _PostgresBackend:
    def __init__(self) -> None:
        self._dsn = os.environ["DATABASE_URL"]
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS idempotency_keys (
                        key        TEXT PRIMARY KEY,
                        value      JSONB NOT NULL,
                        expires_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
        finally:
            conn.close()

    def _connect(self):
        import psycopg2  # type: ignore
        return psycopg2.connect(self._dsn)

    def get(self, key: str) -> Optional[Any]:
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT value FROM idempotency_keys WHERE key = %s AND expires_at > now()",
                    (key,),
                )
                row = cur.fetchone()
                return row[0] if row else None
        finally:
            conn.close()

    def set(self, key: str, value: Any, ttl_seconds: int = 86400) -> None:
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                # json.dumps + cast to jsonb rather than passing the dict
                # directly: psycopg2 has no implicit Python-dict-to-jsonb
                # adapter registered by default.
                cur.execute(
                    """
                    INSERT INTO idempotency_keys (key, value, expires_at)
                    VALUES (%s, %s::jsonb, now() + (%s || ' seconds')::interval)
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        expires_at = EXCLUDED.expires_at
                    """,
                    (key, json.dumps(value, default=str), ttl_seconds),
                )
        finally:
            conn.close()

    def purge_expired(self) -> int:
        """Deletes rows past their TTL. Not called automatically — intended
        for a periodic cleanup job (cron, or scripts/verify_system.py
        --check-idempotency) since this class has no background thread."""
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute("DELETE FROM idempotency_keys WHERE expires_at <= now()")
                return cur.rowcount
        finally:
            conn.close()
