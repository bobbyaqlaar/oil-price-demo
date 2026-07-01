"""Quick model ID check — run from oil-price-demo/ with .env sourced.

Named check_model_ids.py rather than test_*.py deliberately: a name
matching pytest's discovery glob would make `pytest` import (and thereby
execute, since there's no `if __name__ == "__main__":` guard) this
module during test collection — module-level `asyncio.run(main())` would
run for real with no ANTHROPIC_API_KEY set in CI, hit `sys.exit(1)`, and
abort the whole pytest run with an INTERNALERROR rather than a normal
test failure (confirmed — this is exactly what happened before the
rename)."""

import asyncio
import os
import sys
import httpx


async def check_model(model_id: str, api_key: str) -> None:
    body = {
        "model": model_id,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Say hi"}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages", json=body, headers=headers
        )
        print(f"[{model_id}] {resp.status_code}")
        print(resp.text[:500])


async def main() -> None:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    await check_model("claude-haiku-4-5-20251001", key)
    print()
    await check_model("claude-sonnet-4-6", key)


if __name__ == "__main__":
    asyncio.run(main())
