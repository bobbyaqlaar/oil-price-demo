"""Quick model ID test — run from oil-price-demo/ with .env sourced."""

import asyncio
import os
import sys
import httpx


async def test(model_id: str, api_key: str) -> None:
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
    await test("claude-haiku-4-5-20251001", key)
    print()
    await test("claude-sonnet-4-6", key)


asyncio.run(main())
