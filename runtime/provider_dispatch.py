"""
runtime/provider_dispatch.py — shared provider request building and response
parsing for runtime/llm_gateway.py and scripts/cost_router.py.

Before this module existed, llm_gateway.py and cost_router.py each
independently built provider request bodies/headers and parsed responses,
with no shared code (FIXES_AND_CLEANUP.md 4.3). Only the provider-dispatch
shape is shared here; each caller's own routing/budget/degrade-ladder logic
stays in its own file.

Two tiers of providers:

1. Direct-API providers (`anthropic`, `openai_compatible`) — same host every
   time, static API-key auth. `build_request`/`parse_response` below, used
   by both llm_gateway.py and cost_router.py. llm_gateway.py resolves the
   base_url/api_key itself (optionally overridden per-model via
   `models.yaml`'s `endpoint`/`api_key_env` fields) and prepends it to the
   path these return.

2. Cloud-native providers (`vertex_ai`, `azure_openai`, `bedrock`,
   `huawei_modelarts`) — each needs its own auth scheme (OAuth2 service
   account, api-key + api-version, SigV4, AK/SK signing) and its own
   request/response envelope, not just a different host. These implement
   the `CloudProviderAdapter` protocol below and are looked up via
   `get_cloud_adapter(provider)`. `build_cloud_request`/`parse_cloud_response`
   are the entry points llm_gateway.py calls — each returns/accepts a full
   URL, not just a path, since cloud providers bake project/region/
   deployment/endpoint-id into the URL itself.

A flat `if/elif` per cloud provider was considered and rejected: each cloud
provider's auth, URL shape, and envelope are genuinely independent axes (you
can't share an "is it Bearer-token-shaped" branch across SigV4 and OAuth2),
so a one-class-per-provider Protocol keeps each provider's quirks contained
and makes adding a fifth provider additive rather than another branch
threaded through three concerns at once.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Protocol


# ── Direct-API providers (anthropic / openai-compatible) ──────────────────────


def infer_provider(base_url: str) -> str:
    """cost_router.py only has a base_url (no separate provider field) — this
    mirrors its existing "anthropic" in base_url check."""
    return "anthropic" if "anthropic" in base_url else "openai_compatible"


def build_request(
    provider: str,
    model_id: str,
    messages: list[dict],
    api_key: str,
    max_tokens: int,
    temperature: float = 0.2,
) -> tuple[str, dict, dict]:
    """Returns (url_path, headers, body) for the given provider.

    provider="anthropic" uses the Messages API shape (system pulled out of
    the messages list into its own top-level field); anything else is
    treated as OpenAI-compatible (openai, groq, ollama, ...).
    """
    if provider == "anthropic":
        system = (
            "\n".join(m["content"] for m in messages if m["role"] == "system") or None
        )
        user_messages = [m for m in messages if m["role"] != "system"]
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": user_messages,
        }
        if system:
            body["system"] = system
        return "/v1/messages", headers, body

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    return "/chat/completions", headers, body


def parse_response(provider: str, data: dict) -> tuple[str, int, int]:
    """Returns (text, input_tokens, output_tokens)."""
    if provider == "anthropic":
        text = data["content"][0]["text"]
        usage = data.get("usage", {})
        return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)

    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


# ── Cloud-native providers (Vertex AI / Azure OpenAI / Bedrock / Huawei ModelArts) ──


class CloudProviderAdapter(Protocol):
    """One implementation per cloud-hosted provider. Each owns its own
    credential acquisition, URL templating, and request/response envelope —
    the three things that differ across SigV4 / OAuth2 / api-key+api-version
    / AK-SK auth and Bedrock/Vertex/Azure/ModelArts envelopes."""

    def build_request(
        self,
        model_id: str,
        messages: list[dict],
        cfg: dict,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, dict, dict]:
        """Returns (full_url, headers, body)."""
        ...

    def parse_response(self, data: dict) -> tuple[str, int, int]:
        """Returns (text, input_tokens, output_tokens)."""
        ...


def _anthropic_messages_body(
    model_id: str, messages: list[dict], max_tokens: int
) -> dict:
    system = "\n".join(m["content"] for m in messages if m["role"] == "system") or None
    user_messages = [m for m in messages if m["role"] != "system"]
    body: dict[str, Any] = {
        "anthropic_version": "vertex-2023-10-16",
        "max_tokens": max_tokens,
        "messages": user_messages,
    }
    if system:
        body["system"] = system
    return body


class VertexAIAdapter:
    """GCP Vertex AI — Gemini-on-Vertex (`publishers/google/models/...
    :generateContent`) is the default and recommended path: it's Google's
    own first-party model on its own platform, so it doesn't carry the
    region/availability uncertainty of a third-party model (e.g. Claude)
    being hosted on someone else's cloud, which rolls out per-region on its
    own schedule independent of the underlying Vertex AI region's GA status.
    Anthropic-on-Vertex (`publishers/anthropic/models/...:streamRawPredict`)
    is still supported (set `publisher: anthropic`) but deprioritized as
    the default for this provider — use it only if you specifically need
    Claude and have confirmed it's enabled for your project's region.

    Auth: OAuth2 service-account token via `google-auth`. Required
    models.yaml fields: `project`, `region` (defaults to `us-central1`),
    `publisher` (defaults to `google`, i.e. Gemini; set to `anthropic` for
    Claude-on-Vertex). Credentials resolved the standard google-auth way:
    `GOOGLE_APPLICATION_CREDENTIALS` env var pointing at a service-account
    JSON key, or any other `google.auth.default()`-supported source
    (workload identity, gcloud ADC, etc).

    Region note (verified live against a real Vertex AI project): GCP's
    GCC regions — `me-central1` (Doha, Qatar) and `me-central2` (Dammam,
    Saudi Arabia) — do NOT currently serve `gemini-2.5-flash` (confirmed
    404 "Publisher model ... was not found" on both). `us-central1`,
    `europe-west1`, `europe-west4`, and `asia-south1` were confirmed
    working. If GCC-region hosting is a hard requirement, verify model
    availability for that region first via a live call before overriding
    `region` — do not assume any specific GCC region serves a given model.
    """

    _cached_token: str | None = None
    _cached_token_expiry: float = 0.0

    def _get_access_token(self) -> str:
        if self._cached_token and time.time() < self._cached_token_expiry - 60:
            return self._cached_token
        import google.auth
        import google.auth.transport.requests

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(google.auth.transport.requests.Request())
        VertexAIAdapter._cached_token = credentials.token
        VertexAIAdapter._cached_token_expiry = (
            credentials.expiry.timestamp() if credentials.expiry else time.time() + 3600
        )
        return self._cached_token

    _DEFAULT_URL_TEMPLATE_ANTHROPIC = (
        "https://{region}-aiplatform.googleapis.com/v1/projects/{project}"
        "/locations/{region}/publishers/anthropic/models/{model_id}:streamRawPredict"
    )
    _DEFAULT_URL_TEMPLATE_GENERIC = (
        "https://{region}-aiplatform.googleapis.com/v1/projects/{project}"
        "/locations/{region}/publishers/{publisher}/models/{model_id}:generateContent"
    )

    def build_request(
        self,
        model_id: str,
        messages: list[dict],
        cfg: dict,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, dict, dict]:
        project = os.path.expandvars(
            cfg["project"]
        )  # supports ${VAR} so a project id never has to be committed literally
        region = cfg.get(
            "region", "us-central1"
        )  # verified working; GCC regions confirmed NOT to serve gemini-2.5-flash, see class docstring
        publisher = cfg.get(
            "publisher", "google"
        )  # Gemini — first-party on Vertex, no cross-vendor rollout lag
        token = self._get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # `url_template` (models.yaml) overrides the default vendor URL shape
        # entirely — needed for a regional API variant, a private VPC
        # endpoint, or a proxy in front of the Vertex AI API. Falls back to
        # the standard public endpoint when not set.
        fmt = {
            "project": project,
            "region": region,
            "publisher": publisher,
            "model_id": model_id,
        }
        if publisher == "anthropic":
            url = cfg.get("url_template", self._DEFAULT_URL_TEMPLATE_ANTHROPIC).format(
                **fmt
            )
            body = _anthropic_messages_body(model_id, messages, max_tokens)
        else:
            url = cfg.get("url_template", self._DEFAULT_URL_TEMPLATE_GENERIC).format(
                **fmt
            )
            body = {
                "contents": [
                    {
                        "role": "user" if m["role"] != "assistant" else "model",
                        "parts": [{"text": m["content"]}],
                    }
                    for m in messages
                    if m["role"] != "system"
                ],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature,
                },
            }
        return url, headers, body

    def parse_response(self, data: dict) -> tuple[str, int, int]:
        if (
            "content" in data
        ):  # Anthropic-on-Vertex envelope mirrors the direct Messages API shape
            text = data["content"][0]["text"]
            usage = data.get("usage", {})
            return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        candidate = data["candidates"][0]
        text = candidate["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        return (
            text,
            usage.get("promptTokenCount", 0),
            usage.get("candidatesTokenCount", 0),
        )


class AzureOpenAIAdapter:
    """Azure OpenAI — same chat-completions body shape as direct OpenAI, but
    api-key header (not Bearer), deployment name in the URL path instead of
    a bare model id, and a required `api-version` query param.

    Required models.yaml fields: `resource` (the Azure OpenAI resource
    name), `deployment` (the deployment name — may differ from `id`),
    `api_version` (defaults to "2024-06-01"). `api_key_env` defaults to
    `AZURE_OPENAI_API_KEY`. Optional `url_template` overrides the URL shape
    entirely (e.g. for Azure Gov / sovereign cloud endpoints).
    """

    _DEFAULT_URL_TEMPLATE = (
        "https://{resource}.openai.azure.com/openai/deployments/{deployment}"
        "/chat/completions?api-version={api_version}"
    )

    def build_request(
        self,
        model_id: str,
        messages: list[dict],
        cfg: dict,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, dict, dict]:
        resource = cfg["resource"]
        deployment = cfg.get("deployment", model_id)
        api_version = cfg.get("api_version", "2024-06-01")
        api_key = os.environ.get(cfg.get("api_key_env", "AZURE_OPENAI_API_KEY"), "")

        url = cfg.get("url_template", self._DEFAULT_URL_TEMPLATE).format(
            resource=resource,
            deployment=deployment,
            api_version=api_version,
            model_id=model_id,
        )
        headers = {"api-key": api_key, "Content-Type": "application/json"}
        body = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        return url, headers, body

    def parse_response(self, data: dict) -> tuple[str, int, int]:
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


class BedrockAdapter:
    """AWS Bedrock — SigV4-signed requests (AWS credentials via boto3/
    botocore, not a bearer token), `bedrock-runtime.{region}.amazonaws.com/
    model/{model-id}/invoke` URL shape, Anthropic-on-Bedrock envelope (its
    own shape, distinct from both the direct Anthropic API and
    Anthropic-on-Vertex).

    Required models.yaml fields: `region` (defaults to `us-east-1`, one of
    Bedrock's original/best-model-coverage regions). AWS credentials
    resolved the standard boto3 way (env vars, shared config file,
    instance/task role) — no per-model credential override, since Bedrock
    access is normally scoped at the IAM-role level, not per model.
    Optional `url_template` overrides the URL shape (e.g. a VPC interface
    endpoint instead of the public `bedrock-runtime` host) — the override
    still gets correctly SigV4-signed since signing happens against
    whatever URL is built, not a hard-coded host.

    GCC region note: AWS's GCC region is `me-central-1` (UAE/Dubai);
    `me-south-1` (Bahrain) is the other Middle East option. Neither has
    been verified to have Bedrock (or the specific foundation model you
    need) enabled — the live test that disproved the equivalent assumption
    for GCP Vertex AI's GCC regions (see VertexAIAdapter docstring) was not
    repeatable here for lack of AWS credentials in the test environment.
    Confirm via a live call before overriding `region` to a GCC value;
    do not assume it works by default.
    """

    _DEFAULT_URL_TEMPLATE = (
        "https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/invoke"
    )

    def build_request(
        self,
        model_id: str,
        messages: list[dict],
        cfg: dict,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, dict, dict]:
        import boto3
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        region = cfg.get(
            "region", "us-east-1"
        )  # broad model coverage; GCC region unverified, see class docstring
        url = cfg.get("url_template", self._DEFAULT_URL_TEMPLATE).format(
            region=region, model_id=model_id
        )

        system = (
            "\n".join(m["content"] for m in messages if m["role"] == "system") or None
        )
        user_messages = [m for m in messages if m["role"] != "system"]
        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": user_messages,
        }
        if system:
            body["system"] = system
        body_bytes = json.dumps(body).encode()

        session = boto3.Session()
        credentials = session.get_credentials()
        request = AWSRequest(
            method="POST",
            url=url,
            data=body_bytes,
            headers={"Content-Type": "application/json"},
        )
        SigV4Auth(credentials, "bedrock", region).add_auth(request)
        headers = dict(request.headers)
        return url, headers, body

    def parse_response(self, data: dict) -> tuple[str, int, int]:
        text = data["content"][0]["text"]
        usage = data.get("usage", {})
        return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)


class HuaweiModelArtsAdapter:
    """Huawei Cloud ModelArts inference — AK/SK request signing (Huawei's
    own "SDK-HMAC-SHA256" scheme, conceptually similar to AWS SigV4 but a
    distinct canonical-request format), hitting a per-deployment custom
    inference endpoint domain.

    NOTE: this is the least-documented of the four cloud providers in
    English-language sources. The endpoint/signing shape below follows
    Huawei's published API Gateway signing algorithm structure but has not
    been validated against a live ModelArts inference endpoint — treat as
    a starting point to verify against current Huawei API docs and a real
    deployment, not as confirmed-correct.

    Required models.yaml fields: `endpoint` (the full custom inference
    endpoint host, e.g. `xxxxx.{region}.modelarts-infer.com`), `region`.
    Credentials: `HUAWEICLOUD_SDK_AK` / `HUAWEICLOUD_SDK_SK` env vars
    (access key / secret key). Optional `path_template` overrides the
    request path (e.g. a differently-versioned or custom inference route)
    — `endpoint` always supplies the host, since the signature covers the
    `Host` header and path together and they must agree.
    """

    _DEFAULT_PATH_TEMPLATE = "/v1/infers/{model_id}/chat/completions"

    def build_request(
        self,
        model_id: str,
        messages: list[dict],
        cfg: dict,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, dict, dict]:
        import hashlib
        import hmac
        from datetime import datetime, timezone

        endpoint = cfg["endpoint"]
        path = cfg.get("path_template", self._DEFAULT_PATH_TEMPLATE).format(
            model_id=model_id
        )
        url = f"https://{endpoint}{path}"
        body = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        body_bytes = json.dumps(body).encode()

        ak = os.environ.get("HUAWEICLOUD_SDK_AK", "")
        sk = os.environ.get("HUAWEICLOUD_SDK_SK", "")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        content_sha256 = hashlib.sha256(body_bytes).hexdigest()
        canonical_request = "\n".join(
            [
                "POST",
                path,
                "",
                f"content-type:application/json\nhost:{endpoint}\nx-sdk-date:{timestamp}\n",
                "content-type;host;x-sdk-date",
                content_sha256,
            ]
        )
        string_to_sign = "\n".join(
            [
                "SDK-HMAC-SHA256",
                timestamp,
                hashlib.sha256(canonical_request.encode()).hexdigest(),
            ]
        )
        signature = hmac.new(
            sk.encode(), string_to_sign.encode(), hashlib.sha256
        ).hexdigest()
        headers = {
            "Content-Type": "application/json",
            "X-Sdk-Date": timestamp,
            "Authorization": (
                f"SDK-HMAC-SHA256 Access={ak}, "
                f"SignedHeaders=content-type;host;x-sdk-date, Signature={signature}"
            ),
            "Host": endpoint,
        }
        return url, headers, body

    def parse_response(self, data: dict) -> tuple[str, int, int]:
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


_CLOUD_ADAPTERS: dict[str, CloudProviderAdapter] = {
    "vertex_ai": VertexAIAdapter(),
    "azure_openai": AzureOpenAIAdapter(),
    "bedrock": BedrockAdapter(),
    "huawei_modelarts": HuaweiModelArtsAdapter(),
}


def is_cloud_provider(provider: str) -> bool:
    return provider in _CLOUD_ADAPTERS


def get_cloud_adapter(provider: str) -> CloudProviderAdapter:
    try:
        return _CLOUD_ADAPTERS[provider]
    except KeyError:
        raise ValueError(
            f"Unknown cloud provider {provider!r}. Supported: {sorted(_CLOUD_ADAPTERS)}"
        ) from None


def build_cloud_request(
    provider: str,
    model_id: str,
    messages: list[dict],
    cfg: dict,
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict, dict]:
    """Returns (full_url, headers, body) for a cloud-native provider."""
    return get_cloud_adapter(provider).build_request(
        model_id, messages, cfg, max_tokens, temperature
    )


def parse_cloud_response(provider: str, data: dict) -> tuple[str, int, int]:
    """Returns (text, input_tokens, output_tokens) for a cloud-native provider."""
    return get_cloud_adapter(provider).parse_response(data)
