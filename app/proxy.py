"""
Proxy layer — forwards requests to Anthropic API via httpx.

Handles both streaming (SSE) and non-streaming responses.
For streaming, intercepts the `message_start` event to extract
cache usage stats without slowing down the response stream.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx
import structlog

from app.config import settings

logger = structlog.get_logger()

# Shared client — created at import, closed in app lifespan
_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    """Get or create the shared httpx client."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=settings.anthropic_base_url,
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
    return _client


async def close_client():
    """Close the shared client (called from app lifespan)."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None


# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------

# Headers to NOT forward from the incoming request
_HOP_BY_HOP = frozenset({
    "content-length", "transfer-encoding", "connection", "upgrade",
    "host", "te", "trailer", "proxy-connection", "proxy-authorization",
})


def _extract_api_key(headers: dict[str, str]) -> str:
    """Extract Anthropic API key from request headers or fall back to config."""
    # Authorization: Bearer sk-ant-...
    auth = headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    # x-api-key header
    api_key = headers.get("x-api-key", "")
    if api_key:
        return api_key
    # Fall back to environment
    return settings.anthropic_api_key


def _build_upstream_headers(incoming_headers: dict[str, str], api_key: str) -> dict[str, str]:
    """Build headers dict for the upstream Anthropic request."""
    out = {}
    for k, v in incoming_headers.items():
        if k.lower() not in _HOP_BY_HOP:
            out[k] = v
    # Always set required Anthropic headers
    out["content-type"] = "application/json"
    out["anthropic-version"] = incoming_headers.get("anthropic-version", "2023-06-01")
    # Auth: Anthropic uses x-api-key
    out.pop("authorization", None)
    out["x-api-key"] = api_key
    return out


# ---------------------------------------------------------------------------
# Forwarding
# ---------------------------------------------------------------------------


async def forward_request(body: dict, incoming_headers: dict[str, str]) -> httpx.Response:
    """
    Forward a non-streaming request to Anthropic. Returns the complete response.
    Caller is responsible for closing the response.
    """
    api_key = _extract_api_key(incoming_headers)
    headers = _build_upstream_headers(incoming_headers, api_key)
    client = get_client()

    logger.debug("proxy.forward", model=body.get("model"), streaming=False)

    resp = await client.post("/v1/messages", json=body, headers=headers)
    return resp


async def forward_streaming(
    body: dict, incoming_headers: dict[str, str]
) -> tuple[int, dict[str, str], AsyncIterator[bytes], dict]:
    """
    Forward a streaming request to Anthropic.

    Returns:
        (status_code, response_headers, chunk_iterator, usage_holder)

    usage_holder is a mutable dict that gets populated with cache stats
    when the `message_start` SSE event is intercepted during streaming.
    """
    api_key = _extract_api_key(incoming_headers)
    headers = _build_upstream_headers(incoming_headers, api_key)
    client = get_client()

    logger.debug("proxy.forward", model=body.get("model"), streaming=True)

    # Use stream() context — caller must handle cleanup via the iterator
    req = client.build_request("POST", "/v1/messages", json=body, headers=headers)
    resp = await client.send(req, stream=True)

    # Build response headers to forward
    resp_headers = {}
    for k, v in resp.headers.items():
        lower = k.lower()
        if lower not in ("content-encoding", "content-length", "transfer-encoding"):
            resp_headers[k] = v

    # Usage holder — populated by the stream interceptor
    usage_holder: dict = {}

    async def _stream_with_intercept() -> AsyncIterator[bytes]:
        """Yield raw SSE chunks while intercepting message_start for usage stats."""
        try:
            buffer = ""
            async for chunk in resp.aiter_text():
                # Yield the raw chunk immediately for minimal latency
                yield chunk.encode("utf-8") if isinstance(chunk, str) else chunk

                # Side-channel: parse SSE events to extract usage from message_start
                buffer += chunk
                while "\n\n" in buffer:
                    event_str, buffer = buffer.split("\n\n", 1)
                    _try_extract_usage(event_str, usage_holder)
        except Exception:
            logger.exception("proxy.stream_error")
            raise
        finally:
            await resp.aclose()

    return resp.status_code, resp_headers, _stream_with_intercept(), usage_holder


def _try_extract_usage(event_str: str, usage_holder: dict):
    """Try to extract usage info from an SSE event string (side-channel, non-blocking)."""
    try:
        # SSE format: "event: message_start\ndata: {...}"
        if "message_start" not in event_str:
            return
        for line in event_str.split("\n"):
            if line.startswith("data: "):
                data = json.loads(line[6:])
                usage = data.get("message", {}).get("usage", {})
                if usage:
                    usage_holder.update(usage)
                    logger.debug(
                        "proxy.usage_intercepted",
                        cache_read=usage.get("cache_read_input_tokens", 0),
                        cache_creation=usage.get("cache_creation_input_tokens", 0),
                        input_tokens=usage.get("input_tokens", 0),
                    )
                return
    except (json.JSONDecodeError, AttributeError, KeyError):
        pass  # Non-critical — don't break the stream for stats
