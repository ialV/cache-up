"""
HTTP routes — the API surface of autocache.

POST /v1/messages          — Main proxy endpoint (drop-in for Anthropic API)
POST /v1/chat/completions  — OpenAI-compatible endpoint (for OpenWebUI etc.)
GET  /v1/models            — Model list (OpenAI-compatible)
GET  /health               — Health check
GET  /savings              — Cache hit statistics
"""

from __future__ import annotations

import json
import time
import uuid

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app import proxy
from app.cache_injector import InjectionReport, build_cache_metadata, inject_cache_breakpoints
from app.config import settings
from app.observability import RequestRecord, stats
from app.openai_compat import (
    anthropic_to_openai,
    openai_to_anthropic,
    translate_streaming_chunk,
)

logger = structlog.get_logger()
router = APIRouter()

# ---------------------------------------------------------------------------
# Model list cache — fetches from Anthropic, caches for 5 minutes
# ---------------------------------------------------------------------------

_model_cache: dict = {"models": [], "fetched_at": 0.0}
_MODEL_CACHE_TTL = 300  # 5 minutes

# Minimal fallback if Anthropic API is unreachable
_FALLBACK_MODELS = [
    {"id": "claude-sonnet-4-20250514", "display_name": "Claude Sonnet 4"},
    {"id": "claude-3-5-haiku-20241022", "display_name": "Claude 3.5 Haiku"},
]


async def _fetch_anthropic_models() -> list[dict]:
    """Fetch all models from Anthropic /v1/models with pagination."""
    from app.proxy import get_client
    client = get_client()
    api_key = settings.anthropic_api_key
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    all_models = []
    after_id = None

    try:
        for _ in range(10):  # safety: max 10 pages
            params = {"limit": 100}
            if after_id:
                params["after_id"] = after_id

            resp = await client.get("/v1/models", params=params, headers=headers)
            if resp.status_code != 200:
                logger.warning("models.fetch_error", status=resp.status_code)
                break

            data = resp.json()
            all_models.extend(data.get("data", []))

            if not data.get("has_more"):
                break
            after_id = data.get("last_id")

        logger.info("models.fetched", count=len(all_models))
    except Exception:
        logger.exception("models.fetch_exception")

    return all_models


async def _get_models_cached() -> list[dict]:
    """Get models with in-memory caching."""
    now = time.time()
    if _model_cache["models"] and (now - _model_cache["fetched_at"]) < _MODEL_CACHE_TTL:
        return _model_cache["models"]

    models = await _fetch_anthropic_models()
    if models:
        _model_cache["models"] = models
        _model_cache["fetched_at"] = now
        return models

    # Fallback: return cached (even if stale) or hardcoded
    return _model_cache["models"] or _FALLBACK_MODELS


# ---------------------------------------------------------------------------
# POST /v1/messages (Anthropic native)
# ---------------------------------------------------------------------------


@router.post("/v1/messages")
async def messages(request: Request):
    """Proxy endpoint — injects cache breakpoints, forwards to Anthropic."""
    try:
        body: dict = await request.json()
    except Exception:
        return JSONResponse({"error": {"type": "invalid_request", "message": "Invalid JSON body"}}, status_code=400)

    model = body.get("model", "unknown")
    is_streaming = body.get("stream", False)

    # Check bypass header
    if request.headers.get("x-autocache-bypass", "").lower() in ("true", "1"):
        logger.info("route.bypass", model=model)
        return await _forward_without_cache(body, dict(request.headers), is_streaming)

    # Determine min tokens threshold (Haiku models need 2048)
    min_tokens = settings.min_tokens_for_cache
    if "haiku" in model.lower():
        min_tokens = max(min_tokens, 2048)

    # Inject cache breakpoints
    body, report = inject_cache_breakpoints(body, min_tokens=min_tokens)
    injected_count = len(report.breakpoints_injected)

    meta = build_cache_metadata(body, injected_count)
    autocache_headers = {f"x-autocache-{k}": v for k, v in meta.items()}

    if request.headers.get("x-autocache-debug", "").lower() in ("true", "1"):
        autocache_headers["x-autocache-model"] = model
        autocache_headers["x-autocache-min-tokens"] = str(min_tokens)

    logger.debug("route.injection_report", **report.to_log_dict())

    incoming_headers = dict(request.headers)

    if is_streaming:
        return await _handle_streaming(body, incoming_headers, autocache_headers, meta, model, report)
    else:
        return await _handle_non_streaming(body, incoming_headers, autocache_headers, meta, model, report)


# ---------------------------------------------------------------------------
# POST /v1/chat/completions (OpenAI compatible)
# ---------------------------------------------------------------------------


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint.

    Translates OpenAI format → Anthropic, injects cache, forwards,
    then translates response back to OpenAI format.
    """
    try:
        openai_body: dict = await request.json()
    except Exception:
        return JSONResponse(
            {"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}},
            status_code=400,
        )

    model = openai_body.get("model", "unknown")
    is_streaming = openai_body.get("stream", False)

    logger.info("route.openai_compat", model=model, streaming=is_streaming)

    # Translate OpenAI → Anthropic format
    try:
        anthropic_body = openai_to_anthropic(openai_body)
    except Exception as e:
        logger.exception("route.openai_translate_error")
        return JSONResponse(
            {"error": {"message": f"Failed to translate request: {e}", "type": "invalid_request_error"}},
            status_code=400,
        )

    # Apply cache injection
    min_tokens = settings.min_tokens_for_cache
    if "haiku" in model.lower():
        min_tokens = max(min_tokens, 2048)

    anthropic_body, report = inject_cache_breakpoints(anthropic_body, min_tokens=min_tokens)
    injected_count = len(report.breakpoints_injected)

    meta = build_cache_metadata(anthropic_body, injected_count)
    autocache_headers = {f"x-autocache-{k}": v for k, v in meta.items()}

    logger.debug("route.injection_report", **report.to_log_dict())

    incoming_headers = dict(request.headers)

    if is_streaming:
        return await _handle_openai_streaming(anthropic_body, incoming_headers, autocache_headers, meta, model, report)
    else:
        return await _handle_openai_non_streaming(anthropic_body, incoming_headers, autocache_headers, meta, model, report)


async def _handle_openai_non_streaming(
    body: dict, incoming_headers: dict, autocache_headers: dict, meta: dict, model: str,
    report: InjectionReport | None = None,
) -> JSONResponse:
    """Handle non-streaming OpenAI-compat request."""
    resp = await proxy.forward_request(body, incoming_headers)

    usage = {}
    try:
        resp_data = json.loads(resp.content)
        usage = resp_data.get("usage", {})
    except (json.JSONDecodeError, AttributeError):
        resp_data = {}

    _record_stats(meta, model, streaming=False, usage=usage, report=report)

    if resp.status_code != 200:
        return JSONResponse(content=resp_data, status_code=resp.status_code, headers=autocache_headers)

    openai_resp = anthropic_to_openai(resp_data, model)
    return JSONResponse(content=openai_resp, status_code=200, headers=autocache_headers)


async def _handle_openai_streaming(
    body: dict, incoming_headers: dict, autocache_headers: dict, meta: dict, model: str,
    report: InjectionReport | None = None,
) -> StreamingResponse:
    """Handle streaming OpenAI-compat request — translate SSE chunks on the fly."""
    body["stream"] = True
    status_code, resp_headers, chunk_iter, usage_holder = await proxy.forward_streaming(body, incoming_headers)

    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    async def _translated_stream():
        buffer = ""
        try:
            async for raw_chunk in chunk_iter:
                text = raw_chunk.decode("utf-8") if isinstance(raw_chunk, bytes) else raw_chunk
                buffer += text

                while "\n\n" in buffer:
                    event_str, buffer = buffer.split("\n\n", 1)
                    openai_chunks = translate_streaming_chunk(event_str, model, response_id)
                    for chunk in openai_chunks:
                        yield chunk.encode("utf-8")
        except Exception:
            logger.exception("route.openai_stream_error")
        finally:
            _record_stats(meta, model, streaming=True, usage=usage_holder, report=report)

    return StreamingResponse(
        _translated_stream(),
        status_code=status_code,
        headers=autocache_headers,
        media_type="text/event-stream",
    )


# ---------------------------------------------------------------------------
# GET /v1/models (OpenAI compatible)
# ---------------------------------------------------------------------------


@router.get("/v1/models")
async def list_models():
    """Return available models in OpenAI format (dynamically fetched from Anthropic)."""
    models = await _get_models_cached()
    return {
        "object": "list",
        "data": [
            {
                "id": m["id"],
                "object": "model",
                "created": int(time.mktime(time.strptime(m["created_at"][:10], "%Y-%m-%d"))) if m.get("created_at") else 1700000000,
                "owned_by": "anthropic",
            }
            for m in models
        ],
    }


# ---------------------------------------------------------------------------
# Streaming (Anthropic native)
# ---------------------------------------------------------------------------


async def _handle_streaming(
    body: dict, incoming_headers: dict, autocache_headers: dict, meta: dict, model: str,
    report: InjectionReport | None = None,
) -> StreamingResponse:
    status_code, resp_headers, chunk_iter, usage_holder = await proxy.forward_streaming(body, incoming_headers)
    all_headers = {**resp_headers, **autocache_headers}

    async def _wrapped_stream():
        async for chunk in chunk_iter:
            yield chunk
        _record_stats(meta, model, streaming=True, usage=usage_holder, report=report)

    return StreamingResponse(
        _wrapped_stream(), status_code=status_code, headers=all_headers, media_type="text/event-stream",
    )


# ---------------------------------------------------------------------------
# Non-streaming (Anthropic native)
# ---------------------------------------------------------------------------


async def _handle_non_streaming(
    body: dict, incoming_headers: dict, autocache_headers: dict, meta: dict, model: str,
    report: InjectionReport | None = None,
) -> JSONResponse:
    resp = await proxy.forward_request(body, incoming_headers)

    resp_body = resp.content
    usage = {}
    try:
        resp_data = json.loads(resp_body)
        usage = resp_data.get("usage", {})
    except (json.JSONDecodeError, AttributeError):
        pass

    _record_stats(meta, model, streaming=False, usage=usage, report=report)

    response = JSONResponse(
        content=json.loads(resp_body) if resp_body else {},
        status_code=resp.status_code,
        headers=autocache_headers,
    )
    for key in ("x-request-id", "anthropic-ratelimit-requests-remaining", "anthropic-ratelimit-tokens-remaining"):
        val = resp.headers.get(key)
        if val:
            response.headers[key] = val

    return response


# ---------------------------------------------------------------------------
# Bypass / Health / Savings
# ---------------------------------------------------------------------------


async def _forward_without_cache(body: dict, incoming_headers: dict, is_streaming: bool):
    bypass_headers = {"x-autocache-injected": "false"}
    if is_streaming:
        status_code, resp_headers, chunk_iter, _ = await proxy.forward_streaming(body, incoming_headers)
        return StreamingResponse(
            chunk_iter, status_code=status_code,
            headers={**resp_headers, **bypass_headers}, media_type="text/event-stream",
        )
    else:
        resp = await proxy.forward_request(body, incoming_headers)
        return JSONResponse(
            content=json.loads(resp.content) if resp.content else {},
            status_code=resp.status_code, headers=bypass_headers,
        )


@router.get("/health")
async def health():
    return {"status": "healthy", "version": "0.2.0"}


@router.get("/savings")
async def savings():
    return stats.get_summary()


@router.get("/savings/recent")
async def savings_recent(n: int = 10):
    """Return the last N requests with full injection diagnostics."""
    return stats.get_recent(n=min(n, 50))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record_stats(
    meta: dict, model: str, streaming: bool, usage: dict,
    report: InjectionReport | None = None,
):
    rec = RequestRecord(
        timestamp=time.time(),
        model=model,
        estimated_total_tokens=int(meta.get("total_tokens", 0)),
        estimated_cached_tokens=int(meta.get("cached_tokens", 0)),
        breakpoints_injected=int(meta.get("breakpoints", 0)),
        actual_input_tokens=usage.get("input_tokens", 0),
        actual_cache_read_tokens=usage.get("cache_read_input_tokens", 0),
        actual_cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
        streaming=streaming,
    )
    if report:
        rec.system_tokens = report.system_tokens
        rec.system_chars = report.system_chars
        rec.tool_tokens = report.tool_tokens
        rec.prefix_tokens = report.prefix_tokens
        rec.message_count = report.message_count
        rec.message_breakdown = report.message_breakdown
        rec.breakpoint_positions = report.breakpoints_injected
        rec.injection_decisions = [
            {"name": d.name, "action": d.action, "reason": d.reason, "tokens": d.tokens}
            for d in report.decisions
        ]
    stats.record(rec)
