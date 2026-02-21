"""
OpenAI ↔ Anthropic format translation layer.

Translates between OpenAI's /v1/chat/completions format and Anthropic's
/v1/messages format, enabling OpenWebUI and other OpenAI-compatible
clients to use autocache transparently.

Key differences handled:
  - System message: OpenAI puts it in messages[]; Anthropic uses a separate field
  - Content format: OpenAI uses string; Anthropic uses string or content blocks
  - Streaming: OpenAI uses `data: {"choices":[...]}` SSE; Anthropic uses typed events
  - Model names: Passthrough (user specifies full Anthropic model name)
  - Token fields: OpenAI uses prompt_tokens/completion_tokens; Anthropic uses input/output
"""

from __future__ import annotations

import json
import time
import uuid

import structlog

logger = structlog.get_logger()


def _supports_adaptive(model: str) -> bool:
    """Check if a model supports adaptive thinking (4.6 series).

    Opus 4.6 and Sonnet 4.6 use thinking: {type: "adaptive"} + output_config.effort.
    Older models (Opus 4.5, Sonnet 4.5, etc.) use thinking: {type: "enabled", budget_tokens: N}.
    """
    lower = model.lower()
    return "4-6" in lower or "4.6" in lower


# ---------------------------------------------------------------------------
# Request: OpenAI → Anthropic
# ---------------------------------------------------------------------------


def openai_to_anthropic(request: dict) -> dict:
    """
    Convert an OpenAI chat completion request to Anthropic messages format.

    OpenAI format:
        {"model": "...", "messages": [{"role": "system", "content": "..."},
                                       {"role": "user", "content": "..."}],
         "max_tokens": 100, "stream": false}

    Anthropic format:
        {"model": "...", "system": "...",
         "messages": [{"role": "user", "content": "..."}],
         "max_tokens": 100, "stream": false}
    """
    messages = request.get("messages", [])

    # Extract system messages (OpenAI puts them in messages array)
    system_parts = []
    non_system_messages = []

    for msg in messages:
        role = msg.get("role", "")
        if role == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                # OpenAI multi-part system content
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        system_parts.append(part)
        else:
            # Translate OpenAI message to Anthropic format
            non_system_messages.append(_translate_message(msg))

    # Merge consecutive same-role messages (Anthropic requires alternating)
    merged_messages = _merge_consecutive_roles(non_system_messages)

    # Build Anthropic request
    anthropic_req: dict = {
        "model": request.get("model", ""),
        "max_tokens": request.get("max_tokens") or request.get("max_completion_tokens") or 4096,
        "messages": merged_messages,
    }

    # System prompt
    if system_parts:
        anthropic_req["system"] = "\n\n".join(system_parts)

    # Optional parameters
    if request.get("stream"):
        anthropic_req["stream"] = True
    if request.get("temperature") is not None:
        anthropic_req["temperature"] = request["temperature"]
    if request.get("top_p") is not None:
        anthropic_req["top_p"] = request["top_p"]
    if request.get("stop"):
        anthropic_req["stop_sequences"] = request["stop"] if isinstance(request["stop"], list) else [request["stop"]]

    # Tools (OpenAI → Anthropic format)
    if request.get("tools"):
        anthropic_req["tools"] = [_translate_tool(t) for t in request["tools"]]

    # --- Anthropic-specific parameter passthrough ---
    _ANTHROPIC_PASSTHROUGH = {
        "metadata",       # Request metadata
        "top_k",          # Top-K sampling
        "tool_choice",    # Tool choice strategy
    }
    for key in _ANTHROPIC_PASSTHROUGH:
        if key in request:
            anthropic_req[key] = request[key]

    # --- Thinking + Effort auto-adaptation ---
    # Priority: client-sent > environment default > disabled
    #
    # Latest Anthropic API (2026-02):
    #   Opus 4.6 / Sonnet 4.6  → thinking: {type: "adaptive"} (budget_tokens deprecated)
    #   Older models            → thinking: {type: "enabled", budget_tokens: N}
    #   Effort control          → output_config: {effort: "low/medium/high/max"}
    from app.config import settings

    thinking = request.get("thinking")
    if thinking is None and settings.default_thinking:
        # Client didn't send thinking — apply server default
        val = settings.default_thinking.strip().lower()
        if val == "true":
            thinking = True
        elif val == "adaptive":
            thinking = {"type": "adaptive"}
        elif val.isdigit():
            thinking = {"type": "enabled", "budget_tokens": int(val)}

    if thinking:
        model_lower = anthropic_req.get("model", "").lower()
        adaptive = _supports_adaptive(model_lower)
        default_budget = 10000

        if isinstance(thinking, bool):
            # thinking: true → auto-detect best config for model
            if adaptive:
                anthropic_req["thinking"] = {"type": "adaptive"}
            else:
                anthropic_req["thinking"] = {"type": "enabled", "budget_tokens": default_budget}
        elif isinstance(thinking, dict):
            thinking_type = thinking.get("type", "")
            if adaptive:
                # 4.6 models: always use adaptive (budget_tokens is deprecated)
                anthropic_req["thinking"] = {"type": "adaptive"}
            elif thinking_type == "adaptive":
                # Non-4.6 model but client asked for adaptive → fallback to enabled
                anthropic_req["thinking"] = {"type": "enabled", "budget_tokens": default_budget}
            else:
                # enabled + budget_tokens (older models)
                anthropic_req["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking.get("budget_tokens", default_budget),
                }

    # --- Effort parameter → output_config ---
    effort = request.get("effort") or request.get("output_config", {}).get("effort")
    if effort is None and settings.default_effort:
        effort = settings.default_effort.strip().lower()
    if effort:
        anthropic_req.setdefault("output_config", {})["effort"] = effort

    # Also pass through any key starting with "anthropic_" (future-proof)
    for key, value in request.items():
        if key.startswith("anthropic_") and key not in anthropic_req:
            anthropic_req[key] = value

    # Debug: log the final thinking/effort config
    logger.info(
        "openai_compat.translated",
        model=anthropic_req.get("model"),
        thinking=anthropic_req.get("thinking"),
        output_config=anthropic_req.get("output_config"),
    )

    return anthropic_req


def _translate_message(msg: dict) -> dict:
    """Translate a single OpenAI message to Anthropic format."""
    role = msg.get("role", "user")
    content = msg.get("content")

    # Map "assistant" tool_calls to Anthropic tool_use blocks
    if role == "assistant" and msg.get("tool_calls"):
        blocks = []
        if content:
            blocks.append({"type": "text", "text": content})
        for tc in msg["tool_calls"]:
            fn = tc.get("function", {})
            blocks.append({
                "type": "tool_use",
                "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "name": fn.get("name", ""),
                "input": json.loads(fn.get("arguments", "{}")),
            })
        return {"role": "assistant", "content": blocks}

    # Map "tool" role to Anthropic tool_result
    if role == "tool":
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": content if isinstance(content, str) else json.dumps(content),
            }],
        }

    # Regular message
    if isinstance(content, list):
        # OpenAI multi-part content (text + image_url etc.)
        blocks = []
        for part in content:
            if isinstance(part, str):
                blocks.append({"type": "text", "text": part})
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    blocks.append({"type": "text", "text": part.get("text", "")})
                elif part.get("type") == "image_url":
                    # Anthropic image format
                    url = part.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        # Base64 data URL
                        media_type, _, data = url.partition(";base64,")
                        media_type = media_type.replace("data:", "")
                        blocks.append({
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": data},
                        })
                    else:
                        blocks.append({
                            "type": "image",
                            "source": {"type": "url", "url": url},
                        })
        return {"role": role, "content": blocks}

    return {"role": role, "content": content or ""}


def _translate_tool(tool: dict) -> dict:
    """OpenAI tool definition → Anthropic tool definition."""
    fn = tool.get("function", {})
    return {
        "name": fn.get("name", ""),
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters", {"type": "object"}),
    }


def _merge_consecutive_roles(messages: list[dict]) -> list[dict]:
    """
    Merge consecutive messages with the same role.
    Anthropic requires strict user/assistant alternation.
    """
    if not messages:
        return messages

    merged = [messages[0]]
    for msg in messages[1:]:
        if msg["role"] == merged[-1]["role"]:
            # Merge content
            prev_content = merged[-1]["content"]
            curr_content = msg["content"]

            # Normalize both to list form
            if isinstance(prev_content, str):
                prev_content = [{"type": "text", "text": prev_content}]
            if isinstance(curr_content, str):
                curr_content = [{"type": "text", "text": curr_content}]
            if not isinstance(prev_content, list):
                prev_content = [prev_content]
            if not isinstance(curr_content, list):
                curr_content = [curr_content]

            merged[-1]["content"] = prev_content + curr_content
        else:
            merged.append(msg)

    return merged


# ---------------------------------------------------------------------------
# Response: Anthropic → OpenAI
# ---------------------------------------------------------------------------


def anthropic_to_openai(response: dict, model: str) -> dict:
    """
    Convert an Anthropic response to OpenAI chat completion format.

    Anthropic: {"content": [{"type": "text", "text": "Hello"}], "usage": {...}}
    OpenAI:    {"choices": [{"message": {"role": "assistant", "content": "Hello"}}], "usage": {...}}
    """
    # Extract text content
    content_blocks = response.get("content", [])
    text_parts = []
    thinking_parts = []
    tool_calls = []

    for block in content_blocks:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "thinking":
            thinking_parts.append(block.get("thinking", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })

    # Build content: prepend thinking in a <details> block if present
    final_text = ""
    if thinking_parts:
        thinking_text = "\n".join(thinking_parts)
        final_text += f"<details>\n<summary>Thinking</summary>\n\n{thinking_text}\n\n</details>\n\n"
    if text_parts:
        final_text += "\n".join(text_parts)

    message: dict = {
        "role": "assistant",
        "content": final_text if final_text else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    # Map usage
    anthropic_usage = response.get("usage", {})
    usage = {
        "prompt_tokens": anthropic_usage.get("input_tokens", 0),
        "completion_tokens": anthropic_usage.get("output_tokens", 0),
        "total_tokens": anthropic_usage.get("input_tokens", 0) + anthropic_usage.get("output_tokens", 0),
    }

    # Add cache info as extra fields (non-standard but useful)
    if anthropic_usage.get("cache_read_input_tokens"):
        usage["cache_read_input_tokens"] = anthropic_usage["cache_read_input_tokens"]
    if anthropic_usage.get("cache_creation_input_tokens"):
        usage["cache_creation_input_tokens"] = anthropic_usage["cache_creation_input_tokens"]

    # Map stop reason
    stop_reason_map = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
    }

    return {
        "id": f"chatcmpl-{response.get('id', uuid.uuid4().hex[:12])}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": stop_reason_map.get(response.get("stop_reason", ""), "stop"),
        }],
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# Streaming: Anthropic SSE → OpenAI SSE
# ---------------------------------------------------------------------------


class StreamTranslator:
    """
    Stateful translator for Anthropic SSE → OpenAI SSE.

    Tracks the current content block type so thinking blocks
    can be properly wrapped in <details> open/close tags.
    """

    def __init__(self, model: str, response_id: str):
        self.model = model
        self.response_id = response_id
        # Track block types by index for proper close handling
        self._block_types: dict[int, str] = {}

    def translate(self, event_str: str) -> list[str]:
        """Translate a raw Anthropic SSE event string into OpenAI SSE chunk(s)."""
        chunks = []

        for line in event_str.strip().split("\n"):
            if not line.startswith("data: "):
                continue

            try:
                data = json.loads(line[6:])
            except json.JSONDecodeError:
                continue

            event_type = data.get("type", "")

            if event_type == "message_start":
                chunk = _make_openai_chunk(self.response_id, self.model, delta={"role": "assistant", "content": ""})
                chunks.append(f"data: {json.dumps(chunk)}\n\n")

            elif event_type == "content_block_start":
                index = data.get("index", 0)
                block = data.get("content_block", {})
                block_type = block.get("type", "")
                self._block_types[index] = block_type

                if block_type == "thinking":
                    chunk = _make_openai_chunk(self.response_id, self.model, delta={
                        "content": "<details>\n<summary>Thinking</summary>\n\n"
                    })
                    chunks.append(f"data: {json.dumps(chunk)}\n\n")
                elif block_type == "tool_use":
                    chunk = _make_openai_chunk(self.response_id, self.model, delta={
                        "tool_calls": [{
                            "index": 0,
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {"name": block.get("name", ""), "arguments": ""},
                        }]
                    })
                    chunks.append(f"data: {json.dumps(chunk)}\n\n")

            elif event_type == "content_block_delta":
                delta = data.get("delta", {})
                if delta.get("type") == "text_delta":
                    chunk = _make_openai_chunk(self.response_id, self.model, delta={"content": delta.get("text", "")})
                    chunks.append(f"data: {json.dumps(chunk)}\n\n")
                elif delta.get("type") == "thinking_delta":
                    chunk = _make_openai_chunk(self.response_id, self.model, delta={"content": delta.get("thinking", "")})
                    chunks.append(f"data: {json.dumps(chunk)}\n\n")
                elif delta.get("type") == "input_json_delta":
                    chunk = _make_openai_chunk(self.response_id, self.model, delta={
                        "tool_calls": [{"index": 0, "function": {"arguments": delta.get("partial_json", "")}}]
                    })
                    chunks.append(f"data: {json.dumps(chunk)}\n\n")

            elif event_type == "content_block_stop":
                index = data.get("index", 0)
                block_type = self._block_types.pop(index, "")
                if block_type == "thinking":
                    chunk = _make_openai_chunk(self.response_id, self.model, delta={
                        "content": "\n\n</details>\n\n"
                    })
                    chunks.append(f"data: {json.dumps(chunk)}\n\n")

            elif event_type == "message_delta":
                stop_reason = data.get("delta", {}).get("stop_reason", "")
                stop_map = {"end_turn": "stop", "max_tokens": "length", "tool_use": "tool_calls"}
                chunk = _make_openai_chunk(
                    self.response_id, self.model,
                    delta={},
                    finish_reason=stop_map.get(stop_reason, "stop"),
                )
                chunks.append(f"data: {json.dumps(chunk)}\n\n")

            elif event_type == "message_stop":
                chunks.append("data: [DONE]\n\n")

        return chunks


def translate_streaming_chunk(event_str: str, model: str, response_id: str) -> list[str]:
    """
    Stateless wrapper for backward compatibility.
    For proper thinking block support, use StreamTranslator directly.
    """
    translator = StreamTranslator(model, response_id)
    return translator.translate(event_str)


def _make_openai_chunk(
    response_id: str, model: str, delta: dict, finish_reason: str | None = None
) -> dict:
    """Build a single OpenAI streaming chunk."""
    choice: dict = {"index": 0, "delta": delta, "finish_reason": finish_reason}
    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [choice],
    }
