"""
Core cache injection logic — the brains of autocache.

Implements a simple, deterministic 3-breakpoint strategy for Anthropic's
prompt caching. Zero configuration, just works.

Breakpoint placement (up to 4 allowed by Anthropic):
  BP1: system message — nearly always stable, highest cache hit rate
  BP2: tools[-1]     — tool definitions rarely change within a session
  BP3: last suitable content block before the final user message
       (ensures conversation history is cached for the next round)

Each breakpoint uses {"type": "ephemeral"} (Anthropic's only supported type).
TTL defaults to 5 minutes server-side; can be extended to 1 hour via the
optional ttl parameter, but we use the default in Phase 1 for simplicity.
"""

from __future__ import annotations

import re
import structlog

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

# Rough but conservative: English ~4 chars/token, Chinese ~1.5 chars/token.
# We intentionally use a slightly generous estimate to avoid missing cache
# opportunities near the threshold boundary. A future upgrade can swap in
# tiktoken for precision.
_CJK_RANGE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]")


def estimate_tokens(text: str) -> int:
    """Estimate token count for a string. Conservative (tends to overcount)."""
    if not text:
        return 0
    cjk_chars = len(_CJK_RANGE.findall(text))
    other_chars = len(text) - cjk_chars
    # CJK: ~1.5 chars/token; other: ~4 chars/token
    return max(1, int(cjk_chars / 1.5 + other_chars / 4))


def estimate_block_tokens(block: dict) -> int:
    """Estimate tokens for a single content block or tool definition."""
    if isinstance(block, str):
        return estimate_tokens(block)
    # text block
    if block.get("type") == "text":
        return estimate_tokens(block.get("text", ""))
    # tool definition — name + description + schema
    if "input_schema" in block:
        parts = [block.get("name", ""), block.get("description", ""), str(block.get("input_schema", ""))]
        return estimate_tokens(" ".join(parts))
    # tool_use / tool_result / image — don't estimate, treat as small
    return 10


# ---------------------------------------------------------------------------
# Breakpoint injection
# ---------------------------------------------------------------------------

CACHE_CONTROL = {"type": "ephemeral"}


def _already_has_cache_control(obj: dict) -> bool:
    """Check if a dict already carries cache_control."""
    return "cache_control" in obj and obj["cache_control"] is not None


def _count_existing_breakpoints(request: dict) -> int:
    """Count breakpoints the caller already placed in the request."""
    count = 0
    # system blocks
    system = request.get("system")
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and _already_has_cache_control(block):
                count += 1
    # tools
    for tool in request.get("tools", []):
        if isinstance(tool, dict) and _already_has_cache_control(tool):
            count += 1
    # messages
    for msg in request.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and _already_has_cache_control(block):
                    count += 1
    return count


def _is_cacheable_block(block: dict) -> bool:
    """A content block is cacheable if it's a text block (not tool_result, not image)."""
    if not isinstance(block, dict):
        return False
    btype = block.get("type", "")
    return btype == "text"


def _find_last_cacheable_block_in_message(msg: dict) -> dict | None:
    """Find the last cacheable (text) content block in a message."""
    content = msg.get("content")
    if isinstance(content, str):
        # Will need to convert to block form — return sentinel
        return msg
    if isinstance(content, list):
        for block in reversed(content):
            if isinstance(block, dict) and _is_cacheable_block(block) and not _already_has_cache_control(block):
                return block
    return None


def _estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate total tokens across a list of messages."""
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            total += sum(estimate_block_tokens(b) for b in content if isinstance(b, dict))
    return total


def inject_cache_breakpoints(request: dict, min_tokens: int = 1024) -> dict:
    """
    Inject cache_control breakpoints into an Anthropic API request dict.

    Mutates and returns the same dict. Respects the 4-breakpoint limit.
    Does not overwrite existing cache_control fields.

    Strategy: Uses aggregate token counting — system + tools tokens are summed
    together when checking thresholds. This prevents skipping cache on Haiku
    where system alone is under 2048 but system + tools combined are over.

    Returns:
        The mutated request dict (also returned for convenience).
    """
    max_breakpoints = 4
    existing = _count_existing_breakpoints(request)
    remaining = max_breakpoints - existing

    if remaining <= 0:
        logger.info("cache.skip", reason="max_breakpoints_already_reached", existing=existing)
        return request

    injected: list[str] = []

    # Pre-calculate token counts for smarter threshold decisions
    system = request.get("system")
    system_tokens = 0
    if isinstance(system, str):
        system_tokens = estimate_tokens(system)
    elif isinstance(system, list) and system:
        system_tokens = sum(estimate_block_tokens(b) for b in system if isinstance(b, dict))

    tools = request.get("tools")
    tool_tokens = 0
    if tools and isinstance(tools, list):
        tool_tokens = sum(estimate_block_tokens(t) for t in tools)

    # Aggregate prefix tokens — used for threshold check
    prefix_tokens = system_tokens + tool_tokens

    logger.debug(
        "cache.token_analysis",
        system_tokens=system_tokens,
        tool_tokens=tool_tokens,
        prefix_tokens=prefix_tokens,
        min_tokens=min_tokens,
    )

    # --- BP1: system ---
    # Cache if system alone OR system+tools combined exceed threshold
    if remaining > 0 and system_tokens > 0 and prefix_tokens >= min_tokens:
        if isinstance(system, str):
            request["system"] = [{"type": "text", "text": system, "cache_control": CACHE_CONTROL}]
            injected.append("system")
            remaining -= 1
        elif isinstance(system, list) and system:
            for block in reversed(system):
                if isinstance(block, dict) and _is_cacheable_block(block) and not _already_has_cache_control(block):
                    block["cache_control"] = CACHE_CONTROL
                    injected.append("system")
                    remaining -= 1
                    break
    elif system_tokens > 0 and prefix_tokens < min_tokens:
        logger.info(
            "cache.skip_bp",
            bp="system",
            reason="insufficient_tokens",
            system_tokens=system_tokens,
            prefix_tokens=prefix_tokens,
            min_tokens=min_tokens,
        )

    # --- BP2: tools[-1] ---
    # Cache if tools present and prefix exceeds threshold
    if remaining > 0 and tool_tokens > 0 and prefix_tokens >= min_tokens:
        last_tool = tools[-1]
        if isinstance(last_tool, dict) and not _already_has_cache_control(last_tool):
            last_tool["cache_control"] = CACHE_CONTROL
            injected.append("tools")
            remaining -= 1
    elif tool_tokens > 0 and prefix_tokens < min_tokens:
        logger.info(
            "cache.skip_bp",
            bp="tools",
            reason="insufficient_tokens",
            tool_tokens=tool_tokens,
            prefix_tokens=prefix_tokens,
            min_tokens=min_tokens,
        )

    # --- BP3: best message before the last user message ---
    # Walk backwards from messages[-2], find the first message with enough
    # total tokens whose last cacheable content block can hold a breakpoint.
    # Token threshold is checked per-MESSAGE (sum of all blocks), not per-block.
    # This handles:
    #   - Normal text messages (string or blocks)
    #   - Assistant messages with tool_use blocks (picks last text block)
    #   - Skips tool_result-only messages (no cacheable text block)
    if remaining > 0:
        messages = request.get("messages", [])
        if len(messages) >= 2:
            for i in range(len(messages) - 2, -1, -1):
                msg = messages[i]
                content = msg.get("content")

                # Calculate total tokens for this message
                if isinstance(content, str):
                    msg_tokens = estimate_tokens(content)
                elif isinstance(content, list):
                    msg_tokens = sum(estimate_block_tokens(b) for b in content if isinstance(b, dict))
                else:
                    continue

                if msg_tokens < min_tokens:
                    continue  # Message too small, try earlier ones

                # Message has enough tokens — find a cacheable block to attach BP
                if isinstance(content, str):
                    messages[i]["content"] = [
                        {"type": "text", "text": content, "cache_control": CACHE_CONTROL}
                    ]
                    injected.append(f"message[{i}]")
                    remaining -= 1
                    break

                block = _find_last_cacheable_block_in_message(msg)
                if block is not None and isinstance(block, dict) and block.get("type") == "text":
                    block["cache_control"] = CACHE_CONTROL
                    injected.append(f"message[{i}]")
                    remaining -= 1
                    break

            # Fallback: if no single message is big enough, check cumulative
            # tokens across all history messages and place BP on messages[-2]
            if not any("message[" in bp for bp in injected) and len(messages) >= 3:
                cumulative = _estimate_messages_tokens(messages[:-1])
                if cumulative >= min_tokens:
                    for i in range(len(messages) - 2, -1, -1):
                        block = _find_last_cacheable_block_in_message(messages[i])
                        if block is not None and isinstance(block, dict) and block.get("type") == "text":
                            block["cache_control"] = CACHE_CONTROL
                            injected.append(f"message[{i}]:cumulative")
                            remaining -= 1
                            break

    logger.info(
        "cache.injected",
        breakpoints=injected,
        total=len(injected) + existing,
        existing=existing,
    )
    return request


# ---------------------------------------------------------------------------
# Metadata for response headers
# ---------------------------------------------------------------------------


def build_cache_metadata(request: dict, injected_count: int) -> dict:
    """Build metadata dict for X-Autocache-* response headers."""
    total_tokens = 0
    cached_tokens = 0

    # Estimate total input tokens
    system = request.get("system")
    if isinstance(system, str):
        total_tokens += estimate_tokens(system)
    elif isinstance(system, list):
        for b in system:
            t = estimate_block_tokens(b)
            total_tokens += t
            if isinstance(b, dict) and _already_has_cache_control(b):
                cached_tokens += t

    for tool in request.get("tools", []):
        t = estimate_block_tokens(tool)
        total_tokens += t
        if isinstance(tool, dict) and _already_has_cache_control(tool):
            cached_tokens += t

    for msg in request.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            total_tokens += estimate_tokens(content)
        elif isinstance(content, list):
            for b in content:
                t = estimate_block_tokens(b)
                total_tokens += t
                if isinstance(b, dict) and _already_has_cache_control(b):
                    cached_tokens += t

    cache_ratio = cached_tokens / total_tokens if total_tokens > 0 else 0.0

    return {
        "injected": str(injected_count > 0).lower(),
        "total_tokens": str(total_tokens),
        "cached_tokens": str(cached_tokens),
        "cache_ratio": f"{cache_ratio:.3f}",
        "breakpoints": str(injected_count),
    }
