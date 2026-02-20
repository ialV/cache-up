"""
Unit tests for cache_injector — the core breakpoint injection logic.

Tests cover:
  - System message (string and blocks format)
  - Tools breakpoint
  - Message breakpoint (normal, tool_use, tool_result edge cases)
  - Existing cache_control preservation
  - Token threshold gating
  - Max 4 breakpoints limit
"""

import copy

import pytest

from app.cache_injector import (
    CACHE_CONTROL,
    estimate_tokens,
    inject_cache_breakpoints,
    build_cache_metadata,
)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_english(self):
        # "Hello world" = 11 chars / 4 ≈ 2-3 tokens
        t = estimate_tokens("Hello world")
        assert 1 <= t <= 5

    def test_chinese(self):
        # 10 Chinese chars / 1.5 ≈ 6-7 tokens
        t = estimate_tokens("你好世界测试中文文本了")
        assert t >= 5

    def test_long_english_above_threshold(self):
        # 5000 chars / 4 = 1250 tokens — above default 1024 threshold
        text = "a " * 2500
        assert estimate_tokens(text) >= 1024


# ---------------------------------------------------------------------------
# System message breakpoint (BP1)
# ---------------------------------------------------------------------------


class TestSystemBreakpoint:
    def test_string_system_gets_breakpoint(self):
        """Long system string → converted to blocks + breakpoint injected."""
        req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "system": "x " * 3000,  # ~1500 tokens
            "messages": [{"role": "user", "content": "hi"}],
        }
        inject_cache_breakpoints(req)

        # Should have been converted to blocks
        assert isinstance(req["system"], list)
        assert len(req["system"]) == 1
        assert req["system"][0]["cache_control"] == CACHE_CONTROL

    def test_short_system_no_breakpoint(self):
        """Short system string → no breakpoint, stays as string."""
        req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "system": "Be helpful.",
            "messages": [{"role": "user", "content": "hi"}],
        }
        inject_cache_breakpoints(req)

        # Should remain a string (not converted)
        assert isinstance(req["system"], str)

    def test_blocks_system_gets_breakpoint(self):
        """System already in blocks format → last text block gets breakpoint."""
        req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "system": [
                {"type": "text", "text": "Instructions: " + "x " * 2000},
                {"type": "text", "text": "More context: " + "y " * 1000},
            ],
            "messages": [{"role": "user", "content": "hi"}],
        }
        inject_cache_breakpoints(req)

        # Breakpoint should be on the LAST text block
        assert req["system"][-1].get("cache_control") == CACHE_CONTROL
        assert req["system"][0].get("cache_control") is None


# ---------------------------------------------------------------------------
# Tools breakpoint (BP2)
# ---------------------------------------------------------------------------


class TestToolsBreakpoint:
    def test_tools_get_breakpoint(self):
        """Tools with enough tokens → last tool gets breakpoint."""
        req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "name": "search",
                    "description": "Search the web for information. " * 100,
                    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
                {
                    "name": "calculate",
                    "description": "Perform calculations. " * 100,
                    "input_schema": {"type": "object", "properties": {"expr": {"type": "string"}}},
                },
            ],
        }
        inject_cache_breakpoints(req)

        # Last tool should have breakpoint
        assert req["tools"][-1].get("cache_control") == CACHE_CONTROL
        assert req["tools"][0].get("cache_control") is None

    def test_short_tools_no_breakpoint(self):
        """Tools below token threshold → no breakpoint."""
        req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"name": "ping", "description": "Ping", "input_schema": {"type": "object"}},
            ],
        }
        inject_cache_breakpoints(req)

        assert req["tools"][0].get("cache_control") is None


# ---------------------------------------------------------------------------
# Message breakpoint (BP3)
# ---------------------------------------------------------------------------


class TestMessageBreakpoint:
    def test_penultimate_message_gets_breakpoint(self):
        """messages[-2] with long text → gets breakpoint."""
        req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Tell me about Python."},
                {"role": "assistant", "content": [{"type": "text", "text": "Python is a language. " * 500}]},
                {"role": "user", "content": "What about Go?"},
            ],
        }
        inject_cache_breakpoints(req)

        # messages[-2] (assistant) should have breakpoint on its text block
        content = req["messages"][1]["content"]
        assert isinstance(content, list)
        assert content[-1].get("cache_control") == CACHE_CONTROL

    def test_single_message_no_content_breakpoint(self):
        """Only 1 message → no content breakpoint (nothing to cache)."""
        req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        }
        inject_cache_breakpoints(req)

        # No breakpoint on the only message
        content = req["messages"][0].get("content")
        if isinstance(content, list):
            for block in content:
                assert block.get("cache_control") is None
        # String content → no breakpoint possible (no change needed)

    def test_tool_use_message_finds_text_block(self):
        """Assistant message with tool_use blocks → breakpoint on last TEXT block."""
        req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Search for AI news. " * 200},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I'll search for that. " * 200},
                        {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "AI"}},
                    ],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "results"}]},
                {"role": "user", "content": "Thanks!"},
            ],
        }
        inject_cache_breakpoints(req)

        # The text block in messages[1] (assistant) should get the breakpoint
        # tool_use block should NOT get breakpoint
        assistant_content = req["messages"][1]["content"]
        text_block = [b for b in assistant_content if b.get("type") == "text"][0]
        tool_block = [b for b in assistant_content if b.get("type") == "tool_use"][0]
        assert text_block.get("cache_control") == CACHE_CONTROL
        assert tool_block.get("cache_control") is None

    def test_tool_result_message_skipped(self):
        """messages[-2] is tool_result → skip, search backwards for text message."""
        req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "First message. " * 200},
                {"role": "assistant", "content": [{"type": "text", "text": "Response. " * 200}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "data"}]},
                {"role": "user", "content": "Follow up question"},
            ],
        }
        inject_cache_breakpoints(req)

        # tool_result message should NOT have breakpoint
        tool_result_msg = req["messages"][2]
        content = tool_result_msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    assert block.get("cache_control") is None

        # The assistant message (messages[1]) should get the breakpoint instead
        assistant_content = req["messages"][1]["content"]
        assert assistant_content[-1].get("cache_control") == CACHE_CONTROL


# ---------------------------------------------------------------------------
# Existing cache_control preservation
# ---------------------------------------------------------------------------


class TestExistingCacheControl:
    def test_existing_breakpoints_preserved(self):
        """If user already placed cache_control, don't overwrite or exceed limit."""
        req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "system": [
                {"type": "text", "text": "System. " * 1000, "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [
                {"role": "user", "content": "hi"},
            ],
        }
        original_cc = req["system"][0]["cache_control"]
        inject_cache_breakpoints(req)

        # Original cache_control should be untouched
        assert req["system"][0]["cache_control"] is original_cc

    def test_max_breakpoints_respected(self):
        """With 4 existing breakpoints, no new ones should be added."""
        req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "system": [
                {"type": "text", "text": "A" * 5000, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "B" * 5000, "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "C" * 5000, "cache_control": {"type": "ephemeral"}},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "D" * 5000, "cache_control": {"type": "ephemeral"}},
                    ],
                },
                {"role": "user", "content": "hi"},
            ],
        }
        original = copy.deepcopy(req)
        inject_cache_breakpoints(req)

        # Nothing should have changed — already at 4 breakpoints
        assert req == original


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestBuildCacheMetadata:
    def test_metadata_after_injection(self):
        req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "system": "System prompt. " * 500,
            "messages": [{"role": "user", "content": "hi"}],
        }
        inject_cache_breakpoints(req)
        meta = build_cache_metadata(req, injected_count=1)

        assert meta["injected"] == "true"
        assert int(meta["total_tokens"]) > 0
        assert int(meta["cached_tokens"]) > 0
        assert float(meta["cache_ratio"]) > 0

    def test_metadata_no_injection(self):
        req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "system": "hi",
            "messages": [{"role": "user", "content": "hi"}],
        }
        inject_cache_breakpoints(req)
        meta = build_cache_metadata(req, injected_count=0)

        assert meta["injected"] == "false"
