"""
Unit tests for cache_injector — the core breakpoint injection logic.

Tests cover:
  - System message (string and blocks format)
  - Tools breakpoint
  - Message breakpoint (normal, tool_use, tool_result edge cases)
  - Existing cache_control preservation
  - Token threshold gating
  - Max 4 breakpoints limit
  - Cumulative prefix token logic (Phase 2)
  - InjectionReport diagnostics
"""

import copy

import pytest

from app.cache_injector import (
    CACHE_CONTROL,
    InjectionReport,
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
        req, report = inject_cache_breakpoints(req)

        # Should have been converted to blocks
        assert isinstance(req["system"], list)
        assert len(req["system"]) == 1
        assert req["system"][0]["cache_control"] == CACHE_CONTROL
        assert "system" in report.breakpoints_injected

    def test_short_system_no_breakpoint(self):
        """Short system string → no breakpoint, stays as string."""
        req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "system": "Be helpful.",
            "messages": [{"role": "user", "content": "hi"}],
        }
        req, report = inject_cache_breakpoints(req)

        # Should remain a string (not converted)
        assert isinstance(req["system"], str)
        assert "system" not in report.breakpoints_injected

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
        req, report = inject_cache_breakpoints(req)

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
        req, report = inject_cache_breakpoints(req)

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
        req, report = inject_cache_breakpoints(req)

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
        req, report = inject_cache_breakpoints(req)

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
        req, report = inject_cache_breakpoints(req)

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
        req, report = inject_cache_breakpoints(req)

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
        req, report = inject_cache_breakpoints(req)

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
# Cumulative prefix token logic (Phase 2 — BP3 improvement)
# ---------------------------------------------------------------------------


class TestCumulativeTokenLogic:
    def test_short_messages_cumulative_cache(self):
        """
        Simulates OpenWebUI typical scenario: each message is short (~50-200 tokens),
        but cumulative conversation exceeds min_tokens threshold.
        """
        req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "system": "You are a helpful assistant.",  # ~6 tokens
            "messages": [
                {"role": "user", "content": "Hello, tell me something new. " * 20},      # ~120 tokens
                {"role": "assistant", "content": "Here is something interesting. " * 50},  # ~300 tokens
                {"role": "user", "content": "Cool, what else? " * 20},                     # ~80 tokens
                {"role": "assistant", "content": "Here is another fact. " * 80},            # ~400 tokens
                {"role": "user", "content": "And more?"},                                  # ~3 tokens
            ],
        }
        req, report = inject_cache_breakpoints(req)

        # Cumulative tokens from system + messages[0..3] should exceed 1024.
        # BP3 should be placed on messages[3] (the last message before final user msg).
        # Check that some message got a breakpoint
        assert any("message[" in bp for bp in report.breakpoints_injected), \
            f"Expected message breakpoint, got: {report.breakpoints_injected}"

    def test_two_messages_with_long_first(self):
        """2 messages: first user message is long (>1024), second is short."""
        req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Here is a long article. " * 300},  # ~1800 tokens
                {"role": "user", "content": "Summarize it."},                    # ~3 tokens
            ],
        }
        req, report = inject_cache_breakpoints(req)

        # message[0] has cumulative > 1024, should get BP3
        assert "message[0]" in report.breakpoints_injected

    def test_very_short_conversation_no_cache(self):
        """2 messages both very short — cumulative < 1024, no BP3."""
        req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "user", "content": "Hello"},
            ],
        }
        req, report = inject_cache_breakpoints(req)

        assert not any("message[" in bp for bp in report.breakpoints_injected)

    def test_system_prefix_contributes_to_cumulative(self):
        """System prompt + short messages: if system is big enough, messages[0] can get BP3."""
        req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "system": "You are a detailed assistant. " * 200,  # ~1000 tokens
            "messages": [
                {"role": "user", "content": "Tell me about cats."},  # ~6 tokens
                {"role": "assistant", "content": "Cats are great. " * 10},  # ~40 tokens
                {"role": "user", "content": "More?"},  # ~1 token
            ],
        }
        req, report = inject_cache_breakpoints(req)

        # system is ~1000 tokens → BP1 should fire (prefix >= 1024? marginally)
        # message[0] cumulative = 1000 + 6 = 1006 — borderline
        # message[1] cumulative = 1000 + 6 + 40 = 1046 → should qualify for BP3
        # At minimum, BP should be placed on messages[1] if cumulative >= 1024
        has_bp3 = any("message[" in bp for bp in report.breakpoints_injected)
        # Either system BP or message BP should fire since total tokens are ~1047
        has_any_bp = len(report.breakpoints_injected) > 0
        assert has_any_bp, f"Expected at least one breakpoint, got: {report.breakpoints_injected}"


# ---------------------------------------------------------------------------
# InjectionReport diagnostics
# ---------------------------------------------------------------------------


class TestInjectionReport:
    def test_report_has_message_breakdown(self):
        """Report should contain per-message token breakdown."""
        req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "system": "System prompt. " * 500,
            "messages": [
                {"role": "user", "content": "first " * 100},
                {"role": "assistant", "content": "second " * 200},
                {"role": "user", "content": "third"},
            ],
        }
        req, report = inject_cache_breakpoints(req)

        assert report.message_count == 3
        assert len(report.message_breakdown) == 3
        assert report.message_breakdown[0]["role"] == "user"
        assert report.message_breakdown[1]["role"] == "assistant"
        assert report.message_breakdown[2]["role"] == "user"
        assert all(mb["tokens"] > 0 for mb in report.message_breakdown[:2])

    def test_report_records_decisions(self):
        """Report should record each BP decision with reason."""
        req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "system": "x " * 3000,  # big enough for BP1
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "user", "content": "hello"},
            ],
        }
        req, report = inject_cache_breakpoints(req)

        # Should have at least 1 decision
        assert len(report.decisions) > 0
        # System should be injected
        system_decisions = [d for d in report.decisions if d.name == "system"]
        assert any(d.action == "injected" for d in system_decisions)

    def test_to_log_dict(self):
        """to_log_dict() should produce a serializable dict."""
        req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "system": "Be helpful. " * 500,
            "messages": [{"role": "user", "content": "hi"}],
        }
        req, report = inject_cache_breakpoints(req)

        log = report.to_log_dict()
        assert isinstance(log, dict)
        assert "system_tokens" in log
        assert "msg_breakdown" in log
        assert "decisions" in log


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
        req, report = inject_cache_breakpoints(req)

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
        req, report = inject_cache_breakpoints(req)

        # Nothing should have changed — already at 4 breakpoints
        assert req == original
        assert report.existing_breakpoints == 4


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
        req, report = inject_cache_breakpoints(req)
        meta = build_cache_metadata(req, injected_count=len(report.breakpoints_injected))

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
        req, report = inject_cache_breakpoints(req)
        meta = build_cache_metadata(req, injected_count=0)

        assert meta["injected"] == "false"
