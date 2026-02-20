"""
Tests for OpenAI ↔ Anthropic format translation.
"""

import json

from app.openai_compat import (
    anthropic_to_openai,
    openai_to_anthropic,
    translate_streaming_chunk,
)


class TestOpenAIToAnthropic:
    def test_basic_message(self):
        """Simple user message translates correctly."""
        req = {
            "model": "claude-3-5-haiku-20241022",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
        }
        result = openai_to_anthropic(req)
        assert result["model"] == "claude-3-5-haiku-20241022"
        assert result["max_tokens"] == 100
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello"
        assert "system" not in result

    def test_system_message_extracted(self):
        """System message moved from messages[] to system field."""
        req = {
            "model": "claude-3-5-haiku-20241022",
            "max_tokens": 100,
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }
        result = openai_to_anthropic(req)
        assert result["system"] == "You are helpful."
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_multiple_system_messages_merged(self):
        """Multiple system messages merged with double newline."""
        req = {
            "model": "test",
            "messages": [
                {"role": "system", "content": "Rule 1"},
                {"role": "system", "content": "Rule 2"},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = openai_to_anthropic(req)
        assert result["system"] == "Rule 1\n\nRule 2"

    def test_default_max_tokens(self):
        """Missing max_tokens defaults to 4096."""
        req = {"model": "test", "messages": [{"role": "user", "content": "Hi"}]}
        result = openai_to_anthropic(req)
        assert result["max_tokens"] == 4096

    def test_tool_calls_translated(self):
        """OpenAI tool_calls → Anthropic tool_use blocks."""
        req = {
            "model": "test",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Tokyo"}',
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": "Sunny, 25°C",
                },
                {"role": "user", "content": "Thanks!"},
            ],
        }
        result = openai_to_anthropic(req)
        # Assistant message should have text + tool_use blocks
        assistant_msg = result["messages"][1]
        assert assistant_msg["role"] == "assistant"
        assert isinstance(assistant_msg["content"], list)
        assert assistant_msg["content"][0]["type"] == "text"
        assert assistant_msg["content"][1]["type"] == "tool_use"
        assert assistant_msg["content"][1]["name"] == "get_weather"

        # Tool result should be mapped to user role with tool_result type
        tool_msg = result["messages"][2]
        assert tool_msg["role"] == "user"
        assert tool_msg["content"][0]["type"] == "tool_result"

    def test_optional_params_forwarded(self):
        """temperature, top_p, stop are forwarded."""
        req = {
            "model": "test",
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["END"],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = openai_to_anthropic(req)
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["stop_sequences"] == ["END"]

    def test_consecutive_roles_merged(self):
        """Consecutive same-role messages are merged for Anthropic."""
        req = {
            "model": "test",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "Fine!"},
            ],
        }
        result = openai_to_anthropic(req)
        # Two user messages should be merged into one
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"

    def test_tools_translated(self):
        """OpenAI tool definitions → Anthropic format."""
        req = {
            "model": "test",
            "max_tokens": 100,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = openai_to_anthropic(req)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "get_weather"
        assert result["tools"][0]["input_schema"]["type"] == "object"


class TestAnthropicToOpenAI:
    def test_basic_response(self):
        """Simple text response translates correctly."""
        resp = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-3-5-haiku-20241022",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = anthropic_to_openai(resp, "claude-3-5-haiku-20241022")
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15

    def test_cache_usage_forwarded(self):
        """Cache usage stats are included in OpenAI response."""
        resp = {
            "id": "msg_123",
            "content": [{"type": "text", "text": "Hi"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 5,
                "cache_read_input_tokens": 80,
                "cache_creation_input_tokens": 0,
            },
        }
        result = anthropic_to_openai(resp, "test")
        assert result["usage"]["cache_read_input_tokens"] == 80

    def test_tool_use_response(self):
        """Tool use blocks → OpenAI tool_calls format."""
        resp = {
            "id": "msg_123",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "test"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        result = anthropic_to_openai(resp, "test")
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        assert len(result["choices"][0]["message"]["tool_calls"]) == 1
        tc = result["choices"][0]["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "search"


class TestStreamingTranslation:
    def test_message_start(self):
        """message_start → OpenAI chunk with role."""
        event = 'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":10}}}'
        chunks = translate_streaming_chunk(event, "test", "chatcmpl-123")
        assert len(chunks) == 1
        data = json.loads(chunks[0].replace("data: ", "").strip())
        assert data["choices"][0]["delta"]["role"] == "assistant"

    def test_text_delta(self):
        """content_block_delta (text) → OpenAI content delta."""
        event = 'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}'
        chunks = translate_streaming_chunk(event, "test", "chatcmpl-123")
        assert len(chunks) == 1
        data = json.loads(chunks[0].replace("data: ", "").strip())
        assert data["choices"][0]["delta"]["content"] == "Hello"

    def test_message_stop(self):
        """message_stop → data: [DONE]."""
        event = 'event: message_stop\ndata: {"type":"message_stop"}'
        chunks = translate_streaming_chunk(event, "test", "chatcmpl-123")
        assert any("[DONE]" in c for c in chunks)
