"""
Integration tests for HTTP routes using FastAPI TestClient.

Tests cover:
  - /health endpoint
  - /savings endpoint
  - /v1/messages bypass header
  - Cache metadata response headers
  - Streaming SSE format preservation
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health & Savings
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_savings_returns_200(self, client):
        resp = client.get("/savings")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_requests" in data


# ---------------------------------------------------------------------------
# POST /v1/messages
# ---------------------------------------------------------------------------


class TestMessagesEndpoint:
    def test_invalid_json_returns_400(self, client):
        resp = client.post(
            "/v1/messages",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400

    @patch("app.routes.proxy")
    def test_non_streaming_with_cache_headers(self, mock_proxy, client):
        """Non-streaming request should return with X-Autocache-* headers."""
        # Mock the response from Anthropic
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = json.dumps({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 100, "output_tokens": 10},
        }).encode()
        mock_resp.headers = {}
        mock_proxy.forward_request = AsyncMock(return_value=mock_resp)

        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "system": "You are a helpful assistant. " * 200,
                "messages": [{"role": "user", "content": "Hello"}],
            },
            headers={
                "x-api-key": "sk-ant-test",
                "anthropic-version": "2023-06-01",
            },
        )

        assert resp.status_code == 200
        # Should have autocache headers
        assert resp.headers.get("x-autocache-injected") is not None

    @patch("app.routes.proxy")
    def test_bypass_header_skips_cache(self, mock_proxy, client):
        """X-Autocache-Bypass: true should skip cache injection."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = json.dumps({"id": "msg_test", "usage": {}}).encode()
        mock_resp.headers = {}
        mock_proxy.forward_request = AsyncMock(return_value=mock_resp)

        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "system": "System" * 1000,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={
                "x-api-key": "sk-ant-test",
                "x-autocache-bypass": "true",
            },
        )

        assert resp.status_code == 200
        assert resp.headers.get("x-autocache-injected") == "false"

    @patch("app.routes.proxy")
    def test_streaming_returns_event_stream(self, mock_proxy, client):
        """Streaming request should return text/event-stream content type."""

        async def mock_chunks():
            yield b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":10}}}\n\n'
            yield b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"Hi"}}\n\n'
            yield b"event: message_stop\ndata: {}\n\n"

        mock_proxy.forward_streaming = AsyncMock(
            return_value=(200, {"content-type": "text/event-stream"}, mock_chunks(), {})
        )

        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"x-api-key": "sk-ant-test"},
        )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        # Body should contain SSE events
        body = resp.content.decode()
        assert "message_start" in body
