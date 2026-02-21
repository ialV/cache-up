"""
In-memory observability — request stats and cache hit tracking.

Keeps a bounded ring buffer of recent request metadata for the /savings endpoint.
No external dependencies (no Redis, no SQLite — Phase 1 keeps it simple).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class RequestRecord:
    """Record of a single proxied request."""

    timestamp: float
    model: str
    estimated_total_tokens: int
    estimated_cached_tokens: int
    breakpoints_injected: int
    # Actual usage from Anthropic response (when available)
    actual_input_tokens: int = 0
    actual_cache_read_tokens: int = 0
    actual_cache_creation_tokens: int = 0
    streaming: bool = False
    # Extended diagnostics (Phase 1.5)
    system_tokens: int = 0
    system_chars: int = 0
    tool_tokens: int = 0
    prefix_tokens: int = 0
    message_count: int = 0
    message_breakdown: list[dict] = field(default_factory=list)
    injection_decisions: list[dict] = field(default_factory=list)
    breakpoint_positions: list[str] = field(default_factory=list)


@dataclass
class Stats:
    """Thread-safe in-memory stats collector."""

    _history: deque[RequestRecord] = field(default_factory=lambda: deque(maxlen=200))
    _lock: Lock = field(default_factory=Lock)

    def record(self, rec: RequestRecord):
        with self._lock:
            self._history.append(rec)

    def get_summary(self) -> dict:
        with self._lock:
            records = list(self._history)

        if not records:
            return {
                "total_requests": 0,
                "requests_with_cache": 0,
                "message": "No requests recorded yet",
            }

        total = len(records)
        with_cache = sum(1 for r in records if r.breakpoints_injected > 0)
        total_est_tokens = sum(r.estimated_total_tokens for r in records)
        total_est_cached = sum(r.estimated_cached_tokens for r in records)
        total_actual_input = sum(r.actual_input_tokens for r in records)
        total_actual_cache_read = sum(r.actual_cache_read_tokens for r in records)
        total_actual_cache_create = sum(r.actual_cache_creation_tokens for r in records)

        return {
            "total_requests": total,
            "requests_with_cache": with_cache,
            "estimated": {
                "total_tokens": total_est_tokens,
                "cached_tokens": total_est_cached,
                "cache_ratio": round(total_est_cached / total_est_tokens, 3) if total_est_tokens else 0,
            },
            "actual_anthropic": {
                "input_tokens": total_actual_input,
                "cache_read_tokens": total_actual_cache_read,
                "cache_creation_tokens": total_actual_cache_create,
                "cache_ratio": (
                    round(total_actual_cache_read / total_actual_input, 3)
                    if total_actual_input else 0
                ),
            },
            "window_seconds": round(time.time() - records[0].timestamp) if records else 0,
        }

    def get_recent(self, n: int = 10) -> list[dict]:
        """Return the last N requests with full diagnostics."""
        with self._lock:
            records = list(self._history)

        recent = records[-n:] if len(records) > n else records
        result = []
        for r in recent:
            entry = {
                "timestamp": round(r.timestamp, 1),
                "model": r.model,
                "streaming": r.streaming,
                "estimated": {
                    "total_tokens": r.estimated_total_tokens,
                    "cached_tokens": r.estimated_cached_tokens,
                    "system_tokens": r.system_tokens,
                    "system_chars": r.system_chars,
                    "tool_tokens": r.tool_tokens,
                    "prefix_tokens": r.prefix_tokens,
                },
                "messages": {
                    "count": r.message_count,
                    "breakdown": r.message_breakdown,
                },
                "breakpoints": {
                    "injected": r.breakpoints_injected,
                    "positions": r.breakpoint_positions,
                    "decisions": r.injection_decisions,
                },
                "actual_anthropic": {
                    "input_tokens": r.actual_input_tokens,
                    "cache_read_tokens": r.actual_cache_read_tokens,
                    "cache_creation_tokens": r.actual_cache_creation_tokens,
                },
            }
            result.append(entry)
        return result


# Singleton
stats = Stats()
