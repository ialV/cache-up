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
                "cache_read_tokens": total_actual_cache_read,
                "cache_creation_tokens": total_actual_cache_create,
                "note": "From Anthropic response usage field (0 if not yet populated)",
            },
            "window_seconds": round(time.time() - records[0].timestamp) if records else 0,
        }


# Singleton
stats = Stats()
