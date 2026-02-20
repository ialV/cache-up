"""
FastAPI application entry point.

Usage:
    uvicorn app.main:app --port 8080
    # or
    python -m uvicorn app.main:app --port 8080 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import logging

import structlog
import uvicorn

from fastapi import FastAPI

from app import proxy
from app.config import settings
from app.routes import router

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer() if settings.log_level == "debug" else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        _LOG_LEVELS.get(settings.log_level.lower(), logging.INFO)
    ),
)

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage httpx client lifecycle."""
    logger.info(
        "autocache.startup",
        version="0.1.0",
        port=settings.port,
        anthropic_url=settings.anthropic_base_url,
        api_key_configured=bool(settings.anthropic_api_key),
        min_tokens=settings.min_tokens_for_cache,
    )
    yield
    await proxy.close_client()
    logger.info("autocache.shutdown")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="autocache",
    description="Zero-config Claude prompt caching proxy",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )
