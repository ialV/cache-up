"""Environment-based configuration using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings — all configurable via environment variables."""

    # Anthropic API (key can also come from request header)
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"

    # Server
    port: int = 8080
    host: str = "0.0.0.0"

    # Cache injection
    min_tokens_for_cache: int = 1024  # Anthropic minimum; Haiku needs 2048

    # Thinking defaults (for clients that can't send thinking param, e.g. OpenWebUI)
    # "true" = auto-detect (adaptive for 4.6, enabled+budget for older)
    # "adaptive" = force adaptive mode
    # "10000" = force enabled + budget_tokens=10000
    # "" = don't inject thinking
    default_thinking: str = "true"
    # effort: "low", "medium", "high" (default), "max" (Opus 4.6 only)
    default_effort: str = "high"

    # Logging
    log_level: str = "info"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
