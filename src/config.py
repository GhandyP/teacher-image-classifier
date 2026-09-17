"""Configuration and logging utilities."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator


class RetryConfig(BaseModel):
    """Retry configuration."""

    max_attempts: int = Field(default=5, ge=1, le=20)
    min_backoff_s: float = Field(default=0.5, gt=0.0)
    max_backoff_s: float = Field(default=8.0, gt=0.0)

    @field_validator("max_backoff_s")
    @classmethod
    def validate_backoff(cls, value: float, info: Any) -> float:
        min_backoff = info.data.get("min_backoff_s", 0.0)
        if value < min_backoff:
            raise ValueError("max_backoff_s must be >= min_backoff_s")
        return value


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""

    requests_per_minute: int = Field(default=60, ge=1, le=6000)


class GeminiConfig(BaseModel):
    """Gemini client configuration."""

    api_key: str = Field(min_length=1)
    model: str = Field(default="gemini-2.5-flash")
    timeout_s: float = Field(default=30.0, gt=0.0, le=120.0)


class ScoringConfig(BaseModel):
    """Scoring configuration."""

    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    weights: Dict[str, float] = Field(default_factory=lambda: {"overall": 1.0})


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO")
    json: bool = Field(default=True)


class ImageConfig(BaseModel):
    """Image ingestion configuration."""

    max_bytes: int = Field(default=10 * 1024 * 1024, ge=1024, le=100 * 1024 * 1024)
    allow_remote_urls: bool = False


class AppConfig(BaseModel):
    """Top-level application configuration."""

    gemini: GeminiConfig
    retry: RetryConfig = RetryConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()
    scoring: ScoringConfig = ScoringConfig()
    logging: LoggingConfig = LoggingConfig()
    image: ImageConfig = ImageConfig()


@dataclass(frozen=True)
class ConfigPaths:
    """Config file paths."""

    yaml_path: Optional[Path] = None


class ConfigError(ValueError):
    """Configuration error."""


class JsonFormatter(logging.Formatter):
    """Structured JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str, json_logs: bool) -> None:
    """Configure structured logging for the application.

    Args:
        level: Logging level string.
        json_logs: Whether to emit JSON logs.
    """

    handler = logging.StreamHandler()
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


def _load_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def load_config(
    *,
    paths: ConfigPaths,
    overrides: Optional[Dict[str, Any]] = None,
) -> AppConfig:
    """Load configuration from env, YAML, and overrides.

    Args:
        paths: Config file paths.
        overrides: CLI overrides to apply last.

    Returns:
        Validated AppConfig.
    """

    load_dotenv()

    yaml_data = _load_yaml(paths.yaml_path)
    env_data: Dict[str, Any] = {
        "gemini": {
            "api_key": os.getenv("GEMINI_API_KEY", ""),
            "model": os.getenv("GEMINI_MODEL"),
            "timeout_s": _to_float(os.getenv("GEMINI_TIMEOUT_S")),
        },
        "rate_limit": {"requests_per_minute": _to_int(os.getenv("GEMINI_RPM"))},
        "retry": {
            "max_attempts": _to_int(os.getenv("RETRY_MAX_ATTEMPTS")),
            "min_backoff_s": _to_float(os.getenv("RETRY_MIN_BACKOFF_S")),
            "max_backoff_s": _to_float(os.getenv("RETRY_MAX_BACKOFF_S")),
        },
        "logging": {
            "level": os.getenv("LOG_LEVEL"),
            "json": _to_bool(os.getenv("LOG_JSON")),
        },
        "image": {"max_bytes": _to_int(os.getenv("IMAGE_MAX_BYTES"))},
    }

    merged = _deep_merge(yaml_data, env_data)
    if overrides:
        merged = _deep_merge(merged, overrides)

    try:
        return AppConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _to_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def _to_bool(value: Optional[str]) -> Optional[bool]:
    if value is None or value == "":
        return None
    return value.strip().lower() in {"1", "true", "yes", "y"}
