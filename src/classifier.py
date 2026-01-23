"""Image classifier core."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from google import genai
from google.genai import types
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import AppConfig
from src.image_loader import LoadedImage
from src.prompts import PromptContext, PromptStrategy, parse_json_output


class TransientGeminiError(RuntimeError):
    """Transient Gemini error suitable for retry."""


class GeminiClientProtocol(Protocol):
    """Gemini client protocol for dependency injection."""

    def generate_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        image: LoadedImage,
        timeout_s: float,
    ) -> str:
        """Return model output as raw text."""


class GeminiClient:
    """Concrete Gemini client wrapper."""

    def __init__(self, api_key: str, logger: logging.Logger):
        self._client = genai.Client(api_key=api_key)
        self._logger = logger
        self._executor = ThreadPoolExecutor(max_workers=1)

    def generate_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        image: LoadedImage,
        timeout_s: float,
    ) -> str:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            system_instruction=system,
        )
        contents = [
            types.Part.from_bytes(data=image.data, mime_type=image.mime_type),
            user,
        ]

        def _call() -> str:
            response = self._client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return response.text or ""

        future = self._executor.submit(_call)
        try:
            return future.result(timeout=timeout_s)
        except Exception as exc:  # noqa: BLE001
            if _is_retryable_error(exc):
                self._logger.warning("Transient Gemini error", exc_info=exc)
                raise TransientGeminiError(str(exc)) from exc
            raise


class RateLimiter:
    """Simple sleep-based rate limiter."""

    def __init__(self, requests_per_minute: int):
        self._min_interval = 60.0 / requests_per_minute
        self._last_time: Optional[float] = None

    def acquire(self) -> None:
        now = time.monotonic()
        if self._last_time is None:
            self._last_time = now
            return
        elapsed = now - self._last_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_time = time.monotonic()


@dataclass(frozen=True)
class ClassificationResult:
    """Classification result payload."""

    prompt_id: str
    source: str
    source_kind: str
    model: str
    output: Dict[str, Any]
    raw_text: str
    errors: List[str]
    attempts: int
    dry_run: bool = False


class ImageClassifier:
    """Orchestrates prompt rendering, model calls, and validation."""

    def __init__(
        self,
        *,
        config: AppConfig,
        gemini: GeminiClientProtocol,
        prompt: PromptStrategy,
        rate_limiter: RateLimiter,
        logger: logging.Logger,
    ):
        self._config = config
        self._gemini = gemini
        self._prompt = prompt
        self._rate_limiter = rate_limiter
        self._logger = logger

    def classify_one(
        self,
        *,
        image: LoadedImage,
        ctx: PromptContext,
        dry_run: bool = False,
    ) -> ClassificationResult:
        rendered = self._prompt.render(ctx)
        if dry_run:
            return ClassificationResult(
                prompt_id=self._prompt.spec.id,
                source=image.source,
                source_kind=image.source_kind,
                model=self._config.gemini.model,
                output={"system": rendered["system"], "user": rendered["user"]},
                raw_text="",
                errors=[],
                attempts=0,
                dry_run=True,
            )

        attempt_counter = {"count": 0}

        @retry(
            reraise=True,
            stop=stop_after_attempt(self._config.retry.max_attempts),
            wait=wait_exponential(
                min=self._config.retry.min_backoff_s,
                max=self._config.retry.max_backoff_s,
            ),
            retry=retry_if_exception_type((TimeoutError, TransientGeminiError)),
            before_sleep=before_sleep_log(self._logger, logging.WARNING),
        )
        def _attempt() -> str:
            attempt_counter["count"] += 1
            self._rate_limiter.acquire()
            return self._gemini.generate_json(
                model=self._config.gemini.model,
                system=rendered["system"],
                user=rendered["user"],
                image=image,
                timeout_s=self._config.gemini.timeout_s,
            )

        raw_text = ""
        errors: List[str] = []
        output: Dict[str, Any] = {}
        try:
            raw_text = _attempt()
            output_obj = parse_json_output(raw_text)
            validation = self._prompt.validate_output(output_obj)
            if validation.ok:
                output = self._prompt.score(output_obj)
            else:
                errors.extend(validation.errors)
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid JSON: {exc}")
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

        return ClassificationResult(
            prompt_id=self._prompt.spec.id,
            source=image.source,
            source_kind=image.source_kind,
            model=self._config.gemini.model,
            output=output,
            raw_text=raw_text,
            errors=errors,
            attempts=attempt_counter["count"],
        )

    def classify_batch(
        self,
        *,
        images: List[LoadedImage],
        ctx: PromptContext,
        dry_run: bool = False,
    ) -> List[ClassificationResult]:
        results: List[ClassificationResult] = []
        for image in images:
            results.append(self.classify_one(image=image, ctx=ctx, dry_run=dry_run))
        return results


def _is_retryable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "429" in message or "503" in message or "rate" in message
