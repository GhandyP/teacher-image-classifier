"""Teacher-reviewed handwritten evaluation transcription service."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.classifier import (
    GeminiClient,
    GeminiClientProtocol,
    RateLimiter,
    TransientGeminiError,
)
from src.config import AppConfig
from src.image_loader import LoadedImage
from src.models import (
    EvaluationDocument,
    EvaluationItem,
    ReviewStatus,
    SourceProvenance,
    TranscriptionSegment,
)
from src.prompts import (
    TRANSCRIPTION_PROMPT_VERSION,
    parse_json_output,
    transcription_prompt,
)


class RawItem(BaseModel):
    """One raw item returned by the transcription model."""

    model_config = ConfigDict(extra="forbid")

    question_label: str = Field(min_length=1)
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    uncertain: bool = False
    uncertainty_note: Optional[str] = None


class RawTranscription(BaseModel):
    """Strict raw JSON contract returned by the model."""

    model_config = ConfigDict(extra="forbid")

    items: List[RawItem]
    student_reference: Optional[str] = None


class TranscriptionError(ValueError):
    """User-facing transcription failure."""


class TranscriptionSchemaError(TranscriptionError):
    """Model output does not match the transcription schema."""


@dataclass(frozen=True)
class TranscriptionOutcome:
    """Result of transcribing one image."""

    document: Optional[EvaluationDocument]
    raw_text: str
    errors: List[str]
    attempts: int
    dry_run: bool
    model: str
    prompt_version: str = TRANSCRIPTION_PROMPT_VERSION


class Transcriber:
    """Orchestrates prompt rendering, model calls, and transcription mapping."""

    def __init__(
        self,
        *,
        config: AppConfig,
        gemini: GeminiClientProtocol,
        rate_limiter: RateLimiter,
        logger: logging.Logger,
    ) -> None:
        self._config = config
        self._gemini = gemini
        self._rate_limiter = rate_limiter
        self._logger = logger

    def transcribe_one(
        self, *, image: LoadedImage, dry_run: bool = False
    ) -> TranscriptionOutcome:
        rendered = transcription_prompt()
        if dry_run:
            return TranscriptionOutcome(
                document=None,
                raw_text=json.dumps(rendered),
                errors=[],
                attempts=0,
                dry_run=True,
                model=self._config.gemini.model,
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
        document: Optional[EvaluationDocument] = None
        try:
            raw_text = _attempt()
            try:
                raw_output = parse_json_output(raw_text)
            except json.JSONDecodeError as exc:
                raise TranscriptionSchemaError(
                    _json_schema_error(exc)
                ) from exc
            except TypeError as exc:
                raise TranscriptionSchemaError(
                    f"Invalid transcription response: {type(exc).__name__}"
                ) from exc
            try:
                parsed = RawTranscription.model_validate(raw_output)
            except ValidationError as exc:
                raise TranscriptionSchemaError(
                    _validation_schema_error(exc)
                ) from exc
            document = self._to_document(image=image, raw=parsed)
        except TranscriptionError as exc:
            errors.append(_safe_error_message(str(exc), raw_text))
        except Exception as exc:  # noqa: BLE001
            errors.append(_safe_error_message(str(exc), raw_text))

        return TranscriptionOutcome(
            document=document,
            raw_text=raw_text,
            errors=errors,
            attempts=attempt_counter["count"],
            dry_run=False,
            model=self._config.gemini.model,
        )

    def transcribe_batch(
        self, *, images: List[LoadedImage], dry_run: bool = False
    ) -> List[TranscriptionOutcome]:
        """Transcribe images sequentially."""

        return [self.transcribe_one(image=image, dry_run=dry_run) for image in images]

    @staticmethod
    def _to_document(*, image: LoadedImage, raw: RawTranscription) -> EvaluationDocument:
        source = SourceProvenance(
            source=image.source,
            source_kind=image.source_kind,
            source_hash=image.sha256,
        )
        items = [
            EvaluationItem(
                item_id=item.question_label,
                source=source,
                model_transcription=item.text,
                segments=[
                    TranscriptionSegment(
                        text=item.text,
                        confidence=item.confidence,
                        uncertain=item.uncertain,
                        uncertainty_note=item.uncertainty_note,
                    )
                ],
                confidence=item.confidence,
                uncertain=item.uncertain,
                uncertainty_note=item.uncertainty_note,
                review_status=ReviewStatus.NEEDS_REVIEW,
            )
            for item in raw.items
        ]
        return EvaluationDocument(
            evaluation_id=f"eval-{image.sha256[:12] if image.sha256 else 'unknown'}",
            student_reference=raw.student_reference,
            source=source,
            items=items,
            review_status=ReviewStatus.NEEDS_REVIEW,
        )


def _json_schema_error(exc: json.JSONDecodeError) -> str:
    """Describe JSON syntax failures without including source text."""

    return _cap_error(
        "Invalid transcription response: "
        f"{type(exc).__name__} at line {exc.lineno}, column {exc.colno} "
        f"(position {exc.pos})"
    )


def _validation_schema_error(exc: ValidationError) -> str:
    """Summarize validation failures using locations and machine types only."""

    summaries = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
        summaries.append(f"{error.get('type', 'validation_error')}@{location}")
    detail = ", ".join(summaries) or "validation_error@<root>"
    return _cap_error(f"Invalid transcription response: ValidationError: {detail}")


def _safe_error_message(message: str, raw_text: str) -> str:
    """Keep outcome diagnostics short and never echo the complete model payload."""

    if raw_text:
        message = message.replace(raw_text, "[REDACTED]")
    return _cap_error(message)


def _cap_error(message: str) -> str:
    return message[:300]


__all__ = [
    "GeminiClient",
    "GeminiClientProtocol",
    "RateLimiter",
    "RawItem",
    "RawTranscription",
    "TRANSCRIPTION_PROMPT_VERSION",
    "Transcriber",
    "TranscriptionError",
    "TranscriptionOutcome",
    "TranscriptionSchemaError",
    "TransientGeminiError",
]
