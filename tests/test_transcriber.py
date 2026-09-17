"""Transcription service tests."""

import json
import logging
from typing import List

from src.classifier import RateLimiter
from src.config import AppConfig, GeminiConfig, RetryConfig
from src.image_loader import LoadedImage
from src.transcriber import Transcriber


class FakeGemini:
    def __init__(self, responses: List[object]):
        self.responses = responses
        self.calls = 0

    def generate_json(self, **kwargs: object) -> str:
        self.calls += 1
        response = self.responses[self.calls - 1]
        if isinstance(response, Exception):
            raise response
        return str(response)


def _image() -> LoadedImage:
    return LoadedImage(
        data=b"image",
        mime_type="image/png",
        source="tests/fixtures/sample.ppm",
        source_kind="path",
        sha256="0123456789abcdef0123456789abcdef",
    )


def _build(fake: FakeGemini) -> Transcriber:
    config = AppConfig(
        gemini=GeminiConfig(api_key="test"),
        retry=RetryConfig(max_attempts=3, min_backoff_s=0.001, max_backoff_s=0.002),
    )
    return Transcriber(
        config=config,
        gemini=fake,
        rate_limiter=RateLimiter(6000),
        logger=logging.getLogger("test.transcriber"),
    )


def _response(**item: object) -> str:
    payload = {
        "student_reference": "anon-001",
        "items": [
            {
                "question_label": "1",
                "text": "The literal answer.",
                "confidence": 0.8,
                "uncertain": False,
                "uncertainty_note": None,
                **item,
            }
        ],
    }
    return json.dumps(payload)


def test_valid_output_maps_to_reviewable_document() -> None:
    fake = FakeGemini([_response()])
    result = _build(fake).transcribe_one(image=_image())

    assert result.errors == []
    assert result.document is not None
    assert result.document.evaluation_id == "eval-0123456789ab"
    assert result.document.source.source_hash == _image().sha256
    assert result.document.items[0].item_id == "1"
    assert result.document.items[0].review_status.value == "needs_review"


def test_unclear_item_is_flagged() -> None:
    fake = FakeGemini([_response(text="[unclear]", confidence=0.2, uncertain=True, uncertainty_note="Faded")])
    item = _build(fake).transcribe_one(image=_image()).document.items[0]  # type: ignore[union-attr]

    assert item.uncertain is True
    assert item.segments[0].uncertain is True
    assert item.uncertainty_note == "Faded"


def test_malformed_json_returns_error_without_raising() -> None:
    result = _build(FakeGemini(["not json"])).transcribe_one(image=_image())

    assert result.document is None
    assert "Invalid transcription response" in result.errors[0]


def test_invalid_confidence_returns_schema_error() -> None:
    result = _build(FakeGemini([_response(confidence=1.1)])).transcribe_one(image=_image())

    assert result.document is None
    assert "Invalid transcription response" in result.errors[0]
    assert "ValidationError" in result.errors[0]
    assert "less than or equal to 1" not in result.errors[0]


def test_extra_field_error_redacts_student_text() -> None:
    marker = "STUDENT_PRIVATE_MARKER_7f3a"
    result = _build(
        FakeGemini([_response(text=marker, extra_key="unexpected")])
    ).transcribe_one(image=_image())

    assert result.document is None
    assert "ValidationError" in result.errors[0]
    assert marker not in result.errors[0]
    assert "extra_forbidden@items.0.extra_key" in result.errors[0]


def test_unknown_key_returns_schema_error() -> None:
    result = _build(FakeGemini([_response(extra_key="unexpected")])).transcribe_one(image=_image())

    assert result.document is None
    assert "extra_key" in result.errors[0]


def test_json_syntax_error_redacts_raw_payload() -> None:
    marker = "STUDENT_PRIVATE_MARKER_json"
    result = _build(FakeGemini([f'{{"text": "{marker}"'])).transcribe_one(image=_image())

    assert result.document is None
    assert "JSONDecodeError" in result.errors[0]
    assert marker not in result.errors[0]


def test_timeout_retries_then_succeeds() -> None:
    fake = FakeGemini([TimeoutError("timeout"), _response()])
    result = _build(fake).transcribe_one(image=_image())

    assert fake.calls == 2
    assert result.attempts == 2
    assert result.document is not None


def test_dry_run_returns_prompts_without_model_call() -> None:
    fake = FakeGemini([])
    result = _build(fake).transcribe_one(image=_image(), dry_run=True)

    assert result.dry_run is True
    assert result.attempts == 0
    assert fake.calls == 0
    assert "literal" in result.raw_text.lower()
    assert "grade" in result.raw_text.lower()


def test_document_never_contains_grade_or_score_fields() -> None:
    result = _build(FakeGemini([_response()])).transcribe_one(image=_image())
    dumped = result.document.model_dump()  # type: ignore[union-attr]

    assert "grade" not in dumped
    assert "score" not in dumped
    assert "grade" not in json.dumps(dumped).lower()
    assert "score" not in json.dumps(dumped).lower()
