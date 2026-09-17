"""Tests for the teacher transcription domain contract."""

import pytest
from pydantic import ValidationError

from src.models import (
    EvaluationDocument,
    EvaluationItem,
    ReviewStatus,
    SourceProvenance,
    TranscriptionSegment,
)


@pytest.fixture
def source() -> SourceProvenance:
    return SourceProvenance(
        source="fixtures/evaluation-001.png",
        source_kind="path",
        source_hash="sha256:demo",
    )


def test_valid_document_contains_traceable_item(source: SourceProvenance) -> None:
    item = EvaluationItem(
        item_id="1",
        source=source,
        model_transcription="The water cycle",
        confidence=0.92,
        segments=[TranscriptionSegment(text="The water cycle", confidence=0.92)],
    )
    document = EvaluationDocument(
        evaluation_id="eval-demo-001",
        student_reference="anon-001",
        source=source,
        items=[item],
    )

    assert document.items[0].source.source == source.source
    assert document.items[0].review_status is ReviewStatus.NEEDS_REVIEW


def test_uncertain_content_is_explicit(source: SourceProvenance) -> None:
    item = EvaluationItem(
        item_id="2",
        source=source,
        model_transcription="photosynthesis?",
        confidence=0.35,
        uncertain=True,
        uncertainty_note="Handwriting is ambiguous",
        segments=[
            TranscriptionSegment(
                text="photosynthesis?",
                confidence=0.35,
                uncertain=True,
                uncertainty_note="Final word is unclear",
            )
        ],
    )

    assert item.uncertain is True
    assert item.segments[0].uncertainty_note == "Final word is unclear"


def test_teacher_correction_is_separate_from_model_transcription(
    source: SourceProvenance,
) -> None:
    item = EvaluationItem(
        item_id="3",
        source=source,
        model_transcription="evaporation",
        teacher_correction="evaporation from the surface",
        confidence=0.7,
    )

    assert item.model_transcription == "evaporation"
    assert item.teacher_correction != item.model_transcription


def test_review_status_defaults_and_accepts_explicit_states(
    source: SourceProvenance,
) -> None:
    item = EvaluationItem(
        item_id="4",
        source=source,
        model_transcription="Answer",
        confidence=1.0,
        review_status=ReviewStatus.REVIEWED,
    )

    assert item.review_status is ReviewStatus.REVIEWED
    assert EvaluationDocument(
        evaluation_id="eval-1", source=source, review_status="needs_rescan"
    ).review_status is ReviewStatus.NEEDS_RESCAN


def test_document_serializes_to_json_and_mapping(source: SourceProvenance) -> None:
    document = EvaluationDocument(
        evaluation_id="eval-demo-001",
        source=source,
        items=[
            EvaluationItem(
                item_id="1",
                source=source,
                model_transcription="Answer",
                confidence=0.8,
                teacher_correction="Corrected answer",
            )
        ],
    )

    payload = document.model_dump(mode="json")
    restored = EvaluationDocument.model_validate_json(document.model_dump_json())

    assert payload["items"][0]["review_status"] == "needs_review"
    assert payload["items"][0]["teacher_correction"] == "Corrected answer"
    assert restored == document


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_confidence_must_be_between_zero_and_one(
    source: SourceProvenance, confidence: float
) -> None:
    with pytest.raises(ValidationError):
        EvaluationItem(
            item_id="bad-confidence",
            source=source,
            model_transcription="Answer",
            confidence=confidence,
        )


def test_invalid_review_status_is_rejected(source: SourceProvenance) -> None:
    with pytest.raises(ValidationError):
        EvaluationItem(
            item_id="bad-status",
            source=source,
            model_transcription="Answer",
            confidence=0.5,
            review_status="approved",
        )
