import csv
import io
import json

from src.export import documents_to_csv, documents_to_json
from src.image_loader import LoadedImage
from src.models import EvaluationDocument, EvaluationItem, ReviewStatus, SourceProvenance, TranscriptionSegment
from src.transcriber import TranscriptionOutcome


def _outcome(*, dry_run=False, error=None):
    source = SourceProvenance(source="sample.ppm", source_kind="path", source_hash="abc123")
    document = None if dry_run or error else EvaluationDocument(
        evaluation_id="eval-1",
        student_reference="anon-001",
        source=source,
        items=[EvaluationItem(
            item_id="1", source=source, model_transcription="answer",
            confidence=0.8, uncertain=True, uncertainty_note="faded",
            teacher_correction="corrected", review_status=ReviewStatus.NEEDS_REVIEW,
            segments=[TranscriptionSegment(text="answer", confidence=0.8)],
        )],
    )
    return TranscriptionOutcome(
        document=document, raw_text='{"items": []}' if not dry_run else '{"system":"prompt"}',
        errors=[error] if error else [], attempts=1, dry_run=dry_run, model="test-model",
    )


def test_json_export_contains_items_and_provenance():
    payload = json.loads(documents_to_json([_outcome()]))[0]
    assert payload["evaluation_id"] == "eval-1"
    assert payload["source"] == "sample.ppm"
    assert payload["source_hash"] == "abc123"
    assert payload["items"][0]["teacher_correction"] == "corrected"
    assert payload["errors"] == []
    assert payload["raw_text"] == '{"items": []}'


def test_json_dry_run_contains_prompt_without_raw_text():
    payload = json.loads(documents_to_json([_outcome(dry_run=True)]))[0]
    assert payload["prompt"] == {"system": "prompt"}
    assert "raw_text" not in payload


def test_csv_header_and_row_do_not_leak_raw_text_or_grades():
    rows = list(csv.DictReader(io.StringIO(documents_to_csv([_outcome()]))))
    assert list(rows[0]) == [
        "evaluation_id", "source_image", "source_kind", "source_hash", "page_number",
        "student_reference", "question", "model_transcription", "confidence", "uncertain",
        "uncertainty_note", "teacher_correction", "document_review_status", "item_review_status", "errors",
    ]
    assert rows[0]["model_transcription"] == "answer"
    assert rows[0]["document_review_status"] == "needs_review"
    assert rows[0]["item_review_status"] == "needs_review"
    assert "raw_text" not in rows[0]
    assert "grade" not in rows[0]


def test_csv_round_trip_exports_item_status_separately_from_document_status():
    source = SourceProvenance(source="sample.ppm", source_kind="path", source_hash="abc123")
    document = EvaluationDocument(
        evaluation_id="eval-mixed",
        source=source,
        review_status=ReviewStatus.NEEDS_REVIEW,
        items=[
            EvaluationItem(item_id="1", source=source, model_transcription="one", confidence=.8,
                           review_status=ReviewStatus.REVIEWED),
            EvaluationItem(item_id="2", source=source, model_transcription="two", confidence=.7,
                           review_status=ReviewStatus.NEEDS_REVIEW),
        ],
    )
    outcome = TranscriptionOutcome(document=document, raw_text="", errors=[], attempts=1,
                                   dry_run=False, model="test-model")
    rows = list(csv.DictReader(io.StringIO(documents_to_csv([outcome]))))
    assert [row["item_review_status"] for row in rows] == ["reviewed", "needs_review"]
    assert {row["document_review_status"] for row in rows} == {"needs_review"}


def test_csv_error_outcome_has_one_error_row(): 
    rows = list(csv.DictReader(io.StringIO(documents_to_csv([_outcome(error="bad response")]))))
    assert len(rows) == 1
    assert rows[0]["errors"] == "bad response"
    assert rows[0]["source_image"] == ""
    assert rows[0]["question"] == ""
    assert rows[0]["document_review_status"] == ""
    assert rows[0]["item_review_status"] == ""
