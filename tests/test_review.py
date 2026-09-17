import csv
import io

import pytest

from src.export import documents_to_csv
from src.models import EvaluationDocument, EvaluationItem, ReviewStatus, SourceProvenance
from src.transcriber import TranscriptionOutcome
from src.review import (
    ReviewFileError,
    apply_csv_corrections,
    load_document,
    load_documents,
    only_reviewed_document,
    pending_items_document,
    review_summary,
    save_document,
    save_documents,
)


def _document():
    source = SourceProvenance(source="page.png")
    return EvaluationDocument(
        evaluation_id="eval-1", source=source,
        items=[
            EvaluationItem(item_id="1", source=source, model_transcription="model one", confidence=.8),
            EvaluationItem(item_id="2", source=source, model_transcription="model two", confidence=.7,
                           teacher_correction="already", review_status=ReviewStatus.REVIEWED),
            EvaluationItem(item_id="3", source=source, model_transcription="model three", confidence=.6,
                           review_status=ReviewStatus.NEEDS_RESCAN),
        ],
    )


def test_save_load_round_trip(tmp_path):
    path = tmp_path / "nested" / "document.json"
    document = _document()
    save_document(document, path)
    assert load_document(path) == document


def test_missing_review_file_raises_clear_error(tmp_path):
    with pytest.raises(ReviewFileError, match="not found"):
        load_document(tmp_path / "missing.json")


def test_save_load_documents_round_trip(tmp_path):
    path = tmp_path / "nested" / "documents.json"
    documents = [_document(), _document().model_copy(update={"evaluation_id": "eval-2"})]
    save_documents(documents, path)
    assert load_documents(path) == documents


def test_invalid_documents_file_raises_clear_error(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("not json")
    with pytest.raises(ReviewFileError, match="Invalid"):
        load_documents(path)


def test_pending_items_document_counts_unreviewed_items():
    pending = pending_items_document(_document())
    assert [item.item_id for item in pending] == ["1", "3"]


def test_apply_csv_corrections_preserves_model_text():
    document = _document()
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["evaluation_id", "question", "teacher_correction", "item_review_status"])
    writer.writeheader()
    writer.writerows([
        {"evaluation_id": "eval-1", "question": "1", "teacher_correction": "teacher one", "item_review_status": "reviewed"},
        {"evaluation_id": "eval-1", "question": "2", "teacher_correction": "", "item_review_status": "bogus"},
        {"evaluation_id": "other", "question": "1", "teacher_correction": "wrong", "item_review_status": "reviewed"},
    ])
    result = apply_csv_corrections(document, output.getvalue())
    assert result.applied == 1
    assert result.skipped_rows == [3, 4]
    assert result.matched_document is True
    assert document.items[0].teacher_correction == "teacher one"
    assert document.items[0].review_status == ReviewStatus.REVIEWED
    assert document.items[0].model_transcription == "model one"


def test_export_import_preserves_mixed_item_statuses():
    document = _document()
    document.items[0].review_status = ReviewStatus.REVIEWED
    outcome = TranscriptionOutcome(document=document, raw_text="", errors=[], attempts=1,
                                   dry_run=False, model="test-model")
    restored = _document()
    result = apply_csv_corrections(restored, documents_to_csv([outcome]))
    assert result.applied == 3
    assert [item.review_status for item in restored.items] == [
        ReviewStatus.REVIEWED, ReviewStatus.REVIEWED, ReviewStatus.NEEDS_RESCAN,
    ]


def test_apply_csv_corrections_falls_back_to_legacy_review_status():
    document = _document()
    csv_text = (
        "evaluation_id,question,teacher_correction,review_status\n"
        "eval-1,1,,reviewed\n"
    )
    result = apply_csv_corrections(document, csv_text)
    assert result.applied == 1
    assert document.items[0].review_status == ReviewStatus.REVIEWED


def test_only_reviewed_document_preserves_order_and_fields():
    document = _document()
    filtered = only_reviewed_document(document)
    assert [item.item_id for item in filtered.items] == ["2"]
    assert filtered.items[0].teacher_correction == "already"
    assert filtered.items[0].model_transcription == "model two"
    assert filtered.review_status == document.review_status


def test_review_summary_counts_statuses_and_corrections():
    summary = review_summary(_document())
    assert summary == {
        "needs_review": 1, "reviewed": 1, "needs_rescan": 1,
        "total_items": 3, "teacher_corrected": 1, "review_status": "needs_review",
    }
