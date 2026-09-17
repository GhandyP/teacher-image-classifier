from pathlib import Path

import pytest

from src.models import EvaluationDocument, EvaluationItem, ReviewStatus, SourceProvenance
from src.review import ReviewFileError
from src.review_ui import (
    apply_item_edit,
    documents_with_images,
    load_workspace,
    pending_documents,
    save_workspace,
    summary_lines,
)


def _document(evaluation_id: str = "eval-1", source: SourceProvenance | None = None):
    source = source or SourceProvenance(source="page.png")
    return EvaluationDocument(
        evaluation_id=evaluation_id,
        student_reference="anon-1",
        source=source,
        items=[
            EvaluationItem(item_id="1", source=source, model_transcription="model one", confidence=0.8),
            EvaluationItem(
                item_id="2", source=source, model_transcription="model two", confidence=0.7,
                teacher_correction="already", review_status=ReviewStatus.REVIEWED,
            ),
        ],
    )


def test_load_workspace_round_trip(tmp_path):
    path = tmp_path / "reviews" / "session.json"
    documents = [_document()]
    save_workspace(documents, path)
    assert load_workspace(path) == documents


def test_load_workspace_names_file_in_error(tmp_path):
    path = tmp_path / "missing.json"
    with pytest.raises(ReviewFileError, match="missing.json"):
        load_workspace(path)


def test_documents_with_images_pairs_existing_and_unavailable_sources(tmp_path):
    image = tmp_path / "page.ppm"
    image.write_bytes(b"synthetic image")
    existing = _document(source=SourceProvenance(source=str(image)))
    missing = _document("eval-2", SourceProvenance(source=str(tmp_path / "gone.ppm")))
    remote = _document("eval-3", SourceProvenance(source="https://example.test/page.png", source_kind="url"))

    paired = documents_with_images([existing, missing, remote])
    assert paired[0][1] == image.resolve()
    assert paired[1][1] is None
    assert paired[2][1] is None


def test_pending_documents_preserves_order():
    reviewed = _document("reviewed")
    reviewed.items[0].review_status = ReviewStatus.REVIEWED
    pending = _document("pending")
    assert [document.evaluation_id for document in pending_documents([reviewed, pending])] == ["pending"]


def test_apply_item_edit_copies_and_updates_item():
    document = _document()
    edited = apply_item_edit(
        document, "1", teacher_correction="teacher text", review_status="reviewed"
    )
    assert edited is not document
    assert edited.items[0].teacher_correction == "teacher text"
    assert edited.items[0].review_status is ReviewStatus.REVIEWED
    assert document.items[0].teacher_correction is None
    assert document.items[0].review_status is ReviewStatus.NEEDS_REVIEW


def test_apply_item_edit_empty_correction_clears_value():
    document = _document()
    edited = apply_item_edit(document, "2", teacher_correction="")
    assert edited.items[1].teacher_correction == ""
    assert document.items[1].teacher_correction == "already"


def test_apply_item_edit_rejects_unknown_item_and_invalid_status():
    document = _document()
    with pytest.raises(ValueError, match="Unknown"):
        apply_item_edit(document, "missing")
    with pytest.raises(ValueError, match="Invalid"):
        apply_item_edit(document, "1", review_status="not-a-status")


def test_summary_lines_are_human_readable():
    assert summary_lines([_document()]) == [
        "eval-1: 2 items · 1 reviewed · 1 needs_review · 1 corrected"
    ]


def test_ui_app_import_does_not_require_streamlit():
    import ui.app as app

    assert callable(app.main)
