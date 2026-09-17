"""Deterministic exports for teacher transcription outcomes."""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Dict, List

from src.transcriber import TranscriptionOutcome


CSV_COLUMNS = [
    "evaluation_id",
    "source_image",
    "source_kind",
    "source_hash",
    "page_number",
    "student_reference",
    "question",
    "model_transcription",
    "confidence",
    "uncertain",
    "uncertainty_note",
    "teacher_correction",
    "document_review_status",
    "item_review_status",
    "errors",
]


def _document_payload(outcome: TranscriptionOutcome) -> Dict[str, Any]:
    document = outcome.document
    payload: Dict[str, Any] = {
        "evaluation_id": document.evaluation_id if document else None,
        "source": document.source.source if document else None,
        "source_kind": document.source.source_kind if document else None,
        "source_hash": document.source.source_hash if document else None,
        "student_reference": document.student_reference if document else None,
        "review_status": document.review_status.value if document else None,
        "items": [],
        "errors": list(outcome.errors),
        "attempts": outcome.attempts,
        "dry_run": outcome.dry_run,
        "model": outcome.model,
        "prompt_version": outcome.prompt_version,
    }
    if document:
        payload["items"] = [
            {
                "question": item.item_id,
                "model_transcription": item.model_transcription,
                "confidence": item.confidence,
                "uncertain": item.uncertain,
                "uncertainty_note": item.uncertainty_note,
                "teacher_correction": item.teacher_correction,
                "item_review_status": item.review_status.value,
            }
            for item in document.items
        ]
    if outcome.dry_run:
        try:
            payload["prompt"] = json.loads(outcome.raw_text)
        except json.JSONDecodeError:
            payload["prompt"] = outcome.raw_text
    else:
        payload["raw_text"] = outcome.raw_text
    return payload


def documents_to_json(outcomes: List[TranscriptionOutcome]) -> str:
    """Serialize transcription outcomes as a stable, indented JSON array."""

    return json.dumps([_document_payload(outcome) for outcome in outcomes], indent=2)


def documents_to_csv(outcomes: List[TranscriptionOutcome]) -> str:
    """Serialize transcription items as deterministic flat CSV rows."""

    rows: List[Dict[str, Any]] = []
    for outcome in outcomes:
        document = outcome.document
        if not document or not document.items:
            rows.append(
                {
                    column: (";".join(outcome.errors) if column == "errors" else "")
                    for column in CSV_COLUMNS
                }
            )
            continue
        for item in document.items:
            rows.append(
                {
                    "evaluation_id": document.evaluation_id,
                    "source_image": document.source.source,
                    "source_kind": document.source.source_kind,
                    "source_hash": document.source.source_hash or "",
                    "page_number": document.source.page_number or "",
                    "student_reference": document.student_reference or "",
                    "question": item.item_id,
                    "model_transcription": item.model_transcription,
                    "confidence": item.confidence,
                    "uncertain": item.uncertain,
                    "uncertainty_note": item.uncertainty_note or "",
                    "teacher_correction": item.teacher_correction or "",
                    "document_review_status": document.review_status.value,
                    "item_review_status": item.review_status.value,
                    "errors": ";".join(outcome.errors),
                }
            )
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


__all__ = ["documents_to_csv", "documents_to_json"]
