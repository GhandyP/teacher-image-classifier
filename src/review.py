"""Local persistence and spreadsheet review helpers."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from pydantic import ValidationError

from src.models import EvaluationDocument, EvaluationItem, ReviewStatus


class ReviewFileError(ValueError):
    """A review document could not be read or validated."""


def save_document(document: EvaluationDocument, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document.model_dump_json(indent=2), encoding="utf-8")


def load_document(path: Path) -> EvaluationDocument:
    if not path.exists():
        raise ReviewFileError(f"Review document not found: {path}")
    try:
        return EvaluationDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ReviewFileError(f"Invalid review document {path}: {exc}") from exc


def save_documents(documents: List[EvaluationDocument], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [document.model_dump(mode="json") for document in documents]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_documents(path: Path) -> List[EvaluationDocument]:
    if not path.exists():
        raise ReviewFileError(f"Review documents not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("review documents JSON must be an array")
        return [EvaluationDocument.model_validate(entry) for entry in payload]
    except (OSError, ValueError, TypeError, ValidationError) as exc:
        raise ReviewFileError(f"Invalid review documents {path}: {exc}") from exc


def pending_items_document(document: EvaluationDocument) -> List[EvaluationItem]:
    return [item for item in document.items if item.review_status is not ReviewStatus.REVIEWED]


@dataclass(frozen=True)
class ReviewCsvResult:
    applied: int
    skipped_rows: List[int]
    matched_document: bool


def apply_csv_corrections(document: EvaluationDocument, csv_text: str) -> ReviewCsvResult:
    reader = csv.DictReader(io.StringIO(csv_text))
    items = {item.item_id: item for item in document.items}
    applied = 0
    skipped_rows: List[int] = []
    matched_document = False
    valid_statuses = {status.value for status in ReviewStatus}

    for row_number, row in enumerate(reader, start=2):
        if row.get("evaluation_id", "") != document.evaluation_id:
            skipped_rows.append(row_number)
            continue
        matched_document = True
        item = items.get(row.get("question", ""))
        if item is None:
            skipped_rows.append(row_number)
            continue
        changed = False
        correction = row.get("teacher_correction", "")
        if correction:
            item.teacher_correction = correction
            changed = True
        status = row.get("item_review_status")
        if status is None:
            status = row.get("review_status", "")
        if status in valid_statuses:
            item.review_status = ReviewStatus(status)
            changed = True
        else:
            skipped_rows.append(row_number)
        if changed:
            applied += 1

    return ReviewCsvResult(applied, skipped_rows, matched_document)


def only_reviewed_document(document: EvaluationDocument) -> EvaluationDocument:
    filtered = document.model_copy(deep=True)
    filtered.items = [item for item in filtered.items if item.review_status == ReviewStatus.REVIEWED]
    return filtered


def review_summary(document: EvaluationDocument) -> Dict[str, Any]:
    counts = {status.value: 0 for status in ReviewStatus}
    teacher_corrected = 0
    for item in document.items:
        counts[item.review_status.value] += 1
        if item.teacher_correction:
            teacher_corrected += 1
    return {
        **counts,
        "total_items": len(document.items),
        "teacher_corrected": teacher_corrected,
        "review_status": document.review_status.value,
    }


__all__ = [
    "ReviewCsvResult",
    "ReviewFileError",
    "apply_csv_corrections",
    "load_document",
    "load_documents",
    "only_reviewed_document",
    "pending_items_document",
    "review_summary",
    "save_document",
    "save_documents",
]
