"""Pure helpers for the local teacher review UI."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from src.models import EvaluationDocument, ReviewStatus
from src.review import ReviewFileError, load_documents, review_summary, save_documents


def load_workspace(path: Path) -> List[EvaluationDocument]:
    """Load a session workspace and identify its file in any error."""

    try:
        return load_documents(path)
    except ReviewFileError as exc:
        raise ReviewFileError(f"Unable to load workspace file {path}: {exc}") from exc


def documents_with_images(
    documents: List[EvaluationDocument],
) -> List[Tuple[EvaluationDocument, Optional[Path]]]:
    """Pair each document with its resolved local source image, when available."""

    paired: List[Tuple[EvaluationDocument, Optional[Path]]] = []
    for document in documents:
        image_path: Optional[Path] = None
        if document.source.source_kind == "path":
            candidate = Path(document.source.source).expanduser().resolve()
            if candidate.exists():
                image_path = candidate
        paired.append((document, image_path))
    return paired


def pending_documents(documents: List[EvaluationDocument]) -> List[EvaluationDocument]:
    """Return documents containing at least one item not yet reviewed."""

    return [
        document
        for document in documents
        if any(item.review_status is not ReviewStatus.REVIEWED for item in document.items)
    ]


def apply_item_edit(
    document: EvaluationDocument,
    item_id: str,
    *,
    teacher_correction: Optional[str] = None,
    review_status: Optional[str] = None,
) -> EvaluationDocument:
    """Apply one UI edit to a deep copy without changing the source document."""

    edited = document.model_copy(deep=True)
    item = next((candidate for candidate in edited.items if candidate.item_id == item_id), None)
    if item is None:
        raise ValueError(f"Unknown evaluation item: {item_id}")

    if teacher_correction is not None:
        item.teacher_correction = teacher_correction
    if review_status is not None:
        try:
            item.review_status = ReviewStatus(review_status)
        except ValueError as exc:
            raise ValueError(f"Invalid review status: {review_status}") from exc
    return edited


def save_workspace(documents: List[EvaluationDocument], path: Path) -> None:
    """Persist the complete workspace through the existing review storage contract."""

    save_documents(documents, path)


def filter_documents(
    documents: List[EvaluationDocument],
    status_filter: Optional[str] = None,
    only_pending: bool = False,
) -> List[EvaluationDocument]:
    """Filter documents by document status and/or pending item state."""

    filtered = pending_documents(documents) if only_pending else list(documents)
    if status_filter is None or status_filter == "all":
        return filtered
    try:
        status = ReviewStatus(status_filter)
    except ValueError as exc:
        raise ValueError(f"Invalid document status: {status_filter}") from exc
    return [document for document in filtered if document.review_status is status]


def summary_lines(documents: List[EvaluationDocument]) -> List[str]:
    """Render one compact progress summary for each document."""

    lines: List[str] = []
    for document in documents:
        summary = review_summary(document)
        statuses = [
            f"{summary[status]} {status}"
            for status in (ReviewStatus.REVIEWED.value, ReviewStatus.NEEDS_REVIEW.value, ReviewStatus.NEEDS_RESCAN.value)
            if summary[status]
        ]
        parts = [f"{document.evaluation_id}: {summary['total_items']} items", *statuses]
        parts.append(f"{summary['teacher_corrected']} corrected")
        lines.append(" · ".join(parts))
    return lines


__all__ = [
    "apply_item_edit",
    "documents_with_images",
    "filter_documents",
    "load_workspace",
    "pending_documents",
    "save_workspace",
    "summary_lines",
]
