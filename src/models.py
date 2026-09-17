"""Domain models for teacher-reviewed evaluation transcriptions."""

from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ReviewStatus(str, Enum):
    """Human review state for a transcription document or item."""

    NEEDS_REVIEW = "needs_review"
    REVIEWED = "reviewed"
    NEEDS_RESCAN = "needs_rescan"


class SourceProvenance(BaseModel):
    """Traceability metadata for the source image containing an evaluation."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    source_kind: Literal["path", "url", "base64"] = "path"
    source_hash: Optional[str] = None
    page_number: Optional[int] = Field(default=None, ge=1)


class TranscriptionSegment(BaseModel):
    """A contiguous portion of handwritten text transcribed by the model."""

    model_config = ConfigDict(extra="forbid")

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    uncertain: bool = False
    uncertainty_note: Optional[str] = None


class EvaluationItem(BaseModel):
    """One question or item and its human-reviewable transcription."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1)
    source: SourceProvenance
    model_transcription: str
    teacher_correction: Optional[str] = None
    segments: List[TranscriptionSegment] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    uncertain: bool = False
    uncertainty_note: Optional[str] = None
    review_status: ReviewStatus = ReviewStatus.NEEDS_REVIEW


class EvaluationDocument(BaseModel):
    """A single photographed or scanned evaluation for teacher review."""

    model_config = ConfigDict(extra="forbid")

    evaluation_id: str = Field(min_length=1)
    student_reference: Optional[str] = None
    source: SourceProvenance
    items: List[EvaluationItem] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.NEEDS_REVIEW
