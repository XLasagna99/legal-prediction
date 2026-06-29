"""Request and response models for the prediction API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class CaseInput(BaseModel):
    """Pre-decision information about a case.

    Only include information that would exist *before* the case is decided.
    `text` is the facts/complaint/brief; the metadata fields mirror what the
    model was trained on.
    """

    text: str = Field(..., min_length=1, description="Pre-decision case text")
    court: str | None = Field(default=None)
    case_type: str | None = Field(default=None)
    represented: str | None = Field(default=None)


class PredictionResponse(BaseModel):
    label: int = Field(..., description="Predicted class (0/1)")
    probability: float = Field(..., description="P(label == 1)")
    confidence: float = Field(..., description="0 at the decision boundary, 1 at the extremes")
    model_version: str
    disclaimer: str


class ModelInfoResponse(BaseModel):
    version: str
    trained_at: str
    metrics: dict
