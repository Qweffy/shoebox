"""Typed data models for the OCR ledger pipeline.

These pydantic models are the parse-before-trust boundary: raw OpenCV / PaddleOCR
output is converted into validated models at the edge, so the rest of the codebase
never touches untyped third-party structures.
"""

from __future__ import annotations

import datetime
from decimal import Decimal

from pydantic import BaseModel, Field

Point = tuple[float, float]


class OcrLine(BaseModel):
    """A single recognized text line with its confidence and polygon."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[Point]


class OcrResult(BaseModel):
    """All recognized lines for one source image."""

    source: str
    lines: list[OcrLine]

    @property
    def full_text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def mean_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return sum(line.confidence for line in self.lines) / len(self.lines)

    @property
    def min_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return min(line.confidence for line in self.lines)


class ReceiptFields(BaseModel):
    """Fields extracted from the OCR text by an Extractor."""

    vendor: str | None = None
    date: datetime.date | None = None
    date_text: str | None = None
    total: Decimal | None = None
    currency: str | None = None


class LedgerRow(BaseModel):
    """One row in the output CSV ledger."""

    source: str
    vendor: str | None
    date: datetime.date | None
    total: Decimal | None
    currency: str | None
    mean_confidence: float
    min_confidence: float
    num_lines: int
