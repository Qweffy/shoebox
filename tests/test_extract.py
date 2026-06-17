from __future__ import annotations

from datetime import date
from decimal import Decimal

from shoebox.extract import RegexExtractor
from shoebox.models import OcrLine, OcrResult

_BBOX = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def _ocr(lines: list[str]) -> OcrResult:
    return OcrResult(
        source="receipt.png",
        lines=[OcrLine(text=text, confidence=0.99, bbox=_BBOX) for text in lines],
    )


def test_extracts_vendor_date_total_currency() -> None:
    ocr = _ocr(["CAFE MARGARITA", "Date: 2026-06-12", "Espresso $3.50", "TOTAL $7.00"])
    fields = RegexExtractor().extract(ocr)
    assert fields.vendor == "CAFE MARGARITA"
    assert fields.date == date(2026, 6, 12)
    assert fields.total == Decimal("7.00")
    assert fields.currency == "USD"


def test_total_prefers_labeled_over_largest_item() -> None:
    ocr = _ocr(["Item A $50.00", "Item B $9.99", "TOTAL $59.99"])
    assert RegexExtractor().extract(ocr).total == Decimal("59.99")


def test_total_falls_back_to_largest_when_unlabeled() -> None:
    ocr = _ocr(["Item A $5.00", "Item B $9.99"])
    assert RegexExtractor().extract(ocr).total == Decimal("9.99")


def test_parses_dd_mm_yyyy_date() -> None:
    fields = RegexExtractor().extract(_ocr(["SHOP", "12/06/2026", "TOTAL $1.00"]))
    assert fields.date == date(2026, 6, 12)


def test_handles_comma_decimal_amounts() -> None:
    assert RegexExtractor().extract(_ocr(["TIENDA", "TOTAL 1.234,56"])).total == Decimal("1234.56")


def test_returns_empty_fields_for_no_text() -> None:
    fields = RegexExtractor().extract(_ocr([]))
    assert fields.vendor is None
    assert fields.total is None
    assert fields.date is None
