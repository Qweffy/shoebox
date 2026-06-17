from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal
from pathlib import Path

from shoebox.ledger import to_ledger_row, write_ledger
from shoebox.models import OcrLine, OcrResult, ReceiptFields

_BBOX = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def _ocr() -> OcrResult:
    return OcrResult(
        source="/tmp/cafe-margarita.png",
        lines=[
            OcrLine(text="CAFE", confidence=0.9, bbox=_BBOX),
            OcrLine(text="TOTAL $7.00", confidence=0.8, bbox=_BBOX),
        ],
    )


def _fields() -> ReceiptFields:
    return ReceiptFields(
        vendor="CAFE", date=date(2026, 6, 12), total=Decimal("7.00"), currency="USD"
    )


def test_to_ledger_row_basename_and_confidence() -> None:
    row = to_ledger_row(_ocr(), _fields())
    assert row.source == "cafe-margarita.png"
    assert row.num_lines == 2
    assert row.min_confidence == 0.8
    assert abs(row.mean_confidence - 0.85) < 1e-9


def test_write_ledger_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "ledger.csv"
    write_ledger([to_ledger_row(_ocr(), _fields())], out)
    with out.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["source"] == "cafe-margarita.png"
    assert rows[0]["vendor"] == "CAFE"
    assert rows[0]["date"] == "2026-06-12"
    assert rows[0]["total"] == "7.00"
    assert rows[0]["currency"] == "USD"
    assert rows[0]["num_lines"] == "2"
