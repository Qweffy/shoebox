"""Assemble pipeline outputs into a CSV ledger."""

from __future__ import annotations

import csv
from pathlib import Path

from shoebox.models import LedgerRow, OcrResult, ReceiptFields

_FIELDNAMES = (
    "source",
    "vendor",
    "date",
    "total",
    "currency",
    "mean_confidence",
    "min_confidence",
    "num_lines",
)


def to_ledger_row(ocr: OcrResult, fields: ReceiptFields) -> LedgerRow:
    return LedgerRow(
        source=Path(ocr.source).name,
        vendor=fields.vendor,
        date=fields.date,
        total=fields.total,
        currency=fields.currency,
        mean_confidence=round(ocr.mean_confidence, 4),
        min_confidence=round(ocr.min_confidence, 4),
        num_lines=len(ocr.lines),
    )


def _row_to_dict(row: LedgerRow) -> dict[str, str]:
    return {
        "source": row.source,
        "vendor": row.vendor or "",
        "date": row.date.isoformat() if row.date is not None else "",
        "total": f"{row.total:.2f}" if row.total is not None else "",
        "currency": row.currency or "",
        "mean_confidence": f"{row.mean_confidence:.4f}",
        "min_confidence": f"{row.min_confidence:.4f}",
        "num_lines": str(row.num_lines),
    }


def write_ledger(rows: list[LedgerRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_FIELDNAMES))
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_to_dict(row))
