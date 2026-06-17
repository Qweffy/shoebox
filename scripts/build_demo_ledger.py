"""Seed data/ledger.csv from SROIE using the local pipeline (PaddleOCR + regex baseline).

GPU-free (PaddleOCR runs on CPU; regex extractor needs no LLM), so it can populate the
dashboard with real receipt data without the local model. Category via the weak labeler.

uv run python scripts/build_demo_ledger.py [--limit 40]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from shoebox.benchmark.data import load_sroie
from shoebox.classifier.labels import weak_label
from shoebox.extract import RegexExtractor
from shoebox.ledger import to_ledger_row
from shoebox.ocr import OcrEngine
from shoebox.preprocess import preprocess

_FIELDS = (
    "source",
    "vendor",
    "date",
    "total",
    "currency",
    "mean_confidence",
    "min_confidence",
    "num_lines",
    "category",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--out", type=Path, default=Path("data/ledger.csv"))
    args = parser.parse_args()

    samples = load_sroie(split="test", limit=args.limit)
    engine = OcrEngine(lang="en")
    extractor = RegexExtractor()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_FIELDS))
        writer.writeheader()
        for index, sample in enumerate(samples):
            array = np.frombuffer(sample.image_bytes, dtype=np.uint8)
            decoded = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if decoded is None:
                continue
            ocr = engine.recognize(preprocess(decoded.astype(np.uint8)), source=sample.key)
            fields = extractor.extract(ocr)
            row = to_ledger_row(ocr, fields)
            writer.writerow(
                {
                    "source": row.source,
                    "vendor": row.vendor or "",
                    "date": row.date.isoformat() if row.date is not None else "",
                    "total": f"{row.total:.2f}" if row.total is not None else "",
                    "currency": row.currency or "",
                    "mean_confidence": f"{row.mean_confidence:.4f}",
                    "min_confidence": f"{row.min_confidence:.4f}",
                    "num_lines": str(row.num_lines),
                    "category": weak_label(row.vendor or ocr.full_text),
                }
            )
            print(f"  {index + 1}/{len(samples)} {row.vendor or '—'}")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
