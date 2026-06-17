from __future__ import annotations

import os
from pathlib import Path

import pytest

from shoebox.extract import RegexExtractor
from shoebox.ocr import OcrEngine

_FIXTURE = Path(__file__).parent / "fixtures" / "receipts" / "cafe-margarita.png"


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("SHOEBOX_RUN_OCR") != "1",
    reason="set SHOEBOX_RUN_OCR=1 to run real OCR (downloads models)",
)
def test_end_to_end_ocr_extracts_fields() -> None:
    assert _FIXTURE.exists(), "run scripts/make_synthetic_receipts.py first"
    ocr = OcrEngine(lang="en").recognize_path(_FIXTURE)
    fields = RegexExtractor().extract(ocr)
    assert ocr.mean_confidence > 0.5
    assert fields.vendor is not None
    assert fields.total is not None
