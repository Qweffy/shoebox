"""PaddleOCR wrapper.

Converts PaddleOCR's untyped ``predict`` output into a validated ``OcrResult`` at the
boundary, so no untyped OCR structure escapes this module. The engine is lazy so
importing the module never triggers a model download.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from shoebox.models import OcrLine, OcrResult
from shoebox.preprocess import Image, load_image

if TYPE_CHECKING:
    from paddleocr import PaddleOCR


class OcrEngine:
    """Lazily-initialized PaddleOCR engine returning typed results."""

    def __init__(self, lang: str = "en") -> None:
        self._lang = lang
        self._ocr: PaddleOCR | None = None

    def _engine(self) -> PaddleOCR:
        if self._ocr is None:
            from paddleocr import PaddleOCR

            # We do our own crop/deskew, so disable PaddleOCR's doc-preprocessing modules.
            self._ocr = PaddleOCR(
                lang=self._lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        return self._ocr

    def recognize(self, image: Image, source: str) -> OcrResult:
        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        lines: list[OcrLine] = []
        for result in self._engine().predict(bgr):
            texts = result["rec_texts"]
            scores = result["rec_scores"]
            polys = result["rec_polys"]
            for text, score, poly in zip(texts, scores, polys, strict=True):
                cleaned = str(text).strip()
                if not cleaned:
                    continue
                points = [(float(x), float(y)) for x, y in np.asarray(poly).tolist()]
                lines.append(OcrLine(text=cleaned, confidence=float(score), bbox=points))
        return OcrResult(source=source, lines=lines)

    def recognize_path(self, path: Path) -> OcrResult:
        return self.recognize(load_image(path), source=str(path))
