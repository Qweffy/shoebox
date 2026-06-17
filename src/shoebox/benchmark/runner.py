"""Run benchmark systems over labeled samples and aggregate scores."""

from __future__ import annotations

import time

import cv2
import numpy as np

from shoebox.benchmark.data import Sample
from shoebox.benchmark.scoring import SystemResult, aggregate, score_one
from shoebox.extract import Extractor, VisionExtractor
from shoebox.models import OcrResult, ReceiptFields
from shoebox.ocr import OcrEngine
from shoebox.preprocess import Image, preprocess


def _decode(data: bytes) -> Image:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode image bytes")
    return image.astype(np.uint8)


def ocr_samples(engine: OcrEngine, samples: list[Sample]) -> dict[str, OcrResult]:
    results: dict[str, OcrResult] = {}
    for sample in samples:
        processed = preprocess(_decode(sample.image_bytes))
        results[sample.key] = engine.recognize(processed, source=sample.key)
    return results


def evaluate_text(
    system: str,
    extractor: Extractor,
    samples: list[Sample],
    ocr_by_key: dict[str, OcrResult],
) -> SystemResult:
    scored = []
    latencies = []
    errors = 0
    for sample in samples:
        start = time.perf_counter()
        try:
            pred = extractor.extract(ocr_by_key[sample.key])
        except Exception:
            errors += 1
            pred = ReceiptFields()
        latencies.append(time.perf_counter() - start)
        scored.append(score_one(pred, sample.gt_vendor, sample.gt_date, sample.gt_total))
    return aggregate(system, scored, latencies, errors)


def evaluate_vision(system: str, extractor: VisionExtractor, samples: list[Sample]) -> SystemResult:
    scored = []
    latencies = []
    errors = 0
    for sample in samples:
        start = time.perf_counter()
        try:
            pred = extractor.extract_image(sample.image_bytes, sample.media_type)
        except Exception:
            errors += 1
            pred = ReceiptFields()
        latencies.append(time.perf_counter() - start)
        scored.append(score_one(pred, sample.gt_vendor, sample.gt_date, sample.gt_total))
    return aggregate(system, scored, latencies, errors)
