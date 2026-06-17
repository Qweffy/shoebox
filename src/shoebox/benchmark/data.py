"""Load labeled receipt samples (SROIE) for the benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from datasets import load_dataset

SROIE_DATASET = "jsdnrs/ICDAR2019-SROIE"


@dataclass(frozen=True)
class Sample:
    key: str
    image_bytes: bytes
    media_type: str
    gt_vendor: str | None
    gt_date: str | None
    gt_total: str | None


def load_sroie(split: str = "test", limit: int | None = None) -> list[Sample]:
    dataset = load_dataset(SROIE_DATASET)[split]
    count = len(dataset) if limit is None else min(limit, len(dataset))
    samples: list[Sample] = []
    for index in range(count):
        row = dataset[index]
        buffer = BytesIO()
        row["image"].convert("RGB").save(buffer, format="JPEG")
        entities = row["entities"]
        samples.append(
            Sample(
                key=str(row["key"]),
                image_bytes=buffer.getvalue(),
                media_type="image/jpeg",
                gt_vendor=entities.get("company"),
                gt_date=entities.get("date"),
                gt_total=entities.get("total"),
            )
        )
    return samples
