"""Build a labeled (text, category, vendor-group) dataset from SROIE."""

from __future__ import annotations

import re
from dataclasses import dataclass

from datasets import concatenate_datasets, load_dataset

from shoebox.benchmark.data import SROIE_DATASET
from shoebox.classifier.labels import weak_label

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class LabeledReceipt:
    text: str
    category: str
    vendor_group: str


def _vendor_group(company: str | None, fallback: str) -> str:
    base = (company or "").lower().strip()
    if base:
        return _WS.sub(" ", re.sub(r"[^a-z0-9 ]", " ", base)).strip()
    return fallback[:24]


def build_dataset(limit: int | None = None) -> list[LabeledReceipt]:
    dataset = load_dataset(SROIE_DATASET)
    combined = concatenate_datasets([dataset["train"], dataset["test"]])
    count = len(combined) if limit is None else min(limit, len(combined))
    receipts: list[LabeledReceipt] = []
    for index in range(count):
        row = combined[index]
        words = row.get("words") or []
        text = " ".join(str(word) for word in words).strip()
        if not text:
            continue
        company = (row.get("entities") or {}).get("company")
        receipts.append(
            LabeledReceipt(
                text=text,
                category=weak_label(f"{company or ''} {text}"),
                vendor_group=_vendor_group(company, text),
            )
        )
    return receipts
