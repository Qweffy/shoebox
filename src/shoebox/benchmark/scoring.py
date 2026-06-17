"""Scoring for the extraction benchmark: normalized field-level exact match."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from shoebox.models import ReceiptFields

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]")
_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d.%m.%Y")
_CENTS = Decimal("0.01")


def normalize_vendor(value: str | None) -> str:
    if not value:
        return ""
    return _WS.sub(" ", _NON_ALNUM.sub(" ", value.lower())).strip()


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    text = value.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_total(value: str | None) -> Decimal | None:
    if not value:
        return None
    cleaned = re.sub(r"[^0-9.,]", "", value)
    if cleaned.count(",") == 1 and cleaned.count(".") == 0:
        cleaned = cleaned.replace(",", ".")
    else:
        cleaned = cleaned.replace(",", "")
    try:
        return Decimal(cleaned).quantize(_CENTS)
    except (InvalidOperation, ValueError):
        return None


@dataclass(frozen=True)
class FieldScores:
    vendor: bool
    date: bool
    total: bool

    @property
    def correct(self) -> int:
        return int(self.vendor) + int(self.date) + int(self.total)


def score_one(
    pred: ReceiptFields, gt_vendor: str | None, gt_date: str | None, gt_total: str | None
) -> FieldScores:
    pred_vendor = normalize_vendor(pred.vendor)
    gt_vendor_norm = normalize_vendor(gt_vendor)
    vendor_ok = bool(gt_vendor_norm) and (
        pred_vendor == gt_vendor_norm
        or (
            len(pred_vendor) >= 4
            and (pred_vendor in gt_vendor_norm or gt_vendor_norm in pred_vendor)
        )
    )

    gt_date_parsed = parse_date(gt_date)
    date_ok = gt_date_parsed is not None and pred.date == gt_date_parsed

    gt_total_parsed = parse_total(gt_total)
    pred_total = pred.total.quantize(_CENTS) if pred.total is not None else None
    total_ok = gt_total_parsed is not None and pred_total == gt_total_parsed

    return FieldScores(vendor=vendor_ok, date=date_ok, total=total_ok)


@dataclass(frozen=True)
class SystemResult:
    system: str
    n: int
    vendor_acc: float
    date_acc: float
    total_acc: float
    overall_acc: float
    mean_latency_s: float
    errors: int


def aggregate(
    system: str, scored: list[FieldScores], latencies: list[float], errors: int
) -> SystemResult:
    n = len(scored)
    if n == 0:
        return SystemResult(system, 0, 0.0, 0.0, 0.0, 0.0, 0.0, errors)
    vendor = sum(s.vendor for s in scored) / n
    date_acc = sum(s.date for s in scored) / n
    total = sum(s.total for s in scored) / n
    overall = sum(s.correct for s in scored) / (3 * n)
    latency = sum(latencies) / len(latencies) if latencies else 0.0
    return SystemResult(system, n, vendor, date_acc, total, overall, latency, errors)
