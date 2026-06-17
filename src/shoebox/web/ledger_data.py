"""Read the local ledger CSV and aggregate it for the dashboard."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from shoebox.classifier.labels import weak_label

REVIEW_THRESHOLD = 0.85

CATEGORY_COLORS: dict[str, str] = {
    "dining": "#8A5A44",
    "groceries": "#5C7457",
    "fuel": "#B7791F",
    "pharmacy": "#B23A2E",
    "retail": "#6B5B95",
    "services": "#3A6EA5",
    "other": "#8C877E",
}


@dataclass(frozen=True)
class LedgerEntry:
    source: str
    vendor: str
    date: date | None
    total: Decimal | None
    currency: str
    confidence: float
    category: str

    @property
    def needs_review(self) -> bool:
        return self.confidence < REVIEW_THRESHOLD


@dataclass(frozen=True)
class CategorySlice:
    name: str
    amount: Decimal
    pct: int
    hex: str
    bar_width: int


@dataclass(frozen=True)
class DonutSegment:
    hex: str
    dasharray: str
    dashoffset: str


_DONUT_CIRCUMFERENCE = 414.69  # 2 * pi * r, r = 66 (matches the handoff SVG)


@dataclass(frozen=True)
class Overview:
    total_spend: Decimal
    receipt_count: int
    mean_confidence: float
    need_review: int
    categories: list[CategorySlice]
    donut: list[DonutSegment]
    recent: list[LedgerEntry]
    has_data: bool


def _to_decimal(value: str) -> Decimal | None:
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError):
        return None


def _to_date(value: str) -> date | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def load_ledger(path: Path) -> list[LedgerEntry]:
    if not path.exists():
        return []
    entries: list[LedgerEntry] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            vendor = row.get("vendor", "").strip()
            category = row.get("category") or weak_label(vendor)
            entries.append(
                LedgerEntry(
                    source=row.get("source", ""),
                    vendor=vendor or "—",
                    date=_to_date(row.get("date", "")),
                    total=_to_decimal(row.get("total", "")),
                    currency=row.get("currency", "") or "",
                    confidence=float(row.get("mean_confidence") or 0.0),
                    category=category,
                )
            )
    return entries


def aggregate(entries: list[LedgerEntry]) -> Overview:
    if not entries:
        return Overview(Decimal(0), 0, 0.0, 0, [], [], [], has_data=False)

    totals = [e.total for e in entries if e.total is not None]
    total_spend = sum(totals, Decimal(0))
    mean_conf = sum(e.confidence for e in entries) / len(entries)
    need_review = sum(1 for e in entries if e.needs_review)

    by_category: dict[str, Decimal] = {}
    for entry in entries:
        if entry.total is not None:
            by_category[entry.category] = by_category.get(entry.category, Decimal(0)) + entry.total

    ranked = sorted(by_category.items(), key=lambda kv: kv[1], reverse=True)
    top_amount = ranked[0][1] if ranked else Decimal(1)
    categories = [
        CategorySlice(
            name=name,
            amount=amount,
            pct=round(amount / total_spend * 100) if total_spend else 0,
            hex=CATEGORY_COLORS.get(name, "#8C877E"),
            bar_width=round(amount / top_amount * 100) if top_amount else 0,
        )
        for name, amount in ranked
    ]

    donut: list[DonutSegment] = []
    cumulative = 0.0
    for category in categories:
        fraction = float(category.amount / total_spend) if total_spend else 0.0
        segment = fraction * _DONUT_CIRCUMFERENCE
        donut.append(
            DonutSegment(
                hex=category.hex,
                dasharray=f"{segment:.1f} {_DONUT_CIRCUMFERENCE - segment:.1f}",
                dashoffset=f"{-cumulative:.1f}",
            )
        )
        cumulative += segment

    recent = sorted(entries, key=lambda e: e.date or date.min, reverse=True)[:6]
    return Overview(
        total_spend, len(entries), mean_conf, need_review, categories, donut, recent, has_data=True
    )
