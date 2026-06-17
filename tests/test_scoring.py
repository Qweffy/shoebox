from __future__ import annotations

from datetime import date
from decimal import Decimal

from shoebox.benchmark.scoring import (
    aggregate,
    normalize_vendor,
    parse_date,
    parse_total,
    score_one,
)
from shoebox.models import ReceiptFields


def test_normalize_vendor_strips_punctuation_and_case() -> None:
    assert normalize_vendor("OJC Marketing SDN. BHD.") == "ojc marketing sdn bhd"
    assert normalize_vendor(None) == ""


def test_parse_date_handles_multiple_formats() -> None:
    assert parse_date("15/01/2019") == date(2019, 1, 15)
    assert parse_date("2026-06-12") == date(2026, 6, 12)
    assert parse_date("nonsense") is None


def test_parse_total_normalizes_amounts() -> None:
    assert parse_total("193.00") == Decimal("193.00")
    assert parse_total("RM 1,234.50") == Decimal("1234.50")
    assert parse_total("7,00") == Decimal("7.00")
    assert parse_total(None) is None


def test_score_one_all_correct() -> None:
    pred = ReceiptFields(
        vendor="OJC MARKETING SDN BHD", date=date(2019, 1, 15), total=Decimal("193.00")
    )
    scores = score_one(pred, "OJC Marketing Sdn Bhd", "15/01/2019", "193.00")
    assert scores.vendor and scores.date and scores.total
    assert scores.correct == 3


def test_score_one_partial() -> None:
    pred = ReceiptFields(vendor=None, date=date(2019, 1, 15), total=Decimal("1.00"))
    scores = score_one(pred, "SHOP", "15/01/2019", "193.00")
    assert not scores.vendor
    assert scores.date
    assert not scores.total
    assert scores.correct == 1


def test_aggregate_computes_means() -> None:
    scored = [score_one(ReceiptFields(date=date(2019, 1, 15)), None, "15/01/2019", None)]
    result = aggregate("x", scored, [0.5], errors=0)
    assert result.n == 1
    assert result.date_acc == 1.0
    assert result.vendor_acc == 0.0
    assert result.mean_latency_s == 0.5
