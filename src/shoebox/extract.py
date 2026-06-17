"""Field extraction from OCR text.

`Extractor` is the swappable contract: M2 adds an Ollama (local LLM) adapter and a
cloud adapter behind it. `RegexExtractor` is the honest no-LLM baseline the benchmark
measures against.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Protocol

from shoebox.models import OcrResult, ReceiptFields

_AMOUNT_RE = re.compile(r"(?<!\d)(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})(?!\d)")
_TOTAL_LINE_RE = re.compile(
    r"\b(total|amount due|balance|importe|total a pagar)\b", re.IGNORECASE
)
_DATE_RES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"), ("%Y-%m-%d",)),
    (re.compile(r"\b(\d{2}/\d{2}/\d{4})\b"), ("%d/%m/%Y", "%m/%d/%Y")),
    (re.compile(r"\b(\d{2}-\d{2}-\d{4})\b"), ("%d-%m-%Y", "%m-%d-%Y")),
    (re.compile(r"\b(\d{1,2}/\d{1,2}/\d{2})\b"), ("%d/%m/%y", "%m/%d/%y")),
)
_CURRENCY_SYMBOLS: tuple[tuple[str, str], ...] = (("$", "USD"), ("€", "EUR"), ("£", "GBP"))
_CURRENCY_CODES: tuple[str, ...] = ("USD", "ARS", "EUR", "GBP", "BRL")


class Extractor(Protocol):
    def extract(self, ocr: OcrResult) -> ReceiptFields: ...


def _parse_amount(raw: str) -> Decimal | None:
    s = raw.strip()
    if "." in s and "," in s:
        if s.rfind(".") > s.rfind(","):
            s = s.replace(",", "")
        else:
            s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".") if re.search(r",\d{2}$", s) else s.replace(",", "")
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def _find_date(text: str) -> tuple[date | None, str | None]:
    for pattern, formats in _DATE_RES:
        match = pattern.search(text)
        if match is None:
            continue
        raw = match.group(1)
        for fmt in formats:
            try:
                return datetime.strptime(raw, fmt).date(), raw
            except ValueError:
                continue
        return None, raw
    return None, None


class RegexExtractor:
    """Rule-based baseline extractor."""

    def extract(self, ocr: OcrResult) -> ReceiptFields:
        lines = [line.text for line in ocr.lines]
        parsed_date, date_text = _find_date(ocr.full_text)
        return ReceiptFields(
            vendor=self._vendor(lines),
            date=parsed_date,
            date_text=date_text,
            total=self._total(lines),
            currency=self._currency(ocr.full_text),
        )

    def _total(self, lines: list[str]) -> Decimal | None:
        candidates: list[Decimal] = []
        for line in lines:
            parsed = [
                amount
                for amount in (_parse_amount(m) for m in _AMOUNT_RE.findall(line))
                if amount is not None
            ]
            if not parsed:
                continue
            if _TOTAL_LINE_RE.search(line):
                return max(parsed)
            candidates.extend(parsed)
        return max(candidates) if candidates else None

    def _vendor(self, lines: list[str]) -> str | None:
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if _TOTAL_LINE_RE.search(stripped) or _find_date(stripped)[1] is not None:
                continue
            if len(re.sub(r"[^A-Za-z]", "", stripped)) >= 3:
                return stripped
        return None

    def _currency(self, text: str) -> str | None:
        for symbol, code in _CURRENCY_SYMBOLS:
            if symbol in text:
                return code
        for code in _CURRENCY_CODES:
            if re.search(rf"\b{code}\b", text):
                return code
        return None
