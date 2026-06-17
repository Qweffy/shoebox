"""Generate deterministic synthetic receipt images for tests and demos.

Run: uv run python scripts/make_synthetic_receipts.py [out_dir]
Defaults to tests/fixtures/receipts. The text content is fixed so tests can assert
exact extracted fields without depending on real photos.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/Library/Fonts/Courier New.ttf",
)


class ReceiptSpec(TypedDict):
    slug: str
    vendor: str
    date: str
    items: list[tuple[str, str]]
    total: str
    currency_symbol: str


RECEIPTS: list[ReceiptSpec] = [
    {
        "slug": "cafe-margarita",
        "vendor": "CAFE MARGARITA",
        "date": "2026-06-12",
        "items": [("Espresso", "3.50"), ("Medialuna", "2.00"), ("Agua Mineral", "1.50")],
        "total": "7.00",
        "currency_symbol": "$",
    },
    {
        "slug": "farmacia-central",
        "vendor": "FARMACIA CENTRAL",
        "date": "2026-05-28",
        "items": [("Ibuprofeno 400mg", "8.20"), ("Vitamina C", "5.30"), ("Alcohol en gel", "3.10")],
        "total": "16.60",
        "currency_symbol": "$",
    },
    {
        "slug": "super-norte",
        "vendor": "SUPERMERCADO NORTE",
        "date": "2026-06-01",
        "items": [
            ("Leche entera", "2.40"),
            ("Pan integral", "3.00"),
            ("Cafe molido", "9.90"),
            ("Huevos x12", "4.75"),
        ],
        "total": "20.05",
        "currency_symbol": "$",
    },
]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)


def render_receipt(spec: ReceiptSpec, width: int = 540) -> Image.Image:
    pad = 28
    line_h = 30
    font = _load_font(22)
    font_bold = _load_font(28)

    rows: list[tuple[str, ImageFont.FreeTypeFont | ImageFont.ImageFont, str]] = []
    rows.append((spec["vendor"], font_bold, "center"))
    rows.append((f"Date: {spec['date']}", font, "left"))
    rows.append(("-" * 30, font, "left"))
    for name, price in spec["items"]:
        rows.append((f"{name:<22}{spec['currency_symbol']}{price:>6}", font, "left"))
    rows.append(("-" * 30, font, "left"))
    rows.append((f"{'TOTAL':<22}{spec['currency_symbol']}{spec['total']:>6}", font_bold, "left"))
    rows.append(("Thank you!", font, "center"))

    height = pad * 2 + line_h * len(rows)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = pad
    for text, used_font, align in rows:
        if align == "center":
            text_width = draw.textlength(text, font=used_font)
            x = (width - text_width) / 2
        else:
            x = pad
        draw.text((x, y), text, fill="black", font=used_font)
        y += line_h
    return image


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in RECEIPTS:
        image = render_receipt(spec)
        path = out_dir / f"{spec['slug']}.png"
        image.save(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures/receipts")
    main(target)
