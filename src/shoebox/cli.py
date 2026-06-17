"""shoebox CLI — turn receipt photos into a CSV ledger, 100% on-device."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from shoebox.extract import RegexExtractor
from shoebox.ledger import to_ledger_row, write_ledger
from shoebox.models import LedgerRow
from shoebox.ocr import OcrEngine
from shoebox.preprocess import load_image, preprocess

app = typer.Typer(
    add_completion=False,
    help="Privacy-first receipt digitizer — 100% on-device, fully offline.",
)
console = Console()

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@app.callback()
def _root() -> None:
    """Privacy-first receipt digitizer — 100% on-device, fully offline."""


def _gather_images(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
    if path.suffix.lower() in _IMAGE_SUFFIXES:
        return [path]
    raise typer.BadParameter(f"Not an image or a directory of images: {path}")


@app.command()
def process(
    input_path: Annotated[
        Path, typer.Argument(exists=True, help="Receipt image or directory of images.")
    ],
    out: Annotated[Path, typer.Option(help="Output CSV path.")] = Path("ledger.csv"),
    lang: Annotated[str, typer.Option(help="PaddleOCR language code.")] = "en",
    crop: Annotated[bool, typer.Option(help="Perspective-crop the document.")] = True,
    threshold: Annotated[
        bool, typer.Option(help="Adaptive-threshold before OCR (off by default).")
    ] = False,
) -> None:
    """Process receipts into a CSV ledger with confidence scores."""
    images = _gather_images(input_path)
    if not images:
        console.print("[yellow]No images found.[/yellow]")
        raise typer.Exit(code=1)

    engine = OcrEngine(lang=lang)
    extractor = RegexExtractor()
    rows: list[LedgerRow] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing receipts", total=len(images))
        for image_path in images:
            progress.update(task, description=f"OCR {image_path.name}")
            processed = preprocess(load_image(image_path), crop=crop, threshold=threshold)
            ocr_result = engine.recognize(processed, source=str(image_path))
            fields = extractor.extract(ocr_result)
            rows.append(to_ledger_row(ocr_result, fields))
            progress.advance(task)

    write_ledger(rows, out)
    _print_summary(rows, out)


def _print_summary(rows: list[LedgerRow], out: Path) -> None:
    table = Table(title=f"Ledger — {out}")
    table.add_column("Source")
    table.add_column("Vendor")
    table.add_column("Date")
    table.add_column("Total", justify="right")
    table.add_column("Conf", justify="right")
    for row in rows:
        total = f"{row.currency or ''}{row.total:.2f}" if row.total is not None else "—"
        table.add_row(
            row.source,
            row.vendor or "—",
            row.date.isoformat() if row.date is not None else "—",
            total,
            f"{row.mean_confidence:.2f}",
        )
    console.print(table)
    console.print(f"[green]Wrote {len(rows)} rows to {out}[/green]")


def main() -> None:
    app()
