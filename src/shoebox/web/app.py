"""FastAPI dashboard for shoebox — implements the Claude Design handoff.

Run: uv run uvicorn shoebox.web.app:app --reload
Reads data/ledger.csv and benchmark/results.csv from the current working directory.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from shoebox.web.benchmark_data import load_benchmark
from shoebox.web.ledger_data import CATEGORY_COLORS, aggregate, load_ledger

_WEB_DIR = Path(__file__).parent
# Project root (src/shoebox/web/app.py -> repo root), so data/ resolves regardless of CWD.
_ROOT = _WEB_DIR.resolve().parents[2]
_LEDGER_PATH = _ROOT / "data" / "ledger.csv"
_BENCHMARK_PATH = _ROOT / "benchmark" / "results.csv"

app = FastAPI(title="shoebox")
app.mount("/static", StaticFiles(directory=_WEB_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(_WEB_DIR / "templates"))


def _money(value: Decimal | None) -> str:
    return "—" if value is None else f"${value:,.2f}"


def _money_compact(value: Decimal | None) -> str:
    return "—" if value is None else f"${value:,.0f}"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


templates.env.filters["money"] = _money
templates.env.filters["money_compact"] = _money_compact
templates.env.filters["pct"] = _pct


@app.get("/", response_class=HTMLResponse)
def overview(request: Request) -> Response:
    overview_data = aggregate(load_ledger(_LEDGER_PATH))
    return templates.TemplateResponse(
        request,
        "overview.html",
        {
            "active": "overview",
            "ov": overview_data,
            "colors": CATEGORY_COLORS,
            "month": "All time",
            "greeting": "Welcome back, Nico",
        },
    )


@app.get("/ledger", response_class=HTMLResponse)
def ledger(request: Request) -> Response:
    entries = sorted(
        load_ledger(_LEDGER_PATH),
        key=lambda e: (e.date is not None, e.date),
        reverse=True,
    )
    return templates.TemplateResponse(
        request,
        "ledger.html",
        {"active": "ledger", "entries": entries, "colors": CATEGORY_COLORS},
    )


@app.get("/benchmark", response_class=HTMLResponse)
def benchmark(request: Request) -> Response:
    return templates.TemplateResponse(
        request,
        "benchmark.html",
        {"active": "benchmark", "bench": load_benchmark(_BENCHMARK_PATH)},
    )


@app.get("/import", response_class=HTMLResponse)
def import_screen(request: Request) -> Response:
    return templates.TemplateResponse(
        request,
        "stub.html",
        {
            "active": "import",
            "title": "Import & Scan",
            "note": "Use the CLI today: uv run shoebox process <image|dir>.",
        },
    )


@app.get("/settings", response_class=HTMLResponse)
def settings_screen(request: Request) -> Response:
    return templates.TemplateResponse(
        request,
        "stub.html",
        {
            "active": "settings",
            "title": "Settings",
            "note": "Model + privacy controls map to .env (local Qwen, Gemini, Groq).",
        },
    )


@app.get("/ledger.csv")
def ledger_csv() -> Response:
    if _LEDGER_PATH.exists():
        return FileResponse(_LEDGER_PATH, media_type="text/csv", filename="shoebox-ledger.csv")
    return PlainTextResponse("No ledger yet.", status_code=404)
