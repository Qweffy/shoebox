# shoebox

Turn a shoebox of paper receipts into a clean expense ledger — **100% on-device, fully
offline, no receipt ever leaves your laptop.** The deliberate counter-narrative to a
cloud bill-reader: same document domain, opposite deployment philosophy, benchmarked
head-to-head against frontier cloud vision.

> **Measured (M3, SROIE test):** the local stack (PaddleOCR + Qwen-3B q8) reaches **~90% of
> frontier cloud-vision accuracy** (Groq Llama-4-Scout) on field-level extraction — at **$0**
> and **fully offline**. 74.0% vs 82.7% overall; see [`benchmark/RESULTS.md`](benchmark/RESULTS.md).

Built on Python 3.12, OpenCV + PaddleOCR (PP-OCRv5) for OCR, a local LLM (Ollama + Qwen
3B) for structured extraction, scikit-learn for category classification, and a QLoRA
fine-tune via MLX. Cloud comparison adapters (Gemini, Groq) are all free-tier.

## M1 — OCR ledger CLI (this milestone)

```bash
uv sync                                            # install deps (Python 3.12, pinned)
uv run python scripts/make_synthetic_receipts.py   # generate demo receipts
uv run shoebox process tests/fixtures/receipts --out ledger.csv
```

Pipeline: receipt photo → OpenCV preprocessing (perspective crop, deskew, optional
adaptive threshold) → PaddleOCR → CSV ledger with per-receipt confidence scores. A
regex baseline pulls vendor / date / total (the honest no-LLM benchmark M2's local LLM
must beat).

```bash
uv run shoebox process path/to/receipt.jpg          # a single image
uv run shoebox process data/receipts --out out.csv  # a folder (drop your own photos here)
uv run shoebox process data/receipts --threshold    # binarize before OCR (off by default)
```

## Commands

| Command | What it does |
| ------- | ------------ |
| `uv run shoebox process <img\|dir>` | OCR receipts → CSV ledger with confidence |
| `uv run python scripts/make_synthetic_receipts.py` | Generate deterministic demo receipts |
| `uv run mypy` | Strict type check |
| `uv run ruff check` / `uv run ruff format` | Lint / format |
| `uv run pytest` | Unit suite (fast; OCR integration test gated by `SHOEBOX_RUN_OCR=1`) |

## Privacy

Your real receipts (`data/receipts/`) are gitignored and only ever processed locally.
The optional cloud comparison adapters run **only on public benchmark datasets**, never
on your personal documents.

## Roadmap

- **M1** — OCR ledger CLI ✅
- **M2** — Local structured extraction (Ollama + Qwen 3B) behind an `Extractor` interface; Gemini / Groq cloud adapters ✅
- **M3** — Labeled benchmark (SROIE): field accuracy / latency / cost-per-1k; quantization study ✅
- **M4** — scikit-learn expense-category classifier (macro-F1 0.49 vs 0.09 majority) ✅
- **M5** — QLoRA fine-tune via MLX; publish weights + model card on HF Hub
- **M6** — Local web dashboard (Overview / Ledger / Benchmark) from the Claude Design handoff ✅

Run the dashboard: `uv run uvicorn shoebox.web.app:app` → http://localhost:8000

See [`AGENTS.md`](AGENTS.md) for house conventions.
