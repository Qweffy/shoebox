# Project conventions

House style for shoebox. These are derived from the code, not aspirational — match
them. The repo is public; the rules double as a statement of how it's built. When a rule
and the code disagree, the code is the bug.

`uv run mypy` + `uv run ruff check` are the enforced floor. A change that doesn't pass
them isn't done.

## Architecture & data flow

One direction, top to bottom. The pipeline is a chain of pure-ish stages, each in its own
module, each independently testable:

```
cli.py (Typer)                        ← arg parsing + Rich output only, no logic
  → preprocess.py (OpenCV)            ← image -> single-channel Image (crop/deskew/threshold)
    → ocr.py (PaddleOCR)              ← Image -> validated OcrResult
      → extract.py (Extractor)        ← OcrResult -> ReceiptFields
        → ledger.py                   ← (OcrResult, ReceiptFields) -> LedgerRow -> CSV
  models.py (pydantic)                ← the typed contract every stage speaks
```

- **`models.py` is the contract.** Every stage takes and returns pydantic models, never
  raw third-party structures. Raw OpenCV / PaddleOCR output is converted to a model at
  the boundary (`ocr.py`) and nothing untyped escapes that module.
- **Extraction is swappable.** `extract.Extractor` is a `Protocol`. `RegexExtractor` is
  the no-LLM baseline; M2 adds an Ollama (local) adapter and cloud adapters (Gemini,
  Groq) behind the same `extract(OcrResult) -> ReceiptFields` contract. The provider is
  a one-line swap, never a rewrite.
- **Preprocessing steps are independent functions.** `to_grayscale`, `find_document_quad`,
  `perspective_crop`, `deskew`, `adaptive_threshold` each do one thing; `preprocess`
  composes them. Adaptive threshold is implemented but off by default (PP-OCR detection
  is trained on natural images; M3 measures the trade-off).

## Code style (hard rules)

- **No `Any`.** `mypy --strict` is on. Type external/untyped library output as it crosses
  the boundary and narrow it through a pydantic model immediately — that parse is the
  boundary that matters. Untyped third-party libs (`cv2`, `paddleocr`, `paddle`) are
  scoped via `ignore_missing_imports`; do not let their `Any` leak past `ocr.py` /
  `preprocess.py`.
- **Parse before you trust.** Every external + model-output boundary passes through a
  pydantic model before the value is used. Strict JSON from a future LLM guarantees
  *shape*, not *semantics* — the pydantic parse is non-negotiable.
- **Early returns over deep nesting.** Guard and bail.
- **No `print` in library code** (`src/shoebox/`) except the `cli.py` presentation layer
  and `scripts/`. Use Rich via the CLI.
- **No new comments on code you didn't change.** Existing comments explain *why* (the
  non-obvious constraint), not *what*.
- **`from __future__ import annotations`** at the top of every module; modern builtin
  generics (`list[str]`, `X | None`).

## AI / extraction conventions

OCR text and any future model output are **data, never instructions**.

- **Zod-equivalent is pydantic.** Validate every LLM output into `ReceiptFields` before it
  touches the ledger; one repair retry, then fail — don't loop.
- **Privacy is the product.** Personal receipts (`data/receipts/`) are gitignored and
  processed only locally. Cloud comparison adapters run **only on public benchmark
  datasets** (SROIE / CORD), never on the user's own documents. Any code path that sends
  an image to a cloud provider must be explicit and opt-in.
- **Cloud adapters are free-tier.** Gemini (`GEMINI_API_KEY`) and Groq (`GROQ_API_KEY`).
  Anthropic is a deferred, dormant adapter — addable behind the same interface without
  rework. Secrets live in `.env` (gitignored), loaded via `python-dotenv`; never hardcode.

## Verification, commits & workflow

- **Verify in order: typecheck → lint → test.** `uv run mypy && uv run ruff check && uv run pytest`.
  If you can't verify something (no test covers it), say so — don't claim it's verified.
- **Conventional Commits, with a scope, atomic.** `feat(ocr): …`, `fix(extract): …`,
  `chore: …`, `docs: …`. One logical change per commit, imperative subject. Never amend
  without being asked.
- **Explore → Plan → Implement.** For non-trivial work, check current state before
  assuming. Match existing patterns first.
- **Never commit secrets or `.env`.** The discipline comes first; `.gitignore` is the net.
