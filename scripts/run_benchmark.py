"""Run the extraction benchmark matrix and write results.

uv run python scripts/run_benchmark.py [--limit N] [--out-dir benchmark]

Systems: regex baseline; local Qwen 3B quant sweep (q4/q8/fp16); Gemini & Groq on OCR
text; Gemini & Groq vision. Scored on SROIE test (vendor/date/total). Cloud runs use
free tiers; the cost column is a documented list-price-equivalent estimate.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from pathlib import Path

from shoebox.benchmark.data import load_sroie
from shoebox.benchmark.runner import evaluate_text, evaluate_vision, ocr_samples
from shoebox.benchmark.scoring import SystemResult
from shoebox.config import load_settings
from shoebox.extract import LocalExtractor, get_extractor, get_vision_extractor
from shoebox.ocr import OcrEngine

# Approx mid-2026 list prices, USD per 1M tokens (input, output) — for the estimate only.
_PRICES = {"gemini": (0.30, 2.50), "groq-text": (0.15, 0.60), "groq-vision": (0.11, 0.34)}
_TOK_TEXT = (350, 60)
_TOK_VISION = (1100, 60)


def _cost_per_1k(system: str) -> float:
    if system.startswith(("regex", "local")):
        return 0.0
    tokens = _TOK_VISION if system.endswith("vision") else _TOK_TEXT
    if system.startswith("gemini"):
        price = _PRICES["gemini"]
    elif system == "groq-vision":
        price = _PRICES["groq-vision"]
    else:
        price = _PRICES["groq-text"]
    return (tokens[0] * price[0] + tokens[1] * price[1]) / 1_000_000 * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark"))
    args = parser.parse_args()

    settings = load_settings()
    print(f"Loading SROIE test (limit={args.limit})...")
    samples = load_sroie(split="test", limit=args.limit)
    print(f"  {len(samples)} samples")

    print("Running OCR on all samples (once, reused by text systems)...")
    engine = OcrEngine(lang="en")
    ocr_by_key = ocr_samples(engine, samples)

    text_systems: dict[str, Callable[[], SystemResult]] = {
        "regex": lambda: evaluate_text(
            "regex", get_extractor("regex", settings), samples, ocr_by_key
        ),
        "local-q4": lambda: evaluate_text(
            "local-q4",
            LocalExtractor("qwen2.5:3b-instruct-q4_K_M", settings.ollama_host),
            samples,
            ocr_by_key,
        ),
        "local-q8": lambda: evaluate_text(
            "local-q8",
            LocalExtractor("qwen2.5:3b-instruct-q8_0", settings.ollama_host),
            samples,
            ocr_by_key,
        ),
        "local-fp16": lambda: evaluate_text(
            "local-fp16",
            LocalExtractor("qwen2.5:3b-instruct-fp16", settings.ollama_host),
            samples,
            ocr_by_key,
        ),
        "gemini-text": lambda: evaluate_text(
            "gemini-text", get_extractor("gemini", settings), samples, ocr_by_key
        ),
        "groq-text": lambda: evaluate_text(
            "groq-text", get_extractor("groq", settings), samples, ocr_by_key
        ),
    }
    vision_systems: dict[str, Callable[[], SystemResult]] = {
        "gemini-vision": lambda: evaluate_vision(
            "gemini-vision", get_vision_extractor("gemini", settings), samples
        ),
        "groq-vision": lambda: evaluate_vision(
            "groq-vision", get_vision_extractor("groq", settings), samples
        ),
    }

    results: list[SystemResult] = []
    for name, run in {**text_systems, **vision_systems}.items():
        print(f"Running {name}...")
        try:
            result = run()
        except Exception as error:  # one bad system (quota/key) must not kill the run
            print(f"  SKIPPED {name}: {type(error).__name__}: {str(error)[:200]}")
            continue
        results.append(result)
        print(
            f"  overall={result.overall_acc:.1%} vendor={result.vendor_acc:.1%} "
            f"date={result.date_acc:.1%} total={result.total_acc:.1%} "
            f"lat={result.mean_latency_s:.2f}s errors={result.errors}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(results, args.out_dir / "results.csv")
    _write_markdown(results, len(samples), args.out_dir / "RESULTS.md")
    print(f"\nWrote {args.out_dir}/results.csv and {args.out_dir}/RESULTS.md")


def _write_csv(results: list[SystemResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "system",
                "n",
                "vendor_acc",
                "date_acc",
                "total_acc",
                "overall_acc",
                "mean_latency_s",
                "errors",
                "cost_per_1k_usd_est",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.system,
                    r.n,
                    f"{r.vendor_acc:.4f}",
                    f"{r.date_acc:.4f}",
                    f"{r.total_acc:.4f}",
                    f"{r.overall_acc:.4f}",
                    f"{r.mean_latency_s:.3f}",
                    r.errors,
                    f"{_cost_per_1k(r.system):.2f}",
                ]
            )


def _write_markdown(results: list[SystemResult], n: int, path: Path) -> None:
    lines = [
        "# Benchmark results",
        "",
        f"SROIE test subsample (n={n}). Field-level normalized exact match on vendor/date/total.",
        "Cloud runs used free tiers; cost is a documented list-price-equivalent estimate.",
        "",
        "| System | Overall | Vendor | Date | Total | Latency/doc | Cost/1k (est) | Errors |",
        "|--------|--------:|-------:|-----:|------:|------------:|--------------:|-------:|",
    ]
    for r in results:
        cost = _cost_per_1k(r.system)
        cost_str = "$0.00" if cost == 0 else f"${cost:.2f}"
        lines.append(
            f"| {r.system} | {r.overall_acc:.1%} | {r.vendor_acc:.1%} | {r.date_acc:.1%} | "
            f"{r.total_acc:.1%} | {r.mean_latency_s:.2f}s | {cost_str} | {r.errors} |"
        )

    best_local = max(
        (r for r in results if r.system.startswith("local")),
        key=lambda r: r.overall_acc,
        default=None,
    )
    best_vision = max(
        (r for r in results if r.system.endswith("vision")),
        key=lambda r: r.overall_acc,
        default=None,
    )
    if best_local is not None and best_vision is not None and best_vision.overall_acc > 0:
        relative = best_local.overall_acc / best_vision.overall_acc
        lines += [
            "",
            f"**Headline:** local ({best_local.system}) reaches **{relative:.0%}** of the best "
            f"cloud-vision system ({best_vision.system}) — at **$0** and fully offline "
            f"({best_local.overall_acc:.1%} vs {best_vision.overall_acc:.1%} overall).",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
