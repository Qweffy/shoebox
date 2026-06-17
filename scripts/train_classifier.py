"""Train the expense-category classifier and write a metrics report.

uv run python scripts/train_classifier.py [--out-dir classifier]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shoebox.classifier.dataset import build_dataset
from shoebox.classifier.train import TrainReport, train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("classifier"))
    args = parser.parse_args()

    print("Building weak-labeled dataset from SROIE...")
    data = build_dataset()
    print(f"  {len(data)} receipts")

    report = train_and_evaluate(data)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_report(report, args.out_dir)

    print(f"macro-F1 = {report.macro_f1:.3f}  (majority baseline {report.majority_macro_f1:.3f})")
    print(f"Wrote {args.out_dir}/metrics.md and {args.out_dir}/metrics.json")


def _write_report(report: TrainReport, out_dir: Path) -> None:
    header = "| gt \\ pred | " + " | ".join(report.classes) + " |"
    separator = "|---" * (len(report.classes) + 1) + "|"
    rows = [
        f"| **{report.classes[i]}** | " + " | ".join(str(v) for v in row) + " |"
        for i, row in enumerate(report.confusion)
    ]
    lines = [
        "# Expense-category classifier",
        "",
        "TF-IDF (1-2 grams) + LinearSVC over SROIE receipt text. **Vendor-grouped, stratified "
        f"{report.n_splits}-fold CV** (no vendor appears in both train and test → leakage-safe). "
        f"n={report.n} receipts, classes={report.classes}.",
        "",
        "> **Labels are weak** (keyword-derived; see `labels.py`). This measures whether TF-IDF "
        "generalizes the weak signal under honest CV, not gold-validated accuracy.",
        "",
        f"- **Macro-F1: {report.macro_f1:.3f}**",
        f"- Majority-class baseline macro-F1: {report.majority_macro_f1:.3f}",
        "",
        "## Class distribution",
        "",
        *[f"- {category}: {count}" for category, count in sorted(report.class_counts.items())],
        "",
        "## Per-class report",
        "",
        "```",
        report.report_text,
        "```",
        "",
        "## Confusion matrix",
        "",
        header,
        separator,
        *rows,
        "",
    ]
    (out_dir / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "n": report.n,
                "classes": report.classes,
                "class_counts": report.class_counts,
                "macro_f1": report.macro_f1,
                "majority_macro_f1": report.majority_macro_f1,
                "confusion": report.confusion,
                "n_splits": report.n_splits,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
