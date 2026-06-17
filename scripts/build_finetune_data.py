"""Build an instruction dataset (OCR text -> JSON fields) from SROIE for MLX QLoRA.

Writes data/finetune/{train,valid}.jsonl in MLX chat format. SROIE 'train' becomes
train+valid (90/10); SROIE 'test' is left untouched for evaluation. The system/user
prompt matches the runtime extractor so the tuned model is good at exactly that task.

uv run python scripts/build_finetune_data.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from shoebox.benchmark.data import SROIE_DATASET
from shoebox.benchmark.scoring import parse_date, parse_total
from shoebox.extract import _SYSTEM_PROMPT, _build_user_prompt


def _example(
    text: str, company: str | None, date_raw: str | None, total_raw: str | None
) -> dict[str, object]:
    iso = parse_date(date_raw)
    total = parse_total(total_raw)
    target = {
        "vendor": company or None,
        "date": iso.isoformat() if iso is not None else None,
        "total": float(total) if total is not None else None,
        "currency": None,
    }
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(text)},
            {"role": "assistant", "content": json.dumps(target)},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/finetune"))
    parser.add_argument("--valid-frac", type=float, default=0.1)
    args = parser.parse_args()

    split = load_dataset(SROIE_DATASET)["train"]
    seen: set[str] = set()
    examples: list[dict[str, object]] = []
    for index in range(len(split)):
        row = split[index]
        text = " ".join(str(word) for word in (row.get("words") or [])).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        entities = row.get("entities") or {}
        examples.append(
            _example(text, entities.get("company"), entities.get("date"), entities.get("total"))
        )

    cut = max(1, int(len(examples) * (1 - args.valid_frac)))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(examples[:cut], args.out_dir / "train.jsonl")
    _write_jsonl(examples[cut:], args.out_dir / "valid.jsonl")
    print(f"train={cut} valid={len(examples) - cut} -> {args.out_dir}")


def _write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
