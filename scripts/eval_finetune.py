"""Evaluate a Qwen model (base or QLoRA-tuned) on SROIE test field accuracy, via MLX.

uv run python scripts/eval_finetune.py --label base
uv run python scripts/eval_finetune.py --label tuned --adapter-path adapters

Prints a JSON line with field-level accuracy so base vs tuned can be compared.
"""

from __future__ import annotations

import argparse
import json
import re

from datasets import load_dataset
from mlx_lm import generate, load

from shoebox.benchmark.data import SROIE_DATASET
from shoebox.benchmark.scoring import aggregate, score_one
from shoebox.extract import _SYSTEM_PROMPT, _build_user_prompt
from shoebox.models import ReceiptFields

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse(text: str) -> ReceiptFields:
    match = _JSON_RE.search(text)
    if match is None:
        return ReceiptFields()
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return ReceiptFields()
    return ReceiptFields.model_validate(data) if isinstance(data, dict) else ReceiptFields()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="mlx-community/Qwen2.5-3B-Instruct-4bit")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--label", default="model")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    loaded = (
        load(args.base, adapter_path=args.adapter_path) if args.adapter_path else load(args.base)
    )
    model, tokenizer = loaded[0], loaded[1]

    test = load_dataset(SROIE_DATASET)["test"]
    count = min(args.limit, len(test))
    scored = []
    for index in range(count):
        row = test[index]
        text = " ".join(str(word) for word in (row.get("words") or [])).strip()
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(text)},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        output = generate(model, tokenizer, prompt=prompt, max_tokens=128, verbose=False)
        entities = row.get("entities") or {}
        scored.append(
            score_one(
                _parse(output), entities.get("company"), entities.get("date"), entities.get("total")
            )
        )

    result = aggregate(args.label, scored, [], 0)
    print(
        json.dumps(
            {
                "label": args.label,
                "n": result.n,
                "overall": round(result.overall_acc, 4),
                "vendor": round(result.vendor_acc, 4),
                "date": round(result.date_acc, 4),
                "total": round(result.total_acc, 4),
            }
        )
    )


if __name__ == "__main__":
    main()
