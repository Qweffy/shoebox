"""Fuse the QLoRA adapter into a merged model and publish it to the HF Hub.

Requires HF_TOKEN in .env and a trained adapter at --adapter-path. Writes an honest
model card, then uploads. Run after eval so the card can quote real numbers.

uv run python scripts/publish_model.py --repo-name shoebox-receipts-qwen3b \
  --tuned-overall 0.80 --base-overall 0.74
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

_CARD = """---
license: apache-2.0
base_model: Qwen/Qwen2.5-3B-Instruct
tags: [receipt, information-extraction, qlora, mlx, shoebox]
---

# shoebox — receipt field extractor (Qwen2.5-3B QLoRA)

A QLoRA fine-tune of **Qwen2.5-3B-Instruct** (4-bit, MLX) for extracting structured
fields from receipt OCR text — part of the [shoebox](https://github.com/Qweffy/shoebox)
project (a privacy-first, 100%-on-device receipt digitizer).

Given a receipt's OCR text, it returns JSON: `vendor`, `date` (ISO), `total`, `currency`.

## Results (SROIE test, field-level normalized exact match)

| Model | Overall field accuracy |
|-------|-----------------------:|
| Base Qwen2.5-3B (prompted) | {base_overall} |
| This QLoRA fine-tune | {tuned_overall} |

Measured on held-out SROIE receipts. See the repo's `benchmark/RESULTS.md` for the full
local-vs-cloud comparison.

## Training

- **Data:** {n_train} instruction examples derived from the SROIE *train* split
  (OCR text → JSON fields), built by `scripts/build_finetune_data.py`.
- **Method:** QLoRA via `mlx_lm.lora` on a 4-bit base, `--mask-prompt`, {iters} iters.
- **Hardware:** Apple Silicon (MLX).

## Intended use & limitations

- **Use:** local receipt extraction in the shoebox CLI/dashboard; research on small-model
  document extraction.
- **Limitations:** trained on SROIE (largely Malaysian retail receipts); weakest on `date`
  formats and unusual layouts. **Not** a production OCR/extraction service. Labels for the
  category classifier elsewhere in shoebox are weak (keyword-derived) — see the repo.

## Usage (MLX)

```python
from mlx_lm import generate, load
model, tok = load("{repo_id}")
prompt = tok.apply_chat_template(
    [{{"role": "system", "content": "...extract vendor/date/total/currency as JSON..."}},
     {{"role": "user", "content": "OCR TEXT:\\n<receipt text>"}}],
    add_generation_prompt=True, tokenize=False)
print(generate(model, tok, prompt=prompt, max_tokens=128))
```
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="mlx-community/Qwen2.5-3B-Instruct-4bit")
    parser.add_argument("--adapter-path", default="adapters")
    parser.add_argument("--save-path", type=Path, default=Path("models/shoebox-receipts-qwen3b"))
    parser.add_argument("--repo-name", default="shoebox-receipts-qwen3b")
    parser.add_argument("--tuned-overall", default="—")
    parser.add_argument("--base-overall", default="—")
    parser.add_argument("--n-train", default="561")
    parser.add_argument("--iters", default="300")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set in .env")

    subprocess.run(
        [
            "uv",
            "run",
            "mlx_lm.fuse",
            "--model",
            args.base,
            "--adapter-path",
            args.adapter_path,
            "--save-path",
            str(args.save_path),
        ],
        check=True,
    )

    api = HfApi(token=token)
    repo_id = f"{api.whoami()['name']}/{args.repo_name}"
    card = _CARD.format(
        base_overall=args.base_overall,
        tuned_overall=args.tuned_overall,
        n_train=args.n_train,
        iters=args.iters,
        repo_id=repo_id,
    )
    (args.save_path / "README.md").write_text(card, encoding="utf-8")

    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(args.save_path), repo_id=repo_id)
    print(f"published https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
