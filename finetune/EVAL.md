# M5 — QLoRA fine-tune evaluation

QLoRA fine-tune of **Qwen2.5-3B-Instruct (4-bit)** via `mlx_lm.lora` on 561 SROIE-derived
instruction examples (OCR text → JSON fields), 300 iters, `--mask-prompt`, `--num-layers 8`.
Training val loss: **0.965 → 0.018**.

## Tuned vs base (SROIE test, n=30, field-level normalized exact match)

| Model | Overall | Vendor | Date | Total |
|-------|--------:|-------:|-----:|------:|
| Base Qwen2.5-3B (prompted) | 72.2% | 83.3% | 53.3% | 80.0% |
| **QLoRA fine-tune** | **80.0%** | **90.0%** | **60.0%** | **90.0%** |
| Δ | **+7.8** | +6.7 | +6.7 | +10.0 |

The fine-tune improves every field — biggest gains on `total` (+10) and `vendor` (+6.7),
and it narrows the gap to the cloud frontier (Groq ~82.7% overall in the M3 benchmark).

## Published

Merged weights + model card on the HF Hub: **[Qweffy/shoebox-receipts-qwen3b](https://huggingface.co/Qweffy/shoebox-receipts-qwen3b)**.

Reproduce:
```bash
uv run python scripts/build_finetune_data.py
uv run mlx_lm.lora --model mlx-community/Qwen2.5-3B-Instruct-4bit --train --data data/finetune \
  --iters 300 --batch-size 4 --num-layers 8 --adapter-path adapters --grad-checkpoint --mask-prompt
uv run python scripts/eval_finetune.py --label base
uv run python scripts/eval_finetune.py --label tuned --adapter-path adapters
uv run python scripts/publish_model.py --repo-name shoebox-receipts-qwen3b
```

## Follow-up
Running the fine-tune back inside **Ollama** needs a GGUF conversion (MLX `--export-gguf`
supports only Llama/Mistral/Mixtral, not Qwen) via llama.cpp — documented as a follow-up.
The MLX weights are directly usable via `mlx_lm` (see the model card).
