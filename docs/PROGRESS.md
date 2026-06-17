# Build progress & blockers log

Autonomous build of M2–M6 (2026-06-17). Milestones leave a presentable repo; blockers
are logged here and worked around rather than stalling.

## Status
- **M1 — OCR ledger CLI** ✅ shipped (OpenCV → PaddleOCR → CSV + regex baseline)
- **M2 — local + cloud extractors** ✅ `Extractor` protocol with `RegexExtractor`,
  `LocalExtractor` (Ollama/Qwen), `OpenAICompatExtractor` (Gemini + Groq); `--extractor` CLI
  flag; live-verified against real Ollama.
- **M3 — labeled benchmark** ⏳
- **M4 — category classifier** ⏳
- **M5 — QLoRA fine-tune + publish** ⏳
- **M6 — dashboard from handoff** ⏳

## Blockers / decisions (flag-and-continue)
- **M2 local model is imperfect (expected, not a bug).** On synthetic receipts Qwen 3B
  q4_K_M nails the clean one but misses some dates and guessed a wrong currency, and OCR
  splits multi-word vendors. This is the honest baseline the project exists to measure
  (M3) and improve (M5 fine-tune) — prompt deliberately NOT overfit to the fixtures.
