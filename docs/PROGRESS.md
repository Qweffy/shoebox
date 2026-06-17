# Build progress & blockers log

Autonomous build of M2–M6 (2026-06-17). Milestones leave a presentable repo; blockers
are logged here and worked around rather than stalling.

## Status
- **M1 — OCR ledger CLI** ✅ shipped (OpenCV → PaddleOCR → CSV + regex baseline)
- **M2 — local + cloud extractors** ✅ `Extractor` protocol with `RegexExtractor`,
  `LocalExtractor` (Ollama/Qwen), `OpenAICompatExtractor` (Gemini + Groq); `--extractor` CLI
  flag; live-verified against real Ollama.
- **M3 — labeled benchmark** ✅ 50-doc SROIE run done. **Headline: local q8 = 90% of
  groq-vision, $0, offline** (74.0% vs 82.7%). Quant sweep q4 64.7% → q8 74.0% → fp16 73.3%.
- **M4 — category classifier** ✅ TF-IDF + LinearSVC, vendor-grouped stratified 5-fold CV;
  **macro-F1 0.487 vs 0.088 majority** on 987 SROIE receipts (weak keyword labels).
- **M5 — QLoRA fine-tune + publish** ⏳
- **M6 — dashboard from handoff** ⏳

## Blockers / decisions (flag-and-continue)
- **M2 local model is imperfect (expected, not a bug).** On synthetic receipts Qwen 3B
  q4_K_M nails the clean one but misses some dates and guessed a wrong currency, and OCR
  splits multi-word vendors. This is the honest baseline the project exists to measure
  (M3) and improve (M5 fine-tune) — prompt deliberately NOT overfit to the fixtures.
- **M3 Gemini rate-limited (flagged, not fixed).** The benchmark has no backoff, so 50 rapid
  calls trip Gemini's free-tier limits (10 RPM) → most Gemini text/vision calls errored and
  scored ~0. Groq (higher RPM) completed cleanly and anchors the headline. Fix later: add
  retry/backoff or throttle. Gemini works in single-call smoke tests.
- **M4 labels are weak (keyword-derived), documented.** No public receipt dataset has
  expense categories and Nico opted out of personal labels, so categories come from a
  keyword rule (`classifier/labels.py`). M4 measures whether TF-IDF generalizes that weak
  signal under vendor-grouped CV (it does: 0.487 macro-F1 ≫ 0.088 majority) — a methodology
  demo, not a gold-accuracy claim. The keyword rule doubles as the rule-based baseline.
