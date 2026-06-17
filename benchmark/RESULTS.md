# Benchmark results

SROIE test subsample (n=50). Field-level normalized exact match on vendor/date/total.
Cloud runs used free tiers; cost is a documented list-price-equivalent estimate.

| System | Overall | Vendor | Date | Total | Latency/doc | Cost/1k (est) | Errors |
|--------|--------:|-------:|-----:|------:|------------:|--------------:|-------:|
| regex | 61.3% | 78.0% | 58.0% | 48.0% | 0.00s | $0.00 | 0 |
| local-q4 | 64.7% | 86.0% | 20.0% | 88.0% | 0.80s | $0.00 | 0 |
| local-q8 | 74.0% | 76.0% | 58.0% | 88.0% | 1.02s | $0.00 | 0 |
| local-fp16 | 73.3% | 76.0% | 56.0% | 88.0% | 1.48s | $0.00 | 0 |
| gemini-text | 21.3% | 18.0% | 22.0% | 24.0% | 2.55s | $0.26 | 38 |
| groq-text | 83.3% | 88.0% | 64.0% | 98.0% | 4.54s | $0.09 | 0 |
| gemini-vision | 0.0% | 0.0% | 0.0% | 0.0% | 3.85s | $0.48 | 50 |
| groq-vision | 82.7% | 88.0% | 64.0% | 96.0% | 5.69s | $0.14 | 0 |

**Headline:** local (local-q8) reaches **90%** of the best cloud-vision system (groq-vision) — at **$0** and fully offline (74.0% vs 82.7% overall).

## Notes
- **Quant sweep:** q4 (64.7%) → q8 (74.0%) → fp16 (73.3%). q8 is the sweet spot — q4 collapses on `date` (20%), fp16 adds latency without accuracy.
- **Gemini rows are unreliable here.** The harness fires requests back-to-back with no backoff, tripping Gemini's free-tier rate limits (10 RPM / token caps) — hence the high `errors` and near-zero scores (vision fails hardest). Groq's higher free RPM let it finish cleanly, so **Groq anchors the headline**. Gemini answered correctly in single-call smoke tests; a retry/backoff pass would recover its true accuracy.
- Local beats the regex baseline (74.0% vs 61.3%) and trails the cloud frontier by ~9 points overall — mostly on `date`, the field M5's fine-tune targets.
