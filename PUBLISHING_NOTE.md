# mental-health-llm-eval

Zero-dependency Node harness for zero-shot LLM evaluation on 8 public mental-health
benchmarks (CPsyExam, EmoBench, IMHI×9, PsySUICIDE, MentalManip, CBT-Bench, MDD-5k, EATD).

Standalone research module — **not** part of any product codebase.

## Code + aggregate results only
Excluded (license / sensitivity): raw benchmark datasets, row-level `results/*.jsonl`
(real social-media / crisis text / model outputs), `.env` (API key). Obtain each dataset
from its original source (links in `reports/` and the paper); put your key in local `.env`.

## Run
```bash
node run.mjs list
node run.mjs imhi-dreaddit --model deepseek-chat --run-id demo
python3 scripts/scoreboard.py
```

## Honest conclusion: see reports/imhi_uniform_v3u.md
Uniform de-biased protocol → 3 statistically-significant wins vs GPT-4-class
(EmoBench-EU / MentalManip / CPsyExam); IMHI 0/9 clearly beats fine-tuned SOTA.
Not clinical validation.
