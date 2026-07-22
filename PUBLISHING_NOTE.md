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
node run.mjs all --selftest
EVAL_DATASETS_DIR=/authorized/datasets node run.mjs all --data-check
node run.mjs imhi-dreaddit --model deepseek-v4-flash --thinking disabled --run-id demo
python3 scripts/scoreboard.py
```

Set `EVAL_DATASETS_DIR` to an authorized local dataset root for real runs. The standalone selftest uses repository fixtures and intentionally does not claim metric reproduction.

## Reporting boundary

Historical result snapshots remain in `reports/`; current claims must be regenerated from aggregate summaries plus `reports/BASELINES.json`. Benchmark performance is not clinical validation, and raw-result recomputation requires the locally authorized JSONL files that are deliberately excluded here.

In an authorized environment, `python3 scripts/audit_results.py --manifest-out <reviewed-path>` can emit a redacted evidence manifest containing file hashes, row/unique/duplicate/error counts, field coverage, and aggregate model/provider attribution. Review that manifest before publishing it. It deliberately excludes row ids, source text, gold text, prompts, and model outputs; therefore it improves provenance auditing but does not make row-level claims independently reproducible.
