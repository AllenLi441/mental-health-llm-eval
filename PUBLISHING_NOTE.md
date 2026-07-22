# mental-health-llm-eval

Zero-dependency Node harness for zero-shot LLM evaluation on 8 public mental-health
benchmarks (CPsyExam, EmoBench, IMHI×9, PsySUICIDE, MentalManip, CBT-Bench, MDD-5k, EATD).

Standalone research module — **not** part of any product codebase.

## Code + aggregate or redacted evidence only
Excluded (license / sensitivity): raw benchmark datasets, private row-level `results/*.jsonl`
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

## CPsyExam V4 full paired evidence

The confirmatory comparison was frozen by the [2026-07-22 preregistration Release](https://github.com/AllenLi441/mental-health-llm-eval/releases/tag/cpsyexam-v4-prereg-2026-07-22) before inference. Both arms cover the same 3,902 cases: v4-pro scored 3,307/3,902 (84.7514%; 17 invalid, 0 API errors) and v4-flash scored 3,252/3,902 (83.3419%; 0 invalid, 0 API errors). The paired table is 3,088 both correct, 219 v4-pro only, 164 v4-flash only, and 431 neither correct.

Under the preregistered analysis, v4-pro minus v4-flash is +1.4095pp (paired normal 95% CI 0.4274–2.3917pp; 20,000-repetition paired bootstrap 95% CI 0.4357–2.4090pp), with exact two-sided McNemar `p=0.00572255`. This supports “v4-pro was significantly more accurate than v4-flash under the preregistered protocol.” The ±2pp TOST did not establish equivalence (`p_lower=5.086e-12`, `p_upper=0.119332`), so “statistical tie” and “equivalent” are not supported.

The repository summary is `results-summary/cpsyexam-v4-full-paired.summary.json`. A redacted evidence package is planned for [`cpsyexam-v4-full-2026-07-22`](https://github.com/AllenLi441/mental-health-llm-eval/releases/tag/cpsyexam-v4-full-2026-07-22). At the time of this edit, publication and asset verification are pending; do not describe that result Release as immutable or verified until GitHub reports the corresponding final state and verification succeeds.

The public package may contain blinded per-case commitments and paired correctness outcomes, aggregate inference, manifests, checksums, and a notice. It must not contain question text, options, source IDs, gold labels, model predictions, raw outputs, API keys, or key fingerprints. The private input JSONL files remain excluded. Provenance is anchored to runner commit `3e6890374cb39631bb1cc8bca46ef4835df85446`, public case commitment manifest SHA-256 `1275ce7edeb55ad62500ac1692b82bef3800592decc2ece4d615fc8770232c9d`, and the exact `dataset.revision` string recorded in the public summary.
