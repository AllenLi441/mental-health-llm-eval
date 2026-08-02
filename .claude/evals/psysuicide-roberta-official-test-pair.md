# EVAL DEFINITION: PsySUICIDE supervised-v1 official-test pair

## Capability checks

- The frozen Seed 43 checkpoint is scored on all 1,464 retained official-test rows.
- The historical V4-Pro + taxonomy reference is accepted only when its JSONL SHA-256, model identity, prompt identity, fingerprint, and recomputed metrics match the preregistration.
- Candidate and reference rows match exactly on normalized ID, gold label, case commitment, and dataset manifest.
- Accuracy, macro-F1, weighted-F1, per-class metrics, paired macro-F1 randomization/bootstrap, and exact McNemar are recomputed from the paired rows.
- The public result contains aggregate evidence only; candidate row predictions are never persisted.

## Freeze and execution gates

- `reports/psysuicide-roberta-v1-official-test-paired.prereg.json` and `scripts/score_psysuicide_roberta_official_test_pair.py` must be committed and pushed before `--execute`.
- `--execute` requires the full freeze commit SHA and rejects a local-upstream-live-remote mismatch.
- The checkpoint root and all five inference files must match the previously frozen selection manifest.
- A private `0600` exclusive claim is created before official-test candidate inference.
- Existing claim, completion, or public-result files block overwrite and rerun.

## Success criteria

- Completeness: 1,464/1,464 paired rows, no missing or duplicate normalized identities.
- Integrity: reference JSONL SHA-256 and candidate checkpoint SHA-256 match the preregistration.
- Primary inference: apply the preregistered macro-F1 superiority rule; do not call a non-significant result a tie or equivalence.
- Secondary inference: report the accuracy delta in percentage points with exact two-sided McNemar evidence.
- Reporting: write the aggregate JSON result and update the existing model-optimization report with the same-split conclusion and limitations.
- Publication: relevant tests pass; only intended public files are committed and pushed.

## Regression checks

- Existing PsySUICIDE trainer, reporter, holdout scorer, paired analyzer, harness-safety, scoreboard, and project-ledger checks remain green.
- Licensed data, credentials, model weights, private claims, and row-level predictions remain ignored and unstaged.
