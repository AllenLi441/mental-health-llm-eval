# EVAL DEFINITION: cross-benchmark accuracy family v1

## Objective

Build a reusable benchmark-spec registry and generic classification trainer,
then use them to run a leakage-safe four-task IMHI single-Seed screen while
separately evaluating the 2026-07-31 DeepSeek V4-Flash snapshot with the frozen
PsySUICIDE taxonomy prompt on full official valid. The objective is higher
task-specific primary metrics under the same examples and protocol, not one
universal head or an unpaired cross-split score claim.

## Frozen execution order and scope

1. Define and validate the benchmark spec registry.
2. Implement and validate the generic classification trainer offline.
3. Run bounded V4-Flash taxonomy smoke, then full PsySUICIDE valid and paired
   analysis if every smoke gate passes.
4. Run the IMHI DR, dreaddit, IRF, and SAD Seed-42 screen only after the first
   two capability groups pass.
5. Keep PsySUICIDE accuracy-first v2 Arms B/C/D paused throughout this eval.

No official/frozen test split may be used for hyperparameter, loss, prompt,
threshold, checkpoint, or arm selection. Single-Seed screen scores are
development evidence only. Licensed text, row IDs, row-level gold/predictions,
raw model outputs, credentials, logits, weights, and private diagnostics remain
outside Git and public logs.

## Capability evals

### A. Benchmark spec registry

- [x] A versioned machine-readable registry has one canonical spec per supported
      task and rejects duplicate or unknown task IDs.
- [x] Each spec freezes dataset/revision identity, permitted train/dev/test use,
      split manifests or commitments, label mapping/order, task type, primary and
      guard metrics, seed policy, base-model/tokenizer revision fields, and public
      versus private artifact boundaries.
- [x] PsySUICIDE API evaluation and IMHI supervised classification are explicitly
      different experiment families; they do not share an output head or claim.
- [x] DR, dreaddit, IRF, and SAD each have an independent label space, head,
      checkpoint namespace, and aggregate result identity.
- [x] Registry validation fails closed on missing required fields, mutable model
      aliases, forbidden split use, inconsistent label counts/order, metric/task
      mismatch, duplicate output paths, or a repository path for private data.
- [x] Registry schema/selftests are deterministic, offline, and use only synthetic
      fixtures; normal Python and `python -O` both pass if Python implements it.

Result: registry validation passed 19/19 runner tasks. Its 17/17 synthetic
selftests passed under normal Python and `python -O`; the canonical bundle
SHA-256 is
`d1f38dc38cd3745c8c84edc12390609e35580f8304322607424e5e16d366433b`.

### B. Generic classification trainer

- [x] The trainer consumes only a validated registry spec and explicitly supplied
      authorized train/inner-dev inputs; it cannot silently open official test,
      PsySUICIDE valid/test, or the PsySUICIDE v2 private partition.
- [x] Default dry-run performs no training, model download, API call, external-data
      read, or result-row write. The separately explicit `--validate-data` mode
      reads/hashes only declared train/valid data and still performs no training.
- [x] Execution binds task, spec hash, split commitments, exact base model and
      tokenizer revisions, loss arm, Seed, code/Git identity, and run ID.
- [x] The same trainer supports the preregistered `ce`, `weighted-ce`, and `focal`
      arms without sharing classifier heads or checkpoints between tasks.
- [x] Checkpoint selection uses unrounded inner-dev weighted-F1, with a frozen
      deterministic tie-break; resume and best-checkpoint loads are strict.
- [x] New runs refuse overwrite; resume rejects any task/spec/split/model/loss/
      Seed/code mismatch.
- [x] Shared `execute_training` independently reloads and validates the live
      registry instead of trusting an adapter/nonempty proof. It binds the exact
      checker/hash bundle, task/profile/config/protocol, source paths and rows,
      split SHA commitments, and run plan; forged bundle proof is rejected.
- [x] MPS resume requires a strict `mps_rng_state.pth` artifact, saves/restores
      it exactly, and chooses the latest device-complete checkpoint rather than a
      higher step with missing or corrupt MPS RNG state.
- [x] Public outputs contain aggregate metrics, counts, configuration, hashes,
      elapsed resource use, and failure status only. Private ignored artifacts
      may contain row-level diagnostics but never source text or source IDs.
- [x] Synthetic selftests cover binary and multiclass tasks, imbalance handling,
      invalid labels, empty classes, deterministic metrics, checkpoint mismatch,
      forbidden paths/splits, overwrite, and resume identity.
- [x] Offline selftests pass under normal execution and optimized assertions; no
      command is called successful merely because a dependency is unavailable.

Result: generic trainer and fixed IMHI adapter selftests passed under normal
Python and `python -O`, with no model download. The tests execute binary and
three-class CE/weighted-CE/focal loss, imbalanced weights, invalid-label and
empty-class rejection, deterministic repeated metrics, strict resume, and
checkpoint-epoch derivation. An explicit real-data proof audit validated all
12 registry-bound plans and rejected a forged bundle before model loading. The
initial Transformers 5 `save_safetensors` keyword `TypeError` and full-input
leakage-key defect were fixed before any IMHI metric-bearing training run. An
actual MPS identical-random-tensor roundtrip also passed, including
device-aware fallback from incomplete higher checkpoints.

### C. 2026-07-31 V4-Flash + taxonomy on PsySUICIDE valid

- [x] Reference is the completed full-valid baseline with `n=1,459`, Seed `42`,
      dataset manifest SHA-256
      `89bb98ae101f176c1e125a60a6d7ab23f7467c22a77698a0d9668c665dfffc59`,
      requested/response/API model `deepseek-v4-flash`, and fingerprint
      `fp_a18b46594c_prod0820_fp8_kvcache_20260402`.
- [x] Candidate changes only the prompt profile to the already defined frozen
      `taxonomy` profile; retained cases, parser/scorer, model snapshot, Seed,
      decoding controls, and dataset manifest remain identical.
- [x] Execution is dry-run by default, requires an explicit approved budget, and
      records exact requested/response/API model identity, fingerprint, prompt
      hash, manifest hash, usage, error count, invalid count, and cost.
- [x] A bounded smoke runs before full valid. Any model/fingerprint/manifest
      mismatch, API error, insufficient reserve, or invalid rate above `0.5%`
      stops execution before full valid.
- [x] Full valid contains exactly one terminal result for every one of the same
      1,459 retained reference cases; strict-parser invalids count as incorrect.
- [x] Paired analysis verifies identical ID/gold commitments and reports accuracy,
      macro-F1, weighted-F1, invalid/error rates, per-class support/recall/F1,
      paired macro-F1 bootstrap/randomization, and secondary McNemar accuracy.
- [ ] `FAIL / REJECTED`: candidate advances from valid only if macro-F1 delta is at least `+0.01`,
      accuracy delta is at least `-0.005`, every preregistered critical class with
      support at least 10 has recall delta at least `-0.05`, invalid rate is at
      most `0.5%`, and there are no API errors.
- [x] No official test, internal holdout, or PsySUICIDE v2 arm is opened or scored.

Result: smoke `n=50` reached accuracy `0.92`, invalid `0`, errors `0`. Full
valid `n=1,459` reached accuracy `0.8779986292`, macro-F1 `0.5318369277`,
weighted-F1 `0.8809387821`, invalid `16`, errors `0`. Relative to the matched
baseline, deltas were `+0.054832` accuracy, `+0.051654` macro-F1, and
`+0.044373` weighted-F1; McNemar `p=4.0268e-08`. The candidate was rejected:
invalid rate was about `1.10%`, the critical-recall guard failed, and macro-F1
paired inference was not confirmatory (randomization `p=0.06360`; bootstrap
95% CI `[-0.00619, 0.10574]`). No test run is authorized from this result.
The historical harness commit was
`479e5c03d3626bccdd1d0882c161daebc173d7db`, taxonomy prompt SHA-256 was
`eb2180f8305316f4cc2c07a810d137901fc62ab94e18e68392b236a2b869d1c8`, and
the shared response fingerprint was
`fp_a18b46594c_prod0820_fp8_kvcache_20260402`.

### D. IMHI four-task Seed-42 screen

- [x] Screen task set is exactly DR, dreaddit, IRF, and SAD; no task is added or
      removed after metrics are observed.
- [x] Each task uses only its frozen authorized train/inner-dev membership. Test
      files and historical sampled-test scores are not trainer inputs or selectors.
- [x] Base-model/tokenizer revisions, preprocessing, max length, optimizer,
      epochs, evaluation cadence, early-stop/tie rule, and all three loss arms are
      frozen before the first metric-bearing run.
- [x] The planned matrix is exactly four tasks by three arms (`ce`, `weighted-ce`,
      `focal`) at Seed `42`; smoke/subset metrics are integrity-only.
- [x] An execute preflight attempt writes aggregate-only `ATTEMPTED` state and,
      on failure, atomically records `FAILED/PREFLIGHT` before exit 1 without
      persisting the raw error message.
- [x] `--resume` without `--execute` exits 1 instead of silently becoming a
      dry-run; resume execution remains subject to the same preflight gates.
- [ ] Every run validates spec/split/code/model identity and produces a complete
      aggregate or an explicit failed/incomplete status; failed arms are never
      silently omitted from ranking.
- [x] Weighted-F1 is the primary metric per task; accuracy, macro-F1, per-class
      support/precision/recall/F1, confusion aggregates, resource use, and chosen
      checkpoint epoch are guards/reporting fields.
- [ ] A task challenger is compared only with that task's fixed CE control on the
      same inner-dev rows. Historical zero-shot sampled-test and paper-test points
      remain descriptive and are not subtracted from inner-dev scores.
- [ ] The family advances to a separately preregistered three-Seed confirmation
      only if at least 3/4 tasks improve weighted-F1 by `0.02` or more and no task
      regresses by more than `0.01` relative to its fixed control.
- [x] Regardless of point values, this eval makes no stability, test superiority,
      paper superiority, cross-task universal-model, clinical, or deployment claim.

Data-validation checkpoint: explicit `--validate-data` produced a 12-job
`DRY_RUN`. Used train/valid rows were DR `1,002/430` (one cross-split train row
removed), Irf `3,941/985` (zero), SAD `5,547/616` (zero), and dreaddit
`2,814/300` (one).

Execution checkpoint: the committed screen then started, but the first DR/CE
job failed after epoch 4 / step 504. Early stopping selected checkpoint-252;
`load_best_model_at_end` hit a strict state-key mismatch because Transformers 5
serialized 50 LayerNorm parameters as `gamma/beta` while the live model expected
`weight/bias`; matching keys retained identical shape/dtype. This is a harness
compatibility failure, not a data or model-metric failure. The manifest has 0
COMPLETE, 1 FAILED, and 11 PLANNED jobs; no `result.json` or `best-model` exists.
Checkpoint-252 valid weighted-F1 `0.9084805` and accuracy `0.9093023` are
diagnostic only and cannot enter the formal screen. Execution is paused and was
not restarted.

## Regression evals

- [x] Existing 19/19 prompt/parser selftests pass.
- [x] Aggregate audit, scoreboard, baseline registry, harness safety, CPsyExam,
      PsySUICIDE, EmoBench, and existing IMHI offline selftests pass.
- [x] Existing CPsyExam historical-release and PsySUICIDE v1/v2 protocol artifacts
      remain unchanged except for append-only ledger/eval references.
- [x] PsySUICIDE official-test `88.11%` and internal-holdout `93.43%` remain
      explicitly incomparable.
- [x] CBT top-1 results remain explicitly incomparable with paper multi-label F1.
- [x] Git diff and tracked-file scans contain no secrets, licensed/raw rows, local
      absolute dataset paths, model outputs, private diagnostics, or weights.
- [x] `python3 scripts/check_project_ledger.py` passes after every material update.

## Required run log

Every implementation or experiment update must be appended to
`.claude/evals/cross-benchmark-accuracy-family-v1.log` and
`project-ledger/updates/2026-08-01-accuracy-family-execution.md` with:

- timestamp and immutable run identity;
- exact command and exit result;
- files/protocol changed;
- PASS, FAIL, BLOCKED, PAUSED, INCOMPLETE, or NOT RUN status;
- aggregate metrics or public-safe error category;
- gate decision and next action.

Failures and replacement runs remain visible. A retry uses a fresh run identity;
it never overwrites or edits a failed run into a pass.

## Human review required

- Approve paid API budget before the taxonomy smoke/full valid run.
- Review any change to task labels, class weights, critical-class guards, or
  decision thresholds before metric-bearing execution.
- Review rare-class data expansion separately; AI-produced examples or labels
  must never be presented as human gold.
- Review any future test access, deployment, or continuation of PsySUICIDE v2
  B/C/D under a separate authorization and preregistration.
