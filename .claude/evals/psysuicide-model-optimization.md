# EVAL DEFINITION: PsySUICIDE model optimization protocol

## Objective

Improve the model-facing PsySUICIDE classifier without changing labels, deleting
hard cases, or tuning on the official test split.

## Capability evals

- [x] `psysuicide` requires an explicit `--split` for scored runs.
- [x] `train`, `valid`, and `test` load from distinct official files and produce
      globally unique normalized case IDs.
- [x] Authorized-data check rejects exact retained-text overlap across official
      splits.
- [x] `baseline`, `taxonomy`, and `hierarchical` prompt profiles pass the
      standalone prompt/parser selftest without loading licensed data.
- [x] Every result row and summary records split, prompt profile, requested
      model, prompt hash, dataset hash, seed, and run ID.
- [x] Deprecated DeepSeek routing aliases are rejected before any API call.
- [x] A test run is rejected unless `--confirm-test` and a matching frozen
      preregistration artifact are both supplied.
- [x] `--prepare-prereg` writes the exact test-run identity without making an
      API call or writing result rows.
- [x] Resume rejects rows from a different split, prompt profile, model, seed,
      prompt hash, or dataset hash.
- [x] A new run refuses to append to an existing result path unless `--resume`
      is explicit.
- [x] The valid-matrix launcher is dry-run by default and requires both
      `--execute` and an approved budget for paid calls.
- [x] Full valid launch is rejected until all four fixed 50-case smoke summaries
      have zero API errors, exact response-model identity, and matching protocol
      metadata.
- [x] The paired analyzer validates identical complete ID/gold sets, reports all
      four arms, runs only the three prespecified contrasts, applies Holm
      correction, and selects by the frozen metric order.
- [x] Result summaries aggregate token usage without exposing credentials or
      source text.
- [x] Confirmatory test is a single frozen campaign containing exactly two
      arms: A/Flash/baseline as reference and the committed valid winner as
      candidate; each arm runs the full official test exactly once.
- [x] Test campaign preparation refuses an uncommitted valid-selection artifact,
      a winner equal to the reference, or any mutable/mismatched preregistration.
- [x] Test execution is dry-run by default and requires explicit pair
      confirmation, approved budget, a committed campaign, and both committed
      per-arm preregistrations.
- [x] Confirmatory analysis validates complete paired ID/gold/case commitments,
      uses macro-F1 as the preregistered primary metric, and does not infer
      significance from an external aggregate baseline.

## Regression evals

- [x] All task prompt/parser standalone selftests still pass.
- [x] Existing authorized-data checks still pass.
- [x] Harness credential isolation and public-path safety checks still pass.
- [x] Result audit and scoreboard generation still pass.
- [x] Matrix launcher and paired analyzer selftests pass without dataset or API
      access.

## Model experiment acceptance criteria

Model experiments are a later, paid phase and are not part of the offline code
gate.

- Development split: full official `valid`, fixed seed, exact same retained IDs.
- Primary metric: macro-F1; secondary: weighted-F1, accuracy, invalid rate.
- Selection: one frozen profile/model configuration chosen from validation only.
- Test: one frozen two-arm campaign after preregistration is committed; the
  reference and candidate each run the full test once, with no changes after any
  test prediction is seen.
- Report: paired per-case comparison against the frozen baseline, bootstrap
  confidence intervals, confusion changes, usage, latency, and cost.
- Claim gate: no “significant win”, “tie”, or “equivalence” claim without the
  prespecified paired statistical test and complete paired predictions.

## Human review required

- Review taxonomy wording against the PsyGUARD paper/released annotation guide.
- Review any ambiguous validation errors before changing a definition.
- Human review does not create or alter the official benchmark gold labels.
