# EVAL DEFINITION: PsySUICIDE model optimization protocol

## Objective

Improve the model-facing PsySUICIDE classifier without changing labels, deleting
hard cases, or tuning on the official test split.

## Capability evals

- [ ] `psysuicide` requires an explicit `--split` for scored runs.
- [ ] `train`, `valid`, and `test` load from distinct official files and produce
      globally unique normalized case IDs.
- [ ] Authorized-data check rejects exact retained-text overlap across official
      splits.
- [ ] `baseline`, `taxonomy`, and `hierarchical` prompt profiles pass the
      standalone prompt/parser selftest without loading licensed data.
- [ ] Every result row and summary records split, prompt profile, requested
      model, prompt hash, dataset hash, seed, and run ID.
- [ ] Deprecated DeepSeek routing aliases are rejected before any API call.
- [ ] A test run is rejected unless `--confirm-test` and a matching frozen
      preregistration artifact are both supplied.
- [ ] `--prepare-prereg` writes the exact test-run identity without making an
      API call or writing result rows.
- [ ] Resume rejects rows from a different split, prompt profile, model, seed,
      prompt hash, or dataset hash.
- [ ] A new run refuses to append to an existing result path unless `--resume`
      is explicit.
- [ ] The valid-matrix launcher is dry-run by default and requires both
      `--execute` and an approved budget for paid calls.
- [ ] Full valid launch is rejected until all four fixed 50-case smoke summaries
      have zero API errors, exact response-model identity, and matching protocol
      metadata.
- [ ] The paired analyzer validates identical complete ID/gold sets, reports all
      four arms, runs only the three prespecified contrasts, applies Holm
      correction, and selects by the frozen metric order.
- [ ] Result summaries aggregate token usage without exposing credentials or
      source text.

## Regression evals

- [ ] All task prompt/parser standalone selftests still pass.
- [ ] Existing authorized-data checks still pass.
- [ ] Harness credential isolation and public-path safety checks still pass.
- [ ] Result audit and scoreboard generation still pass.
- [ ] Matrix launcher and paired analyzer selftests pass without dataset or API
      access.

## Model experiment acceptance criteria

Model experiments are a later, paid phase and are not part of the offline code
gate.

- Development split: full official `valid`, fixed seed, exact same retained IDs.
- Primary metric: macro-F1; secondary: weighted-F1, accuracy, invalid rate.
- Selection: one frozen profile/model configuration chosen from validation only.
- Test: one full run after preregistration is committed; no prompt changes after
  seeing test predictions.
- Report: paired per-case comparison against the frozen baseline, bootstrap
  confidence intervals, confusion changes, usage, latency, and cost.
- Claim gate: no “significant win”, “tie”, or “equivalence” claim without the
  prespecified paired statistical test and complete paired predictions.

## Human review required

- Review taxonomy wording against the PsyGUARD paper/released annotation guide.
- Review any ambiguous validation errors before changing a definition.
- Human review does not create or alter the official benchmark gold labels.
