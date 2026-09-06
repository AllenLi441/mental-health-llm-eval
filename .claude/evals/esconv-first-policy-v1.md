# EVAL DEFINITION: ESConv-first auditable policy development

## Objective

Build and compare an eight-class ESConv next-strategy policy using only the
immutable author train split for learning and the immutable dev split for
selection.  No new model may read or predict the frozen 2,775-row test before a
single candidate and formal-test campaign are committed.

## Capability evals

- [ ] The trainer accepts only the exact author train/dev filenames, row counts,
      and SHA-256 commitments; it exposes no test-file argument.
- [ ] The final supporter response and its strategy annotation are structurally
      excluded from every policy input.
- [ ] The 12 frozen train/dev model-input overlaps are fail-closed and all 129
      affected train rows are mandatorily excluded, yielding the exact 8,433-row
      derived-train commitment frozen in the preregistration.
- [ ] The base model must be a materialized local regular-file tree whose full
      commitment matches an explicit SHA-256; symlinks and remote model IDs are
      rejected.
- [ ] Training is dry-run by default and requires explicit `--execute` before
      model loading or checkpoint writes.
- [ ] CE, effective-number class-balanced loss, and logit-adjusted loss are
      explicit, tested alternatives.
- [ ] Dev Macro-F1 is the primary checkpoint-selection metric; Accuracy is only
      the first tie-breaker.
- [ ] Every saved checkpoint is bound to a manifest containing data, model-tree,
      configuration, dependency, git, metric, and checkpoint byte commitments.
- [ ] Preregistered runs bind committed preregistration and base-materialization
      receipt bytes, and reject relevant dirty files or code changes during a run.
- [ ] An external checkpoint with unknown training provenance or unresolved
      overlap is always diagnostic-only.
- [ ] Frozen leaderboard eligibility additionally requires a zero-overlap audit
      bound to the frozen test hash and an already committed, unique-candidate
      formal-test authorization.
- [ ] XLM-R profiles remain disabled until an immutable input-template source
      and label-map source are recorded independently.

## Regression evals

- [ ] Existing ESConv reproduction, EmoDynamiX adapter, and generic HF
      classifier tests still pass.
- [ ] EmoDynamiX's released checkpoint remains
      `DIAGNOSTIC_ONLY_TRAIN_CONTAMINATED` on frozen-2775.
- [ ] CausalLM heads and unknown label mappings remain rejected.
- [ ] All changed Python files compile; all JSON artifacts parse; `git diff
      --check` passes.

## Development experiment acceptance criteria

- Data: author train only for optimization; author dev only for selection and
  calibration; no dev examples in retrieval or gradient updates.
- First implementation pilot: fixed local RoBERTa tree, seed 42, max length 256
  with left truncation, three epochs, learning rate 2e-5, weight decay 0.01,
  warmup ratio 0.1, batch 8 with gradient accumulation 2, eval batch 16.
- Compare loss arms on identical train/dev records and record complete manifests.
- Selection claims require all preregistered seeds (42, 43, 44); a single-seed
  run is explicitly a development pilot and cannot freeze the formal candidate.
- Formal frozen test: exactly one run after the unique policy candidate,
  contamination audit, and authorization manifest are committed.

## Human review required

- Review checkpoint, base-model, ESConv, AugESC, and auxiliary-data license and
  provenance before any product use.
- Review crisis, self-harm, clinical-boundary, privacy, and rollback behavior
  before any shadow deployment or replacement of the DeepSeek API.
- Benchmark improvement alone never establishes clinical effectiveness.
