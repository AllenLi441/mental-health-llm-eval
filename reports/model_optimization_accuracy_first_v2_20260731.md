# PsySUICIDE accuracy-first v2 status — 2026-07-31

## Current verdict

Stage A is complete and unchanged. The Stage B optimization-only export,
7,479/1,863 split freeze, and final preregistration are complete and pushed.
No selection-eligible Stage B metric has been produced.

The current blocker is checkpoint integrity, not data access or compute. A
first integrity-only Arm-A smoke exposed a Transformers 5.0 compatibility bug:
the checkpoint saved 49 LayerNorm pairs as legacy `gamma`/`beta`, while Trainer
best/resume reload expects native `weight`/`bias`. The two-step smoke happened
to select its final checkpoint, so the LayerNorm values were retained in
memory rather than randomized; a full run could nevertheless mix different
epochs or resume with base LayerNorm values. Screen remains blocked until the
save/load path is fixed, re-frozen, pushed, and all four smoke arms pass under
a new run ID.

## Stage A re-audit

- Stage A completion commit:
  `1a1c194dbb7168cadd839b17ef37c1ec0b16ca04`.
- Draft PR:
  [#3](https://github.com/AllenLi441/mental-health-llm-eval/pull/3),
  targeting `main` and declaring its dependency on #2.
- Three declared training Seeds 42, 43, and 44 have aggregate artifacts.
- The one-shot internal holdout report remains:
  accuracy `0.9343065693`, macro-F1 `0.7275306390`, and weighted-F1
  `0.9348403838`.
- The v1 preregistration, partition commitment, valid selection, holdout freeze,
  one-shot result, trainer, reporter, and scorer have no diff from the Stage A
  completion commit.
- No Stage A training, reporting, scoring, or `caffeinate` process remains.

These checks do not turn the internal holdout into an official paper test and
do not establish superiority, a statistical tie, or equivalence.

## Predeclared Stage B design

The frozen protocol is
`reports/psysuicide-roberta-v2-accuracy-first.prereg.json`:

- status: `FROZEN`;
- `training_allowed=true`, conditional on clean/pushed Git and every runtime gate;
- accepted source: repository-external permission-`0600` optimization-only
  data;
- required source: exactly 9,342 unique rows with the already published
  optimization commitment;
- forbidden sources: full train, official valid, official test, frozen holdout,
  and any dataset root that can reconstruct those inputs;
- deterministic 80/20 label-stratified split:
  7,479 inner-train and 1,863 inner-dev rows;
- primary selection metric: unrounded inner-dev accuracy;
- training: 10 epochs, evaluation and checkpointing each epoch, no early
  stopping, earlier epoch retained on an exact accuracy tie;
- four arms:
  natural+CE, weighted-sampling+CE, natural+weighted-focal, and
  weighted-sampling+weighted-focal;
- code smoke: integrity only;
- screen: all four arms on Seed 42;
- independent confirmation: eligible challenger versus D on fresh Seeds 43,
  44, and 45;
- private-only diagnostics: logits, gold label IDs, row digests, confusion
  matrix, NLL, multiclass Brier score, and fixed-bin ECE;
- screen selection and confirmation both recompute metrics from those private
  diagnostics; whole inner-dev and 11 per-label digest commitments bind each
  private row digest to its frozen gold-label stratum without publishing rows;
- public boundary: aggregate counts, metrics, commitments, configurations, and
  hashes only.

The trainer and analyzer now enforce exact source fields, path and permission
gates, deterministic split membership, complete checkpoint manifests, safe
resume identity, ignored result paths, clean/pushed Git identity, fixed arms
and seeds, per-epoch best-accuracy checkpointing, and private diagnostic shape,
metric, calibration, row-set, and gold-stratum commitments. Hand-edited screen
selection is rejected because confirmation recomputes it from all four frozen
Seed-42 aggregate and private-diagnostic artifacts.

The exact inner-train, inner-dev, and per-label inner-dev commitments were
frozen before smoke. The final-preregistration commit
`e24ab737fad3f0ec3f173e25034943a1c76192fb` was pushed and passed the trainer's
live-remote preflight. Because smoke then found a checkpoint-integrity defect,
the trainer SHA and execution base must be updated in a new frozen commit before
the replacement smoke; split, arms, metrics, seeds, and decision rules remain
unchanged.

## Verification completed for the protocol-and-code base

- Trainer and analyzer compile successfully.
- Both selftests pass under normal Python and `python -O`.
- Synthetic tests cover four-arm screening, frozen selection, six-run paired
  confirmation, the fixed 20,000-replicate bootstrap configuration with a
  reduced-replicate execution fixture,
  whole and per-label commitment checks, digest-to-gold remapping rejection,
  extra-field rejection, diagnostic shape checks, checkpoint corruption and
  incomplete-marker rejection, safe run IDs, and output overwrite refusal.
- Repository regression passes: 19/19 task prompt/parser selftests, 76-summary
  aggregate audit, 19-summary scoreboard, 22-entry baseline registry, harness
  safety, PsySUICIDE matrix/analyzer/trainer/reporter/scorer selftests,
  EmoBench protocol selftests, and the nine-task IMHI matrix selftest.
- The authorized optimization-only export reproduced the frozen source and all
  partition commitments; only 9,342 optimization rows were written privately.
- The split freeze produced 7,479 train and 1,863 inner-dev rows with zero
  digest overlap and exact whole/per-label commitments.
- The first Arm-A smoke is retained as failure-discovery evidence only. B/C/D
  stopped before training because their live-remote preflight transiently
  failed; none of attempt 1 may be used for selection.
- These are implementation and integrity checks only. They are not smoke,
  screen, confirmation, or accuracy evidence for v2.

## Compute estimate

The local machine has MPS support and enough current disk for a bounded run.
Using the Stage A observed runtimes, one 10-epoch run is estimated at roughly
7–9 hours. Four-arm Seed-42 screening is therefore roughly 28–36 hours
sequentially. Independent confirmation of one challenger and D on three fresh
seeds adds roughly 42–54 hours. `save_total_limit=2` is required because one
checkpoint is about 1.3 GB.

This estimate is not a completed run and is not evidence of accuracy
improvement. Slow execution alone is not a compute failure.

## Required next gate

1. Save checkpoints with native parameter names and reject every legacy
   LayerNorm or key/shape mismatch before writing a completion marker.
2. Enforce strict best-checkpoint and resume model-state loading.
3. Commit and push the integrity fix as the new execution base.
4. Update only the final preregistration trainer SHA and execution-base commit,
   then commit, push, and re-run the live-remote preflight.
5. Run all four integrity-only smoke arms under a fresh run ID.
6. Only after 4/4 smoke passes, start the full Seed-42 four-arm screen.
7. Only an eligible challenger and D proceed to fresh Seeds 43/44/45.

The accurate status is:
`DATA_BOUNDARY_CLOSED / CHECKPOINT_INTEGRITY_FIX_PENDING / NO_SELECTION_METRICS`.

## Deferred work

Decoupled-head training, soft ensembles or model soup, hierarchy assistance,
clean Qwen3.5-4B training, task-specific benchmark families, and 静室 shadow
mode have not started. Each remains behind its own preregistration and the
evidence gates specified in the staged plan.
