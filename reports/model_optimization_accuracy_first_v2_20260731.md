# PsySUICIDE accuracy-first v2 status — 2026-07-31

## Current verdict

Stage A is complete and unchanged. The Stage B protocol-and-code base is
prepared and has passed synthetic and repository regression checks, but no
Stage B model metric has been produced.

The current blocker is a data-boundary blocker, not a compute blocker: the
repository and licensed dataset area do not contain a verified
optimization-only artifact holding the previously committed 9,342 development
rows. The existing v1 loader can reconstruct those rows only by reopening the
licensed full train file, which also contains the already-consumed 2,329-row
frozen holdout. Stage B therefore refuses that loader and does not run smoke,
screen, or confirmation until an optimization-only input is separately
established and verified.

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

The draft protocol is
`reports/psysuicide-roberta-v2-accuracy-first.prereg.draft.json`. It remains
deliberately non-executable:

- status: `PARTITION_FREEZE_PENDING`;
- `training_allowed=false`;
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

The exact inner-train, inner-dev, and per-label inner-dev commitments, final
trainer/analyzer SHA-256 values, and execution base commit are intentionally
still pending. They must be frozen, committed, and pushed before any
metric-producing run. The later final-preregistration HEAD is verified against
the live remote and recorded in each private run identity; it cannot contain
its own commit SHA.

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

1. Establish a repository-external, permission-`0600`, optimization-only input
   without exposing or reusing frozen-holdout membership.
2. Verify its exact row count, per-label counts, and optimization commitment.
3. Generate aggregate-only inner-train, inner-dev, and 11 per-label inner-dev
   commitments.
4. Insert those commitments plus the final trainer/analyzer SHA-256 values and
   execution base commit into the final preregistration.
5. Commit the final preregistration, push it, and
   verify the live remote SHA.
6. Run integrity-only smoke, then the full Seed-42 four-arm screen.
7. Only an eligible challenger and D proceed to fresh Seeds 43/44/45.

Until gates 1–5 pass, the accurate status is:
`PROTOCOL_PREPARED / DATA_BOUNDARY_BLOCKED / NO_V2_METRICS`.

## Deferred work

Decoupled-head training, soft ensembles or model soup, hierarchy assistance,
clean Qwen3.5-4B training, task-specific benchmark families, and 静室 shadow
mode have not started. Each remains behind its own preregistration and the
evidence gates specified in the staged plan.
