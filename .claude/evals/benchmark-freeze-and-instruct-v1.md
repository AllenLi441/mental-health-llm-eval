# EVAL DEFINITION: benchmark freeze and MentalHealth-Instruct v1

## Objective

Freeze the current 8-family, 19-task benchmark protocol and reviewed DeepSeek
aggregate evidence without pretending that legacy mixed-model runs form one
paper-ready control. In parallel, create a public-safe MentalHealth-Instruct v1
design package and an empty repository-external workspace for future real-human
annotation. No training, model download, API call, benchmark test access, or
assistant-authored gold label is authorized by this eval.

## Capability evals

### A. Benchmark suite freeze

- [x] A machine-readable suite freeze binds the 19 task keys, registry/schema
      hashes, registry bundle hash, source Git commit, and public/private boundary.
- [x] The freeze distinguishes protocol identity from raw dataset redistribution.
- [x] The freeze can be checked offline and fails on a changed registry/task hash.

### B. DeepSeek historical baseline freeze

- [x] One aggregate-only manifest covers CPsyExam, EmoBench, PsySUICIDE, IMHI,
      MDD-5k, MentalManip, EATD, and CBT-Bench.
- [x] Every task row names a reviewed public artifact and its SHA-256.
- [x] Exact V4 model/fingerprint evidence is separated from unresolved legacy
      `deepseek-chat` aliases.
- [x] Proxy, custom, incompatible, historical-parser, and split limitations are
      explicit. The manifest must set `paper_control_ready=false`.
- [x] No current score is regenerated, replaced, or selected through test access.

### C. MentalHealth-Instruct v1 design package

- [x] The final-record schema covers benchmark-aligned task families while
      keeping condition labels explicitly non-diagnostic.
- [x] Benchmark valid/test, frozen holdout, row-level benchmark artifacts, and
      unlicensed/private client text are forbidden data sources.
- [x] Every final target must originate from real-human consensus/adjudication;
      assistant/model labels cannot be exported as human gold.
- [x] A blank pilot card starts `UNLABELED`, contains no example text or target,
      and cannot be exported to training.
- [x] Blind author/reviewer separation, expert escalation, PII review, license
      review, source hashing, group split, and overlap checks are documented.

### D. Private workspace initializer

- [x] Default mode is dry-run and performs no writes.
- [x] Execute mode refuses any output inside the public repository.
- [x] Execute mode creates `raw_data/`, empty `train.jsonl` and
      `validation.jsonl`, copied frozen specs, review folders, and a manifest.
- [x] Existing non-identical files are never overwritten.
- [x] Selftest uses only temporary synthetic paths and passes under normal Python
      and `python -O`.

## Regression evals

- [x] Existing benchmark registry, ledger, scoreboard, baseline, harness-safety,
      trainer, and IMHI offline selftests still pass.
- [x] Current IMHI DR/CE execution failure is recorded as post-training
      best-checkpoint loading compatibility, not as a model score or data failure.
- [x] PsySUICIDE v2 B/C/D remain `PAUSED`; no new process or restart is started.
- [x] Git diff contains no credentials, absolute licensed-data paths, raw rows,
      human gold, model weights, checkpoints, or row-level outputs.

## Human review required

- Label ontology approval by qualified mental-health reviewers.
- One manually authored pilot card, followed by two independent blind reviews.
- Expert adjudication for disagreements and every high/imminent-risk item.
- Source-license approval before any row enters the private candidate pool.
- A separate preregistration before LoRA/QLoRA training or benchmark test access.

## Result status

`PASS_PUBLIC_ASSETS / HUMAN_REVIEW_PENDING / TRAINING_NOT_ALLOWED`
