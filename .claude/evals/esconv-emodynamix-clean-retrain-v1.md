# EVAL DEFINITION: EmoDynamiX clean canonical ESConv retraining v1

Defined: 2026-08-20 (America/Los_Angeles)

## User journey

As the Jingshi model developer, I want to retrain an EmoDynamiX-derived
eight-class strategy policy from clean canonical ESConv train data, so that the
result can be selected on canonical dev and later compared once on the frozen
test without inheriting the released checkpoint's split contamination.

## Scope

- Parent protocol: `jingshi-esconv-first-v1`.
- Architecture source: official `cw-wan/EmoDynamiX-v2` commit
  `c9213d718a9684a5e05ce5daa947f9cbbfb7b927`.
- Allowed learning source: canonical author train only, after the mandatory
  train/dev exact-input filter.
- Allowed selection source: canonical author dev only.
- Frozen test: inaccessible to every prepare, feature, audit, smoke, and train
  command in this development pipeline.
- The released EmoDynamiX checkpoint may verify implementation compatibility;
  it must never initialize a clean training run.

## Capability evals

### C1. Canonical causal record builder

- [x] Only the exact canonical train/dev filenames, row counts, and SHA-256
      commitments are accepted; no test or generic split argument exists.
- [x] Conversation reconstruction is deterministic and uses the upstream
      EmoDynamiX five-prior-turn window, adding `<START>` only where causally
      appropriate.
- [x] Each model input contains prior text, prior speaker roles, and prior
      supporter strategies, but never the current target response or current
      gold strategy.
- [x] The mandatory dev-input overlap filter removes exactly 129 train rows and
      yields the frozen 8,433-row derived-train commitment.
- [x] Every record binds its source-line hash, causal model-input hash,
      conversation id, target turn, and label outside the model-input object.

### C2. Precomputed emotion/discourse feature contract

- [x] Feature rows are joined only by the exact causal model-input SHA-256; row
      order or labels cannot select a feature row.
- [x] Each feature row contains one seven-way upstream ERC softmax-output vector
      per input node and discourse edges whose endpoints and relation ids are
      in range. The author-compatible control preserves the upstream model's
      second softmax; true-logit handling is a separate ablation.
- [x] Feature generation and loading exclude the current target response and
      strategy, bind the SDDP/ERC implementation and weight hashes, and reject
      missing, duplicate-key, extra, or mutated feature rows. Record-level
      duplicate inputs remain valid and join many-to-one to a feature key.
- [x] A synthetic/structural fixture backend is permitted only for tests and
      explicitly labelled smoke runs; it can never produce a selectable model.

### C3. EmoDynamiX-derived policy model

- [x] The model contains a locally materialized RoBERTa context encoder, mixed
      ERC prototypes, prior-strategy embeddings, relational graph layers, and
      an eight-class classification head.
- [x] The model accepts precomputed history-only features and has no parameter
      or forward input for the current gold label/response.
- [x] Base model loading is local-tree-only and SHA-256 bound; remote ids,
      symlinks, and the released EmoDynamiX task checkpoint are rejected.
- [ ] CE and train-only class-balanced/logit-adjusted objectives use complete
      accumulation-window normalization and are covered by analytic tests.

### C4. Train/dev-only execution

- [ ] Audit is the default; model loading, optimizer steps, and artifact writes
      require explicit `--execute`.
- [ ] Training never instantiates or loads a test dataset and never invokes an
      automatic test method after selecting a dev checkpoint.
- [ ] Dev Macro-F1 is primary selection, dev Accuracy secondary, and lower dev
      loss tertiary; every epoch reports all three plus Weighted-F1.
- [ ] Checkpoint and manifest bind data, features, base tree, implementation,
      optimizer, loss, dependencies, git state, selected dev metrics, and all
      artifact SHA-256 values.
- [ ] Any fixture-feature run is marked `DEVELOPMENTAL_SMOKE_NOT_SELECTABLE`;
      only verified SDDP/ERC features can enter a full dev pilot.

## Regression evals

- [ ] Existing EmoDynamiX adapter and planner tests pass.
- [ ] Existing canonical policy trainer tests pass.
- [ ] The author-native reproduction remains unchanged and released-checkpoint
      frozen status remains `DIAGNOSTIC_ONLY_TRAIN_CONTAMINATED`.
- [ ] All new/changed Python files compile, JSON artifacts parse, and
      `git diff --check` passes.
- [ ] New production code reaches at least 80% line coverage.
- [ ] Critical regression targets pass three consecutive times before any
      full-data optimization run.

## Staged acceptance

1. `DATA_READY`: C1 passes on all canonical train/dev rows; no model loaded.
2. `FEATURE_CONTRACT_READY`: C2 passes on fixtures and a real feature smoke.
3. `MODEL_SMOKE_READY`: one tiny real forward/backward/checkpoint cycle passes
   on the local machine without reading test.
4. `DEV_PILOT_READY`: verified real feature coverage is complete for derived
   train and canonical dev; configuration and code are committed and clean.
5. `FORMAL_CANDIDATE`: out of scope for this eval; requires three seeds and a
   separately frozen unique-candidate protocol before the one test run.

## Product gate

- [ ] The result is described as a strategy policy component, not a complete
      response-generation or DeepSeek API replacement.
- [ ] Dataset/checkpoint commercial rights, crisis safety, clinical boundaries,
      privacy, expert review, shadow-mode monitoring, and rollback remain
      mandatory before product replacement.

## Automatic failure conditions

- Any canonical frozen-test path or data read.
- Any current target response/strategy in a model input or feature key.
- Any gradient update using dev.
- Any initialization from the released task checkpoint.
- Any claim that fixture features reproduce or improve EmoDynamiX.
- Any paper/frozen leaderboard claim from train/dev-only results.
