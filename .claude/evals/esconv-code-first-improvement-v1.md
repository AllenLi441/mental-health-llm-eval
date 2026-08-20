# EVAL DEFINITION: ESConv code-first selection and improvement v1

Defined: 2026-08-20 (America/Los_Angeles)

## Objective

Select an ESConv engineering base from artifacts that actually contain runnable
implementation code, then test one low-cost improvement without reading or
optimizing against the frozen ESConv test split.

## Capability evals

### C1. Code-first candidate gate

- [x] Every implementation candidate has a paper URL and an author/official code URL.
- [x] A repository counts as implemented only when the relevant model source and a
      concrete train or inference entry point are present; PDF/README/forthcoming-only
      repositories are excluded.
- [x] Checkpoint, dataset/preprocessing, exact command, dependency and license status
      are reported separately. A pretrained base-model download is not called a paper
      checkpoint.
- [x] Paper-only Accuracy is never mixed with locally reproduced results.

### C2. Local artifact audit

- [x] `/Users/allenli/Desktop/EmoDynamiX-v2-master` and
      `/Users/allenli/Desktop/MultiESC-main` are inspected byte-for-byte enough to
      establish whether source, data, paper checkpoints and metric assets are present.
- [x] The EmoDynamiX desktop archive is compared to the pinned official checkout.
- [x] MultiESC's modified strategy taxonomy and five-stage pipeline are disclosed.

### C3. Reproduced base

- [x] EmoDynamiX author-native checkpoint reproduction remains exactly 2,895 rows,
      ACC 33.6097%, Macro-F1 27.7040%, Weighted-F1 32.7087%, invalid 0.
- [x] The released checkpoint is not declared eligible for the frozen leaderboard
      because its author training split overlaps the frozen test conversations.

### C4. MultiESC-inspired planning experiment

- [x] Transition/prior statistics are fitted from the author train split only.
- [x] Hyperparameters are selected on the author valid split only, with Macro-F1 as
      primary metric and Accuracy as secondary metric; remaining ties prefer the
      smaller transition weight, lower history order and lower smoothing value.
- [x] No author test or frozen test data are read by the development command.
- [x] The planner consumes only prior strategy history and base logits; gold target
      labels are used by the scorer only.
- [x] Baseline and candidate valid metrics, per-class metrics, configuration, source
      hashes and output hashes are recorded in a machine-readable receipt.
- [x] The result is labelled developmental and cannot be used as a paper reproduction
      or frozen-test result.

### C5. Train-prior imbalance correction arm

- [x] The only added score is the frozen formula
      `base_logit + transition_weight*log(transition_prior) - tau*log(train_class_prior)`.
- [x] Class priors are computed from author train labels only; `tau` is selected on
      author valid from `0,0.05,0.1,0.2,0.3,0.4,0.6,0.8,1.0`.
- [x] Candidate selection remains Macro-F1, then Accuracy; remaining ties prefer
      smaller transition weight, smaller tau, lower history order and lower smoothing.
- [x] A zero-transition, zero-tau configuration is retained as the exact base-logit
      control. A non-zero arm is adopted only if valid Macro-F1 strictly improves.

## Regression evals

- [x] Existing EmoDynamiX adapter tests pass.
- [x] Existing ESConv metric tests pass.
- [x] New tests prove split rejection, target isolation, deterministic transition
      probabilities, deterministic selection and hash-rich receipts.
- [x] No long training run or frozen-test prediction is started by the test suite.

## Product gate

- [x] A strategy predictor/planner is described as a policy component, not a complete
      replacement for a response-generation API.
- [x] ESConv/checkpoint/data commercial-use clearance and clinical safety evaluation
      remain explicit blockers for production replacement.

## Success criteria

- Capability evals: all deterministic items pass at pass@1.
- Regression evals: pass^3 is required before a formal model-selection run.
- Any frozen-test access, paper/checkpoint conflation, or target-response leakage is an
  automatic failure.

## Execution evidence

- Code-first human report: `reports/esconv_code_first_model_audit_20260820.md`.
- Machine-readable candidate audit: `reports/esconv_code_first_model_audit_20260820.json`.
- Development receipt with per-class metrics and byte commitments:
  `reports/esconv_emodynamix_code_first_dev_20260820.json`.
- Scoped regression: 25/25 tests passed three consecutive runs on 2026-08-20.
- Built-in trace line coverage: `scripts/rerank_esconv_emodynamix.py` 93%;
  `open_response_eval/esconv_metrics.py` 98%.
- Transition-only result: rejected because the valid Macro-F1 optimum was the exact
  zero-weight base control.
- Primary valid selection: transition weight 0.1, class-adjustment tau 0.4;
  Macro-F1 26.5243% -> 27.6500%, Accuracy 33.6517% -> 31.9618%.
- Split receipt explicitly records `author_test_read=false` and
  `frozen_test_read=false`; no formal test campaign was consumed.
