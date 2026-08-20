# EVAL DEFINITION: ESConv code-first selection and improvement v1

Defined: 2026-08-20 (America/Los_Angeles)

## Objective

Select an ESConv engineering base from artifacts that actually contain runnable
implementation code, then test one low-cost improvement without reading or
optimizing against the frozen ESConv test split.

## Capability evals

### C1. Code-first candidate gate

- [ ] Every implementation candidate has a paper URL and an author/official code URL.
- [ ] A repository counts as implemented only when the relevant model source and a
      concrete train or inference entry point are present; PDF/README/forthcoming-only
      repositories are excluded.
- [ ] Checkpoint, dataset/preprocessing, exact command, dependency and license status
      are reported separately. A pretrained base-model download is not called a paper
      checkpoint.
- [ ] Paper-only Accuracy is never mixed with locally reproduced results.

### C2. Local artifact audit

- [ ] `/Users/allenli/Desktop/EmoDynamiX-v2-master` and
      `/Users/allenli/Desktop/MultiESC-main` are inspected byte-for-byte enough to
      establish whether source, data, paper checkpoints and metric assets are present.
- [ ] The EmoDynamiX desktop archive is compared to the pinned official checkout.
- [ ] MultiESC's modified strategy taxonomy and five-stage pipeline are disclosed.

### C3. Reproduced base

- [ ] EmoDynamiX author-native checkpoint reproduction remains exactly 2,895 rows,
      ACC 33.6097%, Macro-F1 27.7040%, Weighted-F1 32.7087%, invalid 0.
- [ ] The released checkpoint is not declared eligible for the frozen leaderboard
      because its author training split overlaps the frozen test conversations.

### C4. MultiESC-inspired planning experiment

- [ ] Transition/prior statistics are fitted from the author train split only.
- [ ] Hyperparameters are selected on the author valid split only, with Macro-F1 as
      primary metric and Accuracy as secondary metric; remaining ties prefer the
      smaller transition weight, lower history order and lower smoothing value.
- [ ] No author test or frozen test data are read by the development command.
- [ ] The planner consumes only prior strategy history and base logits; gold target
      labels are used by the scorer only.
- [ ] Baseline and candidate valid metrics, per-class metrics, configuration, source
      hashes and output hashes are recorded in a machine-readable receipt.
- [ ] The result is labelled developmental and cannot be used as a paper reproduction
      or frozen-test result.

## Regression evals

- [ ] Existing EmoDynamiX adapter tests pass.
- [ ] Existing ESConv metric tests pass.
- [ ] New tests prove split rejection, target isolation, deterministic transition
      probabilities, deterministic selection and hash-rich receipts.
- [ ] No long training run or frozen-test prediction is started by the test suite.

## Product gate

- [ ] A strategy predictor/planner is described as a policy component, not a complete
      replacement for a response-generation API.
- [ ] ESConv/checkpoint/data commercial-use clearance and clinical safety evaluation
      remain explicit blockers for production replacement.

## Success criteria

- Capability evals: all deterministic items pass at pass@1.
- Regression evals: pass^3 is required before a formal model-selection run.
- Any frozen-test access, paper/checkpoint conflation, or target-response leakage is an
  automatic failure.
