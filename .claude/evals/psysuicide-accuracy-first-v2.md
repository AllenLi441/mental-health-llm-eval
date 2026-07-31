# EVAL DEFINITION: PsySUICIDE accuracy-first supervised v2

## Objective

Determine whether the supervised PsySUICIDE classifier can improve accuracy by
changing the training objective and imbalance treatment, without altering the
11 labels, using official valid or the frozen internal holdout for tuning, or
claiming that one task-specific model improves any other benchmark.

## Frozen development boundary

- Source rows: only the previously committed 9,342-row optimization partition.
- Inner split: deterministic, label-stratified train/inner-dev split derived
  only from optimization rows and committed before any v2 metric is observed.
- Forbidden inputs: official valid, official test, and the 2,329-row frozen
  internal holdout.
- Primary model selection metric: inner-dev accuracy.
- Reporting guards: macro-F1, pooled critical-risk recall, and prespecified
  critical-risk per-class recall.
- All raw text, source IDs, row-level gold/prediction pairs, logits, confusion
  matrices, calibration rows, checkpoints, and trainer state remain private and
  ignored.
- The whole inner-dev digest commitment plus one digest commitment per gold
  label bind private diagnostics to the frozen row set and label strata without
  publishing row membership.

## Capability evals

### Authorized one-time optimization export

- [ ] Export is dry-run by default and dry-run opens neither the licensed train
      file nor any valid/test file.
- [ ] Execution requires an explicit one-time authorization flag, the exact
      frozen source-file SHA-256, and the already committed v1 partition
      algorithm and seed.
- [ ] The exporter necessarily parses the authorized full train bytes once but
      never creates a separate holdout-row collection and never writes, prints,
      tokenizes, scores, or returns holdout rows; only their aggregate count and
      digest commitment may be recomputed for source-partition verification.
- [ ] The only row-level output contains exactly the 9,342 optimization rows
      with fields `idx`, `labels`, and `text`; its commitment must equal
      `0c3cbf98db02c609d500b4f9ef9bf95518b617bd4dffd1f4cde8a5c5e85a8dd9`.
- [ ] Output and one-time receipt are repository-external, permission `0600`,
      created without overwrite, and never displayed or added to Git.
- [ ] A fixed per-account machine-global claim path, independent of checkout,
      environment-home, and every CLI source/output value, is atomically created
      before the first source-content read and persists after any failed attempt;
      a second or concurrent attempt must fail closed.
- [ ] Private output basenames are fixed safe constants, and claim, optimization
      output, receipt, and public audit are read back with no symlink following,
      exact permissions, stable file identity, and byte/hash verification.
- [ ] The receipt and public audit contain only counts, commitments, file
      hashes, code/Git identity, UTC time, and basenames—never text, row IDs,
      membership, local absolute paths, or secrets.

- [ ] The trainer accepts only a repository-external, permission-`0600`,
      optimization-only file. It cannot accept a dataset root or reconstruct
      optimization by opening the licensed full train file.
- [ ] A split-freeze mode reads only that verified optimization-only file and
      emits counts and cryptographic commitments, never row content.
- [ ] The inner split is deterministic, label-stratified, mutation-sensitive,
      and has no row-digest overlap between train and inner-dev.
- [ ] The trainer never opens official `valid.json` or `test.json` and never
      returns, tokenizes, samples, or scores the frozen internal holdout.
- [ ] Execution is dry-run by default and requires an exact committed
      preregistration whose trainer SHA-256 matches the executing bytes.
- [ ] The four fixed arms are exactly:
      A natural sampling + CE,
      B weighted replacement sampling + CE,
      C natural sampling + class-weighted focal,
      D weighted replacement sampling + class-weighted focal.
- [ ] Screen and confirmation runs train for 10 epochs, evaluate every epoch,
      save by highest unrounded inner-dev accuracy, and retain the earlier
      epoch on an exact accuracy tie.
- [ ] Code smoke uses only deterministic optimization/inner-dev subsets and is
      integrity-only; smoke metrics cannot select an arm.
- [ ] The single-seed screen runs all four arms under Seed 42 and applies the
      preregistered accuracy-first plus non-inferiority gates after recomputing
      every aggregate metric from permission-`0600` private diagnostics.
- [ ] Independent multi-seed confirmation evaluates only the prespecified
      winner/control arms on fresh Seeds 43, 44, and 45; Seed 42 is not reused
      in the confirmatory mean.
- [ ] Aggregate public output contains metrics, configuration, commitments, and
      hashes only.
- [ ] Private ignored diagnostics contain confusion, logits, gold label IDs,
      row digests, NLL, multiclass Brier score, and fixed-bin ECE, but no text or
      source IDs.
- [ ] A run refuses overwrite, resume under a different identity, uncommitted
      preregistration, changed split commitment, undeclared arm/seed, or any
      attempt to pass a holdout/valid/test path.

## Prespecified four-arm decision

- [ ] Screen ranking is highest Seed-42 inner-dev accuracy, then macro-F1, then
      pooled critical-risk recall, then fixed arm order A/B/C/D.
- [ ] A non-D screen winner advances only if its accuracy is at least 0.002
      above D, macro-F1 is no more than 0.03 below D, pooled critical-risk
      recall is no more than 0.03 below D, and each prespecified critical class
      with inner-dev support at least 10 loses no more than 0.10 recall.
- [ ] Confirmation compares the screen winner with D on fresh Seeds 43/44/45.
      A non-D arm is called an accuracy improvement only if mean accuracy
      improves by at least 0.005, it wins at least two of three seeds, and a
      prespecified paired hierarchical bootstrap gives an accuracy-delta 95%
      lower bound above zero.
- [ ] The same confirmation requires macro-F1-delta and pooled
      critical-risk-recall-delta 95% lower bounds above -0.02, and each
      prespecified critical class with inner-dev support at least 10 has a mean
      recall point difference of at least -0.05.
- [ ] Sparse classes with inner-dev support below 10 are always reported but
      never converted into a significance, equivalence, or stable per-class
      improvement claim.
- [ ] If D wins the screen or no non-D arm passes the guards, the correct result
      is no evidence that removing either imbalance component improves
      accuracy; thresholds are not changed afterward.

## Later ablations

- [ ] Decoupled classifier-head training, three-seed probability ensemble/model
      soup, and flat-head plus coarse auxiliary hierarchy are each gated by a
      separate committed preregistration before execution.
- [ ] Each later ablation changes one declared component at a time and uses the
      same frozen train/inner-dev membership.
- [ ] Qwen3.5-4B, if attempted, starts from the general base model and clean
      optimization split; released PsyGUARD task weights are forbidden.

## Regression evals

- [ ] v1 preregistration, selection, freeze spec, scorer, and completed holdout
      result remain byte-for-byte unchanged.
- [ ] Existing 19/19 task prompt/parser selftests pass.
- [ ] Existing aggregate audit, scoreboard, baseline, safety, PsySUICIDE,
      EmoBench, and IMHI selftests pass.
- [ ] Secrets, licensed/raw data, weights, diagnostics, and local absolute paths
      remain outside Git.

## Protocol/code verification completed (not model evidence)

- [x] Trainer and analyzer compile and pass synthetic selftests under normal
      Python and `python -O`.
- [x] Adversarial source-path, permission, extra-field, mutation, diagnostic
      shape, digest-set, digest-to-gold mapping, checkpoint-completeness, run-ID,
      overwrite, and hand-edited-selection gates reject their fixtures.
- [x] Synthetic screen metrics are recomputed from all four private diagnostic
      archives; selection binds all four aggregate and diagnostic hashes.
- [x] Synthetic confirmation recomputes the screen winner and validates the
      paired six-run identity before the fixed hierarchical bootstrap.
- [x] Existing public repository regression selftests pass.
- [ ] No real split freeze, smoke, screen, confirmation, or accuracy claim is
      complete while the verified optimization-only input is absent.

## Human review required

- Review the critical-risk guard list and margins before the preregistration is
  committed.
- Review any future human-labeled rare-class expansion; AI output must never be
  presented as human gold.
- Review deployment or 静室 shadow-mode integration separately; this experiment
  does not authorize production use.
