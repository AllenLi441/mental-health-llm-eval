# Phase 1 local preparation evidence

This directory contains the reviewable, non-secret handoff for the 2026-09-05
preparation run. It is safe to commit because it contains no counseling text,
API key, model weight, or raw dataset.

* `PHASE1_CONTRACT_AMENDMENT_01.md` and `contract_v1_1.json` separate the four
  data routes, action permissions, evidence, and gates.
* `local_audit_v2/lineage_inventory.json` records the local hashes and five command results.
  The corresponding `.log` files preserve command output.
* `license_scope_matrix.json` records the unresolved CPsyCoun revision/scope
  conflict without making a legal decision.
* `../../scripts/data/candidate_training_guard.py` fails closed if a candidate
  artifact is passed to a training or release action.
* `../../scripts/data/prepare_a3_candidates.py` regenerates the candidate
  artifacts from the local raw inputs.

The generated candidate JSONL files are intentionally outside GitHub. They are
local `CANDIDATE_NOT_FROZEN` artifacts, 289 MiB total, and several sources have
unresolved license scope. Recreate them on the large machine from the recorded
input hashes after the source-use decision. The manifest, not a guessed row
count, is the handoff contract.
