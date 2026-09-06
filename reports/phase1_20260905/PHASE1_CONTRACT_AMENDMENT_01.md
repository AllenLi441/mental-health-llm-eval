# Phase 1 contract amendment 01 (2026-09-05)

This amendment supersedes contradictory wording in the 2026-09-05 contract. It
does not change any frozen export, human-review requirement, or evaluation
rubric. The machine-readable companion is `contract_v1_1.json`.

## Four isolated data routes

The project keeps four routes: `research-v0` (500/50 static smoke export),
`a3-b1-candidates` (the newly rebuilt, not-frozen candidate pool), `m1-v1`
(the historical 8,000/500/500 research candidate), and `formal-v2` (the
historical human-written product route, currently 0 approved records). The
`formal-v2` count is not a project-wide count and is not the only possible
future product route. A candidate pool is never a training export merely
because its source license appears permissive.

The 82,567 and 39,250 figures have different meanings. 82,567 is the raw-row
sum reported by the A3/B1 sources; 39,250 is the arithmetic 70:30 capacity
estimate. The latter has not been materialized, split, deduplicated, or frozen.
`artifacts/phase1-a3-candidate-20260905/manifest.json` records the actual
conversion counts and input hashes.

## Action permissions and state transitions

`content_state`, `permissions`, `evidence`, and `gates` are separate. The
previous single `training_allowed` flag cannot express the difference between
local preparation, a disposable remote smoke, research SFT, product SFT, and
product release. The permitted order is:

```
qualified records -> READY_FOR_EXPORT -> EXPORT_BUILT
 -> EXPORT_STATIC_VALIDATED -> READY_FOR_REMOTE_SMOKE
 -> REMOTE_SMOKE_PASSED -> the separately authorized training action
```

Export validation does not require an export that has not yet been built.
`research-v0` may be used for the already scoped pipeline smoke after its
bundle checks and remote-machine preflight; that permission does not make it
gold data or product data. A new research SFT run requires its own source-use,
split, contamination, runtime, and model records. It does not wait for the
unrelated formal-v2 human gold route. Product training and release remain
blocked until their own gates pass.

## Candidate conversion actually performed

`scripts/data/prepare_a3_candidates.py` read only four local raw files and the
PsyDT heldout file. It emitted one model-neutral target per raw record (a
deterministic turn per PsyDT dialogue, source-final target for the other three),
message-level assistant loss intent, source file SHA256, group provenance, and
review flags. It rejected malformed records, quarantined exact duplicates and
PsyDT heldout matches, and never assigned a final split or approval. The
candidate manifest reports **82,567 raw records and 82,401 normalized
candidates**; this is a preparation artifact, not a release count.

SMILE records lack upstream report/question IDs, so their first-user digest is
only a heuristic and is explicitly not verified lineage. Keyword flags are
review queues, not safety decisions. Token-level assistant masking still needs
to be proven by the selected trainer.

## Evaluation and replacement claims

The frozen five dimensions remain relevance, empathy, safety, boundary, and
naturalness, scored 1–5. The existing amendment's six-test Holm family is
preserved; it must not be silently changed. Before confirmatory use, record the
rubric file hash, judge identity/type, blind order, invalid-output handling,
disagreement adjudication, effect definition, and cluster unit. The five
dimensions may be descriptive unless a test family is pre-registered.

Relative safety uses a pre-approved non-inferiority margin `delta_safety` and
absolute critical-event limit `tau_critical`; both remain `TBD / NOT_APPROVED`.
No observed critical event is a zero-risk claim. Human Gold, Safety Eval and
the current DeepSeek product baseline are required for a replacement decision,
but they do not block local data preparation or an explicitly scoped research
smoke.

## CPsyCoun scope status

The local archived upstream repository identifies `CC-BY-4.0` while the exact
Hugging Face dataset page currently identifies `cc-by-sa-4.0`; these are not
proven to be the same revision or scope. GitHub also states that CPsyCounR is
available on request after a privacy agreement. Therefore CPsyCoun is
`UNRESOLVED_REVISION_OR_SCOPE` for this project until a fixed source commit,
data-file hash, card hash, and applicable license text are reconciled. This
amendment makes no legal conclusion and does not silently admit or ban all
public corpora.
