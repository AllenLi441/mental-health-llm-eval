# Annotation guideline v1

This protocol prepares real people to annotate. It does not authorize an AI
assistant to write labels or to stand in for a human reviewer.

## Start with exactly one pilot card

Open `pilot/blank-card.json`. Complete one field at a time in this order:

1. `source`: record the license, revision, and source hash. Stop if the source
   cannot legally be used or might contain private client material.
2. `task_family`: choose the single benchmark-aligned research task.
3. `language`: choose `zh`, `en`, or `bilingual`.
4. `instruction`: write the task request without copying benchmark prompts.
5. `input`: add one licensed or genuinely human-created, de-identified scenario.
6. `author_private`: the human author records an intended output and rationale.
   This field is never shown in the blind reviewer packet.
7. `reviewer_a` and `reviewer_b`: two real reviewers label independently. They
   cannot see the author target or each other's answer.
8. `adjudication`: only after both reviews are locked, an expert resolves a
   disagreement or approves a mandatory high-risk escalation.
9. Convert to a final `label_schema.json` record only after all gates pass.

Do not begin a 1,000-row batch until this one pilot completes the full workflow.

## Role separation

- Data steward: checks source rights, PII removal, hashes, duplicates, and split.
- Human author: creates or adapts the candidate and records a private rationale.
- Reviewer A/B: independently judge task family, target, safety, and quality.
- Expert adjudicator: resolves disagreement and reviews every high/imminent-risk
  item and every item that could be interpreted as a clinical diagnosis.

Use pseudonymous reviewer IDs. Do not store student numbers, personal email, or
client identifiers in dataset rows.

## Label rules

- `condition_signal` is a dataset classification target, not a diagnosis of a
  person and not clinical advice.
- Use `uncertain` when evidence is insufficient; do not force certainty to make
  class counts look balanced.
- Multi-label CBT items must preserve all independently justified labels.
- Support responses must be non-judgmental, must not promise outcomes, and must
  not replace professional or emergency care.
- High/imminent-risk items require expert escalation and approval before export.

## Blind review packet

The reviewer packet contains only card ID, task family, language, instruction,
input, source category, and the reviewer fields. It excludes author target,
author rationale, other reviewer answers, benchmark labels, model predictions,
and any proposed gold.

## Acceptance gate

A row is `APPROVED` only when:

- source/license/PII/overlap checks pass;
- two independent reviews are complete;
- full agreement exists or expert adjudication is recorded;
- any required expert escalation is approved;
- group split is frozen without cross-split subject/source leakage;
- `model_generated_gold=false` and gold origin is human consensus/adjudication.

Rejected or unresolved cards remain outside `train.jsonl` and `validation.jsonl`.
