# Data source and contamination policy

## Allowed candidate sources

- Public text with a documented license permitting the intended research use.
- Human-authored synthetic scenarios that do not reproduce a real client's story.
- Researcher-created scenarios reviewed for realism, safety, and originality.

Every source needs a stable revision, license identifier, source-record hash, and
normalized-content hash before annotation.

## Prohibited sources

- Any benchmark valid/test example, official test, hidden label, or frozen
  holdout from CPsyExam, EmoBench, PsySUICIDE, IMHI, MentalManip, EATD, MDD-5k,
  or CBT-Bench.
- Row-level benchmark prediction/output files or error cases reconstructed from
  protected benchmark text.
- Private therapy notes, support chats, medical records, or client stories
  without explicit research authorization and ethics/privacy review.
- Content whose license or provenance cannot be verified.
- AI-generated labels represented as human gold.

Benchmark training splits are not automatically allowed. Each source license and
benchmark policy must explicitly permit the new use, and its rows must remain
separate from every evaluation split.

## Privacy gate

Before human review, remove direct identifiers and unnecessary quasi-identifiers.
Do not retain names, handles, phone numbers, email, addresses, exact dates,
employers, schools, or unique event combinations. The data steward records only
aggregate audit counts; sensitive before/after text is never committed to Git.

## Contamination and split gate

1. Normalize text deterministically and compute SHA-256.
2. Check exact overlap against all authorized benchmark train/valid/test hashes.
3. Cluster near-duplicates and shared scenario templates.
4. Assign one `split_group_id` per person, source thread, or synthetic template.
5. Freeze train/validation membership by group, never by individual row.
6. Run the overlap check again after editing and before a version manifest.

No benchmark test score may be used to decide which training row to keep.

## Publication boundary

The public repository may contain schemas, guidelines, empty templates, aggregate
counts, and hashes. Raw text, candidate rows, human annotations, adjudication
notes, reviewer identities, and final JSONL stay in the approved external data
workspace unless a separate release review explicitly authorizes publication.
