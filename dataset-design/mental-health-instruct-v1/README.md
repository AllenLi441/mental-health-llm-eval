# MentalHealth-Instruct Dataset v1

Status: `DESIGN_FROZEN / ZERO_RECORDS / TRAINING_NOT_ALLOWED`

This directory defines the public-safe structure for a future human-reviewed
mental-health instruction dataset. It contains no training row, patient text,
benchmark example, human gold label, or model-generated label.

The design is benchmark-aligned but not benchmark-derived. Official benchmark
valid/test examples, frozen holdouts, row-level predictions, and hidden labels
are prohibited sources.

## Files

- `label_schema.json`: schema for a final human-reviewed training record.
- `annotation_guideline.md`: one-field-at-a-time workflow for real annotators.
- `data_source_policy.md`: license, privacy, contamination, and split rules.
- `expert_review.md`: blank expert sign-off and adjudication template.
- `pilot/blank-card.json`: the only initial pilot card; deliberately unlabeled.

The repository does not store `raw_data/`, `train.jsonl`, `validation.jsonl`,
annotation packets, or reviewer exports. Use the dry-run-first initializer to
create an empty workspace outside the Git checkout:

```bash
python3 scripts/init_mental_health_instruct_workspace.py \
  --output-dir <repository-external-directory>

python3 scripts/init_mental_health_instruct_workspace.py \
  --output-dir <repository-external-directory> --execute
```

## Task families

| Dataset task family | Benchmark motivation | Output boundary |
|---|---|---|
| `knowledge_reasoning` | CPsyExam | fixed answer/label; no benchmark row reuse |
| `emotion_understanding` | EmoBench | emotion/reasoning labels |
| `risk_classification` | PsySUICIDE | non-clinical dataset label plus risk escalation metadata |
| `condition_classification` | IMHI, MDD-5k, EATD | dataset condition signal, never a clinical diagnosis |
| `cognitive_distortion` | CBT-Bench | one or more distortion labels |
| `manipulation_detection` | MentalManip | manipulation label and evidence span policy |
| `supportive_response` | response-quality research | safe response text, not benchmark optimization |

## Training gate

Training remains prohibited until all of the following are true:

1. A qualified human panel approves the ontology and guideline.
2. One human-authored pilot card completes two independent blind reviews.
3. Disagreements and all high/imminent-risk items receive expert adjudication.
4. Every source has a recorded license, revision, and content hash.
5. PII review, benchmark-overlap screening, and group-level train/validation
   separation pass.
6. A version manifest freezes the approved rows and explicitly changes
   `training_allowed` from false under a separate review.

LoRA/QLoRA model choice, GPU rental, and benchmark test execution are later
decisions. This package does not authorize them.
