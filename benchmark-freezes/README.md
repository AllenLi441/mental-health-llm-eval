# Benchmark freeze v1

This folder freezes the protocol identity and the currently reviewed aggregate
DeepSeek evidence before new data construction or model training begins.

## What is frozen

- `mental-health-benchmark-suite-v1.json`: 8 families, 19 task contracts, exact
  registry/schema/task hashes, and the Git commit that supplied them.
- `deepseek-historical-baseline-v1.json`: the public aggregate artifacts and
  metric values selected as the historical reference snapshot.

The baseline file is intentionally marked `paper_control_ready=false`. It is a
reproducible record of what exists, not a claim that one exact DeepSeek model was
run under one uniform paper-compatible protocol.

## Human-readable current snapshot

| Family | Frozen reference snapshot | Identity/protocol boundary |
|---|---|---|
| CPsyExam | V4-Flash accuracy `83.3419%`, n=3,902 | Exact fingerprint; immutable historical release has a prospective parser limitation |
| EmoBench | V4-Pro EA `74.25%`, EU `71.50%`, n=400 each | Exact fingerprint; temp-0 single proxy, not paper 5x4 |
| PsySUICIDE | V4-Flash baseline accuracy `82.8552%`, macro-F1 `0.5494`, n=1,464 | Exact frozen official-test reference; unavailable for new selection |
| IMHI x9 | weighted-F1: DR `.8182`, dreaddit `.8110`, loneliness `.7324`, Irf `.6387`, multiWD `.7115`, SAD `.5515`, CAMS `.4308`, SWMH `.6469`, T-SID `.7440` | Exact response fingerprint but historical per-task sampled-test selection |
| MDD-5k | macro-F1 `.4051`, accuracy `57.25%`, n=400 | Project-defined proxy; unresolved legacy model alias |
| MentalManip | positive F1 `.8510`, accuracy `77.80%`, n=500 | Balanced accuracy absent; unresolved legacy model alias |
| EATD | depressed-class F1 `.48`, accuracy `83.54%`, n=79 | Sensitivity absent; validation consumed as proxy; unresolved legacy alias |
| CBT-Bench | top-1 hit: CD `.5822`, PC `.8913`, FC `.6964` | Incompatible with paper multilabel F1; unresolved legacy alias |

The stronger CPsyExam V4-Pro and PsySUICIDE V4-Pro+taxonomy results remain
frozen candidate evidence, not the control definition above. The 2026-07-31
Flash+taxonomy PsySUICIDE valid run was rejected by its preregistered gates and
does not replace this baseline.

## Comparison rule

A future MentalHealth-LLM result may be subtracted from a frozen reference only
when it uses the same examples, split role, metric, scorer, and frozen protocol.
Otherwise the two point estimates remain descriptive and separate.
