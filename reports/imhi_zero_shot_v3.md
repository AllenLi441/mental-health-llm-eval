# IMHI zero-shot protocol correction (v3)

- Date: 2026-07-11
- Model: `deepseek-chat`
- Temperature: `0`
- Metric: weighted F1 (primary), accuracy (secondary)
Source benchmark: MentaLLaMA / IMHI, WWW 2024

## Result

| Task | n | v1 weighted F1 | v3 weighted F1 | Change | v3 accuracy | ChatGPT zero-shot | Best fine-tuned |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dreaddit | 414 (full test) | 59.89% | **79.30%** | +19.42 pp | 79.71% | 71.79% | 81.76% MentalRoBERTa |
| MultiWD | 500 (seed 42 sample) | 56.18% | **71.15%** | +14.97 pp | 70.60% | 62.72% | 76.69% BERT |

The v3 zero-shot protocol exceeds the paper's reported ChatGPT zero-shot point
on both tasks. It does not exceed the best task-specific fine-tuned classifier.
Comparisons use the same paper Table 2 reference values already encoded in the
evaluation task.

## Error analysis

The original generic binary prompt made `deepseek-chat` treat almost any
negative or clinically relevant wording as a positive label:

| Task | Protocol | no -> yes | yes -> no |
| --- | --- | ---: | ---: |
| dreaddit | v1 | 144 | 1 |
| dreaddit | v3 | 69 | 15 |
| MultiWD | v1 | 198 | 15 |
| MultiWD | v3 | 96 | 51 |

The correction reduces the dominant false-positive mode. The added false
negatives show that this is a real decision-boundary tradeoff rather than a
parser or label shortcut.

## Protocol change

The output format, dataset rows, gold parser, deterministic sampling, model,
temperature, and metric are unchanged. Only task definitions were made explicit.
No test labels or test-row answers appear in the prompt.

### dreaddit criteria

> Use the dataset's stress-detection meaning, not a broad clinical-safety
> screen. Answer yes when the poster expresses meaningful psychological strain,
> such as sustained overwhelm, helplessness, fear, anxiety, inability to cope,
> or an explicit statement of being stressed. Answer no for neutral
> descriptions, ordinary advice requests, isolated annoyance or disagreement,
> and descriptions of a difficult event that do not show the poster experiencing
> psychological strain. Do not infer stress merely because the topic is
> unpleasant or could be stressful.

### MultiWD criteria

> Decide whether the specific wellness dimension named in the question is
> substantively present in the post. Physical = bodily health, illness, sleep,
> nutrition, exercise, or physical self-care. Emotional = awareness, expression,
> acceptance, or regulation of feelings. Social = relationships, support,
> belonging, isolation, or social interaction. Vocational = work, career,
> education-to-career direction, or occupational purpose. Spiritual = meaning,
> purpose, values, faith, or one's place in life. Intellectual = learning,
> curiosity, creativity, problem solving, or cultural and intellectual activity.
> Answer yes for explicit or clearly demonstrated evidence, including impairment
> or absence in that dimension; answer no when the dimension is only remotely
> implied or when evidence belongs to another dimension.

## Reproduction and integrity

The external zero-dependency harness remains at `../datasets/eval-suite/` in the
owner's research workspace. The following commands produced the audited run:

```bash
node run.mjs imhi-dreaddit --selftest
node run.mjs imhi-multiwd --selftest
node run.mjs imhi-dreaddit --run-id v3 --concurrency 8
node run.mjs imhi-multiwd --run-id v3 --concurrency 8
```

SHA-256:

```text
b6e571d6567ce1089fd7901aa3a062fc6e3f8ee0cf06f6e5afb98099054514bd  tasks/imhi.mjs
f83b2382fcb719a23ae1efd2679ffadd6ee1b0985dbfc85a04fa95001b326a04  results/imhi-dreaddit-deepseek-chat-v3.jsonl
0f90083285a773837105d4e3464f62b35be289a075982e48b4a4d1fcf8d48b8a  results/imhi-dreaddit-deepseek-chat-v3.summary.json
9031632adbdf3ec8a8f053bd7cd3f9f10bc0d0adaafefa5f4911aeb0a188ef02  results/imhi-multiwd-deepseek-chat-v3.jsonl
423d0f88340c80306497b0bab4c77774be85f3519b530aaa1bb5a2749b602b40  results/imhi-multiwd-deepseek-chat-v3.summary.json
```

The public repository intentionally excludes raw IMHI social-media rows and
row-level model outputs. Those artifacts may contain sensitive third-party text
and remain governed by the source dataset's distribution terms.

## Interpretation limits

- This measures closed-set classification by the underlying model, not the
  quality or safety of the production counseling conversation pipeline.
- Prompt criteria were developed from dataset documentation and released
  training/validation examples; the test labels were not used as prompt content.
- MultiWD is a deterministic 500-row sample, matching the prior v1 protocol, not
  the entire 2,441-row test set.
- These benchmark results are research evidence, not clinical validation,
  diagnostic performance, or proof of production readiness.
