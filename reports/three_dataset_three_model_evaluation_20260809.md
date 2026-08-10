# Three-dataset, three-model evaluation

## ESConv strategy prediction (fixed n=250)

| Model | Accuracy | Macro-F1 | Weighted-F1 | Invalid/missing |
|---|---:|---:|---:|---:|
| DeepSeek V4-Pro | 14.80% | 9.48% | 13.08% | 17 |
| DeepSeek V4-Flash | 16.40% | 12.26% | 14.85% | 4 |
| Qwen3.6-27B | 14.00% | 10.44% | 11.80% | 0 |
| Published Causal-ESC | 53.53% | not reported | not reported | not reported |

## CPCD proxy judge scores

Proxy: DeepSeek V4-Flash non-thinking with official CPCD rubrics. Conservative scores penalize missing outputs.

| Model | SR observed | SR conservative | MR observed | MR conservative | TCR observed | TCR conservative | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek V4-Pro | 4.057 | 4.057 | 4.756 | 4.756 | 4.438 | 4.438 | 159/159 |
| DeepSeek V4-Flash | 4.061 | 4.061 | 4.699 | 4.581 | 3.500 | 3.150 | 156/159 |
| Qwen3.6-27B | 3.943 | 3.943 | 4.655 | 3.375 | 4.344 | 1.738 | 136/159 |
| Published GPT-5.4 (GPT-5.2 judge) | 4.778 | 4.778 | 4.513 | 4.513 | 4.525 | 4.525 | 159/159 |

## AugESC-style downstream response metrics (ESConv fixed n=250)

AugESC has no independent Accuracy.

| Model | BLEU-2 | BLEU-4 | ROUGE-L | Distinct-1 | Distinct-2 | Distinct-3 |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek V4-Pro | 4.13 | 0.42 | 12.14 | 15.55 | 56.78 | 79.58 |
| DeepSeek V4-Flash | 4.21 | 0.55 | 13.42 | 11.95 | 50.57 | 75.37 |
| Qwen3.6-27B | 5.45 | 1.09 | 14.84 | 14.99 | 51.90 | 72.94 |
| Published BlenderBot 1.4B + AugESC | 7.70 | 2.40 | 16.70 | not reported | 24.30 | 49.40 |

CPCD proxy values cannot be presented as GPT-5.2 official scores. See the JSON report for coverage and missing-output policy.
