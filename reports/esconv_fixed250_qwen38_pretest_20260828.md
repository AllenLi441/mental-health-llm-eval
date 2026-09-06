# ESConv fixed250 paired inference

Protocol: `provider_api_v2 JSON strategy+response; missing/invalid counts wrong`

Items: 250; reconstructed dialogue clusters: 19.

## Arms

| Model | Accuracy | Cluster-bootstrap 95% CI | Macro-F1 | Weighted-F1 | Missing/invalid |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B（未微调·DashScope） | 13.60% | [9.20%, 18.62%] | 9.60% | 11.82% | 0/0 |
| Qwen3.6-27B（未微调） | 14.00% | [9.66%, 19.15%] | 10.44% | 11.80% | 0/0 |
| DeepSeek V4-Pro-0813 | 16.00% | [12.64%, 19.92%] | 9.64% | 13.66% | 10/12 |
| DeepSeek V4-Flash | 16.40% | [12.08%, 21.08%] | 12.26% | 14.85% | 3/4 |

## Pairwise

| Comparison (A-B) | Accuracy delta | Cluster-bootstrap 95% CI | Only A correct | Only B correct | McNemar p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B（未微调·DashScope） - Qwen3.6-27B（未微调） | -0.40% | [-3.86%, +3.06%] | 10 | 11 | 1 | 1 |
| Qwen3.8-27B（未微调·DashScope） - DeepSeek V4-Pro-0813 | -2.40% | [-4.86%, +0.00%] | 5 | 11 | 0.210114 | 1 |
| Qwen3.8-27B（未微调·DashScope） - DeepSeek V4-Flash | -2.80% | [-5.84%, +0.39%] | 6 | 13 | 0.167068 | 1 |
| Qwen3.6-27B（未微调） - DeepSeek V4-Pro-0813 | -2.00% | [-5.41%, +1.21%] | 11 | 16 | 0.442068 | 1 |
| Qwen3.6-27B（未微调） - DeepSeek V4-Flash | -2.40% | [-6.79%, +2.06%] | 9 | 15 | 0.307456 | 1 |
| DeepSeek V4-Pro-0813 - DeepSeek V4-Flash | -0.40% | [-2.76%, +2.51%] | 11 | 12 | 1 | 1 |

These are fixed250 development results. They do not replace a full-test score, and they do not transfer to the new one-letter LoRA protocol.
