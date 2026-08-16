# ESConv fixed250 paired inference

Protocol: `provider_api_v2 JSON strategy+response; missing/invalid counts wrong`

Items: 250; reconstructed dialogue clusters: 19.

## Arms

| Model | Accuracy | Cluster-bootstrap 95% CI | Macro-F1 | Weighted-F1 | Missing/invalid |
|---|---:|---:|---:|---:|---:|
| DeepSeek V4-Pro | 14.80% | [10.33%, 19.92%] | 9.48% | 13.08% | 11/17 |
| DeepSeek V4-Flash | 16.40% | [12.13%, 21.03%] | 12.26% | 14.85% | 3/4 |
| Qwen3.6-27B | 14.00% | [9.62%, 19.20%] | 10.44% | 11.80% | 0/0 |

## Pairwise

| Comparison (A-B) | Accuracy delta | Cluster-bootstrap 95% CI | Only A correct | Only B correct | McNemar p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek V4-Pro - DeepSeek V4-Flash | -1.60% | [-5.11%, +2.08%] | 9 | 13 | 0.523467 | 1 |
| DeepSeek V4-Pro - Qwen3.6-27B | +0.80% | [-3.10%, +4.56%] | 12 | 10 | 0.831812 | 1 |
| DeepSeek V4-Flash - Qwen3.6-27B | +2.40% | [-2.07%, +6.77%] | 15 | 9 | 0.307456 | 0.922369 |

These are fixed250 development results. They do not replace a full-test score, and they do not transfer to the new one-letter LoRA protocol.
