# ESConv fixed250 paired inference

Protocol: `provider_api_v2 JSON strategy+response; missing/invalid counts wrong`

Items: 250; reconstructed dialogue clusters: 19.

## Arms

| Model | Accuracy | Cluster-bootstrap 95% CI | Macro-F1 | Weighted-F1 | Missing/invalid |
|---|---:|---:|---:|---:|---:|
| DeepSeek V4-Flash | 16.40% | [12.13%, 21.12%] | 12.26% | 14.85% | 3/4 |
| DeepSeek V4-Pro（旧测） | 14.80% | [10.31%, 19.92%] | 9.48% | 13.08% | 11/17 |
| Qwen3.6-27B（未微调） | 14.00% | [9.62%, 19.20%] | 10.44% | 11.80% | 0/0 |
| DeepSeek V4-Pro-0813 | 16.00% | [12.59%, 20.00%] | 9.64% | 13.66% | 10/12 |

## Pairwise

| Comparison (A-B) | Accuracy delta | Cluster-bootstrap 95% CI | Only A correct | Only B correct | McNemar p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek V4-Flash - DeepSeek V4-Pro（旧测） | +1.60% | [-2.08%, +5.11%] | 13 | 9 | 0.523467 | 1 |
| DeepSeek V4-Flash - Qwen3.6-27B（未微调） | +2.40% | [-2.08%, +6.75%] | 15 | 9 | 0.307456 | 1 |
| DeepSeek V4-Flash - DeepSeek V4-Pro-0813 | +0.40% | [-2.46%, +2.77%] | 12 | 11 | 1 | 1 |
| DeepSeek V4-Pro（旧测） - Qwen3.6-27B（未微调） | +0.80% | [-3.09%, +4.56%] | 12 | 10 | 0.831812 | 1 |
| DeepSeek V4-Pro（旧测） - DeepSeek V4-Pro-0813 | -1.20% | [-4.78%, +2.06%] | 7 | 10 | 0.629059 | 1 |
| Qwen3.6-27B（未微调） - DeepSeek V4-Pro-0813 | -2.00% | [-5.37%, +1.21%] | 11 | 16 | 0.442068 | 1 |

These are fixed250 development results. They do not replace a full-test score, and they do not transfer to the new one-letter LoRA protocol.
