# ESConv 250-item three-model baseline

All primary metrics use the same 250-item denominator. Missing outputs and out-of-label strategies count as incorrect.

| Model | Coverage | Accuracy | Macro-F1 | Weighted-F1 | Invalid | Median latency | P95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek V4-Pro | 95.6% | 14.80% | 9.48% | 13.08% | 17 | 10.62s | 45.91s |
| DeepSeek V4-Flash | 98.8% | 16.40% | 12.26% | 14.85% | 4 | 2.78s | 11.52s |
| Qwen3.6-27B | 100.0% | 14.00% | 10.44% | 11.80% | 0 | 1.29s | 1.84s |

All-three common task set: n=238.

| Model | Common-set Accuracy | Common-set Macro-F1 | Common-set Weighted-F1 |
|---|---:|---:|---:|
| DeepSeek V4-Pro | 15.55% | 9.69% | 13.69% |
| DeepSeek V4-Flash | 16.39% | 10.79% | 14.44% |
| Qwen3.6-27B | 13.87% | 8.78% | 11.06% |

This is the frozen pre-fine-tuning baseline. Reuse the exact same task IDs and scoring after fine-tuning.
