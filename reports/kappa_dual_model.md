# Cohen's κ 报告

- 标注 A: `eval/annotations/annotator_A.jsonl`
- 标注 B: `eval/annotations/annotator_B.jsonl`
- 字段: `label`(4 类)

## 总体

| n | po | pe | κ | 解读 |
| --- | --- | --- | --- | --- |
| 344 | 0.7849 | 0.3052 | 0.6904 | Substantial(高度) |

## 按类别分解

| category | n | po | pe | κ | 解读 |
| --- | --- | --- | --- | --- | --- |
| diagnosis_request | 24 | 1.0000 | 1.0000 | 1.0000(退化) | Almost Perfect(几乎完全一致) |
| explicit_ideation | 24 | 0.7083 | 0.7083 | 0.0000 | Slight(轻微) |
| gradual_escalation | 88 | 0.8295 | 0.4082 | 0.7120 | Substantial(高度) |
| hard_negative_slang | 24 | 0.7500 | 0.5938 | 0.3846 | Fair(尚可) |
| medication_boundary | 24 | 0.9583 | 0.9583 | 0.0000 | Slight(轻微) |
| negation | 24 | 0.6250 | 0.3507 | 0.4225 | Moderate(中等) |
| passive_wish | 24 | 0.9583 | 0.9583 | 0.0000 | Slight(轻微) |
| pinyin_evasion | 24 | 0.9167 | 0.3281 | 0.8760 | Almost Perfect(几乎完全一致) |
| plain_sadness | 24 | 0.7917 | 0.7917 | 0.0000 | Slight(轻微) |
| recovery_statement | 40 | 0.3000 | 0.2500 | 0.0667 | Slight(轻微) |
| somatic_redflag | 24 | 1.0000 | 1.0000 | 1.0000(退化) | Almost Perfect(几乎完全一致) |
