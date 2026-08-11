# ESConv 2021 faithful reproduction

This directory documents the auditable reproduction of the BlenderBot-small
90M Joint model from *Towards Emotional Support Dialog Systems* (ACL 2021).

## Frozen scope

- Official repository commit: `f262d062ad74cb39b17ea476facc81568ddcba24`
- Implementation: official `codes/`, not `codes_zcj/`
- Dataset: the original 1,053-dialogue fixed TSV split in `codes/dataset/`
- Model: BlenderBot-small 90M Joint
- Training: 5 epochs, Adam, `5e-5`, batch size 4, warmup 120, seed 42
- Selection: lowest validation PPL, never the last checkpoint by default
- Generation: top-p 0.9, top-k 30, temperature 0.7, repetition penalty 1.03

## Frozen checkpoint token map

| Token | ID |
|---|---:|
| `[Questions]` | 54944 |
| `[Reflection of feelings]` | 54945 |
| `[Information]` | 54946 |
| `[Restatement or Paraphrasing]` | 54947 |
| `[Other]` | 54948 |
| `[Self-disclosure]` | 54949 |
| `[Affirmation and Reassurance]` | 54950 |
| `[Providing Suggestions]` | 54951 |
| `[CLS]` | 54952 |

`[Question]`, `[Others]`, positional label interpretation, and contiguous logit
slices are prohibited.

## Execution order

```bash
bash scripts/setup_esconv_legacy_env.sh
bash scripts/train_esconv_2021_joint.sh
bash scripts/generate_esconv_2021_joint.sh
```

The setup and training scripts intentionally refuse a faithful run outside
Linux x86-64 with NVIDIA CUDA. The current Apple M4 workstation can run the
modern corrected diagnostic evaluator, but it cannot establish the legacy
training result.

Corrected checkpoint diagnostics use:

```bash
python scripts/eval_esconv_joint_corrected.py \
  --checkpoint /path/to/pytorch_model.bin \
  --run-name official_joint_modern_full \
  --summary-output reports/esconv_joint_official_corrected_full.json \
  --predictions-output reports/esconv_joint_official_corrected_full.predictions.jsonl \
  --manifest reports/esconv_joint_reproduction_manifest.json
```

The evaluator saves every gold label, predicted label, selected token ID, and
all eight strategy logits. `compare_esconv_modern_legacy.py` compares two such
JSONL files by `item_id`.

## Result semantics

The ACL 2021 paper did not publish strategy classification Accuracy. Any
Accuracy produced here is a post-hoc project metric and must not be presented
as an original paper result. PPL, BLEU-2, ROUGE-L, and Extrema are the paper's
generation metrics.
