# ESConv execution status, 2026-08-12

## Completed locally

- Official Joint checkpoint corrected top-1 replay: `894/2,775 = 32.2162%`.
- Same checkpoint on fixed250: `94/250 = 37.60%`.
- Exact strategy token mapping and explicit eight-ID logit extraction.
- Per-item predictions, Accuracy, Macro-F1, Weighted-F1, per-class metrics and confusion matrices.
- Historical setup, training, checkpoint-selection and generation scripts.
- Modern train/dev strategy-only and strategy-conditioned generation data builder.
- Existing three-API fixed250 paired inference script with dialogue-cluster bootstrap and exact McNemar.

## Not completed locally

- Python 3.7 / PyTorch 1.7.1 legacy checkpoint replay.
- Historical Joint retraining for seeds 42, 43 and 44.
- Qwen3.6-27B LoRA training.

The workstation is Apple Silicon with 24 GiB unified memory and no NVIDIA CUDA.
The historical reproduction scripts therefore stop outside Linux x86-64 CUDA,
and the Qwen LoRA launcher has the same hard guard. This prevents a modern MPS
run from being mislabeled as the requested historical or 27B CUDA experiment.

## Protocol separation

The existing API fixed250 scores use a JSON object containing both strategy and
response. The new LoRA data uses a one-letter strategy-only target. These are
separate tracks. The API scores cannot be copied into the new track; all models
must be rerun with the one-letter prompt before direct zero-shot-versus-LoRA
claims are made.

DPPLM is excluded because no local full paper was available for protocol review.

