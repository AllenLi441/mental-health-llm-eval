#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${ESCONV_LORA_MODE:-strategy}"
MODEL="${QWEN_MODEL:-Qwen/Qwen3.6-27B}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC="${NPROC_PER_NODE:-2}"
SMOKE="${ESCONV_SMOKE:-0}"

if [[ "$(uname -s)" != "Linux" ]] || [[ "$(uname -m)" != "x86_64" ]]; then
  echo "STOP: Qwen3.6-27B LoRA is frozen for Linux x86-64 CUDA, not this host." >&2
  exit 2
fi
command -v nvidia-smi >/dev/null || { echo "STOP: nvidia-smi is unavailable." >&2; exit 2; }
command -v swift >/dev/null || { echo "STOP: install a pinned ms-swift environment first." >&2; exit 2; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

DATA_ROOT="$PROJECT_ROOT/artifacts/esconv/data"
MANIFEST="$DATA_ROOT/modern_sft_manifest.json"
[[ -f "$MANIFEST" ]] || { echo "STOP: run prepare_esconv_modern_sft.py first." >&2; exit 2; }

case "$MODE" in
  strategy)
    TRAIN="$DATA_ROOT/esconv_strategy_train.jsonl"
    DEV="$DATA_ROOT/esconv_strategy_dev.jsonl"
    EPOCHS="${ESCONV_EPOCHS:-3}"
    LR="${ESCONV_LR:-5e-5}"
    MAX_LENGTH="${ESCONV_MAX_LENGTH:-1024}"
    OUTPUT="$PROJECT_ROOT/artifacts/esconv/training/qwen36_strategy_lora"
    ;;
  generation)
    TRAIN="$DATA_ROOT/esconv_generation_train.jsonl"
    DEV="$DATA_ROOT/esconv_generation_dev.jsonl"
    EPOCHS="${ESCONV_EPOCHS:-2}"
    LR="${ESCONV_LR:-2e-5}"
    MAX_LENGTH="${ESCONV_MAX_LENGTH:-2048}"
    OUTPUT="$PROJECT_ROOT/artifacts/esconv/training/qwen36_generation_lora"
    ;;
  *)
    echo "STOP: ESCONV_LORA_MODE must be strategy or generation." >&2
    exit 2
    ;;
esac

[[ -s "$TRAIN" && -s "$DEV" ]] || { echo "STOP: train/dev SFT files are missing." >&2; exit 2; }
if [[ "$SMOKE" == "1" ]]; then
  TRAIN="${TRAIN}#50"
  DEV="${DEV}#50"
  OUTPUT="${OUTPUT}_smoke"
fi
mkdir -p "$OUTPUT"

cat > "$OUTPUT/launch.env" <<EOF
MODE=$MODE
MODEL=$MODEL
CUDA_VISIBLE_DEVICES=$GPU_IDS
NPROC_PER_NODE=$NPROC
TRAIN=$TRAIN
DEV=$DEV
EPOCHS=$EPOCHS
LEARNING_RATE=$LR
MAX_LENGTH=$MAX_LENGTH
SMOKE=$SMOKE
SELECTION_RULE=dev_macro_f1_then_dev_accuracy_external_evaluator
EOF

CUDA_VISIBLE_DEVICES="$GPU_IDS" NPROC_PER_NODE="$NPROC" \
swift sft \
  --model "$MODEL" \
  --use_hf true \
  --tuner_type lora \
  --dataset "$TRAIN" \
  --val_dataset "$DEV" \
  --torch_dtype bfloat16 \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate "$LR" \
  --lora_rank "${ESCONV_LORA_RANK:-16}" \
  --lora_alpha "${ESCONV_LORA_ALPHA:-32}" \
  --target_modules all-linear \
  --max_length "$MAX_LENGTH" \
  --warmup_ratio 0.05 \
  --eval_steps 50 \
  --save_steps 50 \
  --save_total_limit 3 \
  --logging_steps 5 \
  --seed 42 \
  --data_seed 42 \
  --enable_thinking false \
  --add_non_thinking_prefix true \
  --output_dir "$OUTPUT"

echo "Training finished. Do not choose a checkpoint by test performance."
echo "Score saved checkpoints on dev; select by dev Macro-F1, then dev Accuracy."

