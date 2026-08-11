#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${ESCONV_RUN_ROOT:-$PROJECT_ROOT/tmp/esconv2021_faithful}"
WORK_DIR="${ESCONV_WORK_DIR:-$RUN_ROOT/work}"
CODES="$WORK_DIR/codes"
CONDA_ENV="${ESCONV_CONDA_ENV:-esconv2021}"
SEED="${ESCONV_SEED:-42}"
OUTPUT_DIR="$RUN_ROOT/blender_strategy_repro_seed${SEED}"
LOG_DIR="$RUN_ROOT/logs"
LOG_FILE="$LOG_DIR/train_joint_seed${SEED}.log"
SELECTION="$RUN_ROOT/best_checkpoint_seed${SEED}.json"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "Faithful training requires Linux x86-64; refusing to run on $(uname -s) $(uname -m)." >&2
  exit 2
fi
command -v nvidia-smi >/dev/null || { echo "NVIDIA GPU is required for the faithful run" >&2; exit 2; }
command -v conda >/dev/null || { echo "conda is required" >&2; exit 2; }
[[ -f "$CODES/BlenderEmotionalSupport.py" ]] || { echo "Run setup_esconv_legacy_env.sh first" >&2; exit 2; }
[[ -s "$CODES/blender-small/pytorch_model.bin" ]] || { echo "Base BlenderBot weight is missing" >&2; exit 2; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$CODES"
set +e
PYTHONHASHSEED="$SEED" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
conda run --no-capture-output -n "$CONDA_ENV" env \
  PYTHONPATH="$CODES" \
  ESCONV_OUTPUT_DIR="$OUTPUT_DIR" \
  ESCONV_MODEL_PATH="./blender-small" \
  ESCONV_CONFIG_PATH="./blender-small" \
  ESCONV_TOKENIZER_PATH="./blender-small" \
  ESCONV_DATA_PATH="./dataset" \
  ESCONV_TRAIN_FILE="trainWithStrategy_short.tsv" \
  ESCONV_EVAL_FILE="devWithStrategy_short.tsv" \
  ESCONV_CACHE_DIR="$RUN_ROOT/cached_repro_seed${SEED}" \
  ESCONV_DO_TRAIN=1 ESCONV_DO_EVAL=0 \
  ESCONV_GENERATION=0 ESCONV_GENERATE_AND_EVAL=0 \
  ESCONV_EVALUATE_DURING_TRAINING=1 ESCONV_NO_CUDA=0 \
  ESCONV_SEED="$SEED" \
  python BlenderEmotionalSupport.py 2>&1 | tee "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

python3 "$PROJECT_ROOT/scripts/select_esconv_best_checkpoint.py" \
  --output-dir "$OUTPUT_DIR" \
  --training-log "$LOG_FILE" \
  --selection-output "$SELECTION"

echo "Training complete. Best-checkpoint selection: $SELECTION"
