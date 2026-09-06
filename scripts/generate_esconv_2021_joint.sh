#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${ESCONV_RUN_ROOT:-$PROJECT_ROOT/tmp/esconv2021_faithful}"
WORK_DIR="${ESCONV_WORK_DIR:-$RUN_ROOT/work}"
CODES="$WORK_DIR/codes"
CONDA_ENV="${ESCONV_CONDA_ENV:-esconv2021}"
SEED="${ESCONV_SEED:-42}"
SELECTION="$RUN_ROOT/best_checkpoint_seed${SEED}.json"
LOG_DIR="$RUN_ROOT/logs"
LOG_FILE="$LOG_DIR/generate_joint_seed${SEED}.log"
GENERATED="$RUN_ROOT/generated/joint_seed${SEED}.json"

[[ -f "$SELECTION" ]] || { echo "Missing $SELECTION; train and select the best checkpoint first" >&2; exit 2; }
BEST_CHECKPOINT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint"])' "$SELECTION")"
[[ -d "$BEST_CHECKPOINT" ]] || { echo "Best checkpoint does not exist: $BEST_CHECKPOINT" >&2; exit 2; }
mkdir -p "$LOG_DIR" "$(dirname "$GENERATED")"

cd "$CODES"
PYTHONHASHSEED="$SEED" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
conda run --no-capture-output -n "$CONDA_ENV" env \
  PYTHONPATH="$CODES" \
  ESCONV_OUTPUT_DIR="$BEST_CHECKPOINT" \
  ESCONV_MODEL_PATH="./blender-small" \
  ESCONV_CONFIG_PATH="./blender-small" \
  ESCONV_TOKENIZER_PATH="./blender-small" \
  ESCONV_DATA_PATH="./dataset" \
  ESCONV_TRAIN_FILE="trainWithStrategy_short.tsv" \
  ESCONV_EVAL_FILE="testWithStrategy_short.tsv" \
  ESCONV_DO_TRAIN=0 ESCONV_DO_EVAL=0 \
  ESCONV_GENERATION=0 ESCONV_GENERATE_AND_EVAL=1 \
  ESCONV_EVALUATE_DURING_TRAINING=0 ESCONV_NO_CUDA=0 \
  ESCONV_GENERATED_OUTPUT="$GENERATED" ESCONV_SEED="$SEED" \
  python BlenderEmotionalSupport.py 2>&1 | tee "$LOG_FILE"

conda run --no-capture-output -n "$CONDA_ENV" env PYTHONPATH="$CODES" \
python "$PROJECT_ROOT/scripts/eval_esconv_joint_corrected.py" \
  --backend legacy --device cuda --checkpoint "$BEST_CHECKPOINT" \
  --tokenizer-dir "$BEST_CHECKPOINT" --config-dir "$CODES/blender-small" \
  --test-file "$CODES/dataset/testWithStrategy_short.tsv" \
  --official-repo "${ESCONV_OFFICIAL_REPO:-$RUN_ROOT/official}" \
  --run-name "self_trained_seed${SEED}_legacy_full" \
  --summary-output "$PROJECT_ROOT/reports/esconv_joint_self_trained_seed${SEED}_legacy_full.json" \
  --predictions-output "$PROJECT_ROOT/reports/esconv_joint_self_trained_seed${SEED}_legacy_full.predictions.jsonl" \
  --manifest "$PROJECT_ROOT/reports/esconv_joint_reproduction_manifest.json"

echo "Generation and corrected legacy evaluation complete: $LOG_FILE"
