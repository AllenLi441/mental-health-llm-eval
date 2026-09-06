#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${ESCONV_RUN_ROOT:-$PROJECT_ROOT/tmp/esconv2021_faithful}"
OFFICIAL_REPO="${ESCONV_OFFICIAL_REPO:-$RUN_ROOT/official}"
WORK_DIR="${ESCONV_WORK_DIR:-$RUN_ROOT/work}"
BASE_MODEL_DIR="${ESCONV_BASE_MODEL_DIR:-$RUN_ROOT/blenderbot-small-90m}"
CONDA_ENV="${ESCONV_CONDA_ENV:-esconv2021}"
COMMIT="f262d062ad74cb39b17ea476facc81568ddcba24"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  if [[ "${ALLOW_UNSUPPORTED_PLATFORM:-0}" != "1" ]]; then
    echo "Faithful legacy setup requires Linux x86-64. Current: $(uname -s) $(uname -m)" >&2
    exit 2
  fi
fi
command -v conda >/dev/null || { echo "conda is required" >&2; exit 2; }
command -v git >/dev/null || { echo "git is required" >&2; exit 2; }
command -v git-lfs >/dev/null || { echo "git-lfs is required" >&2; exit 2; }

mkdir -p "$RUN_ROOT"
if [[ ! -d "$OFFICIAL_REPO/.git" ]]; then
  git clone https://github.com/thu-coai/Emotional-Support-Conversation.git "$OFFICIAL_REPO"
fi
git -C "$OFFICIAL_REPO" fetch origin "$COMMIT"
git -C "$OFFICIAL_REPO" checkout --detach "$COMMIT"
[[ "$(git -C "$OFFICIAL_REPO" rev-parse HEAD)" == "$COMMIT" ]]

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  conda create -n "$CONDA_ENV" python=3.7 -y
fi
conda install -n "$CONDA_ENV" pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=10.1 -c pytorch -y
conda run -n "$CONDA_ENV" pip install \
  transformers==4.2.2 numpy==1.19.5 pandas==1.3.5 \
  scikit-learn==0.24.2 scipy==1.7.3 tqdm tensorboardX

if [[ ! -f "$BASE_MODEL_DIR/pytorch_model.bin" ]]; then
  git lfs install
  git clone https://huggingface.co/facebook/blenderbot_small-90M "$BASE_MODEL_DIR"
fi
[[ -s "$BASE_MODEL_DIR/pytorch_model.bin" ]]

python3 "$PROJECT_ROOT/scripts/prepare_esconv_2021_official_run.py" \
  --official-repo "$OFFICIAL_REPO" \
  --work-dir "$WORK_DIR" \
  --base-weight "$BASE_MODEL_DIR/pytorch_model.bin" \
  --force

echo "Legacy environment and official-code work copy are ready under $RUN_ROOT"
