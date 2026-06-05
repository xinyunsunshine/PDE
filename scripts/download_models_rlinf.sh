#!/usr/bin/env bash
# Download RLinf models from HuggingFace.
#
# Usage:
#   bash scripts/download_models_rlinf.sh          # download all models
#   bash scripts/download_models_rlinf.sh base      # download one by shorthand
#   bash scripts/download_models_rlinf.sh 130       # download one by shorthand

set -euo pipefail

DEST="./models"

declare -A MODELS=(
    [base]="RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora"
    [130]="RLinf/RLinf-OpenVLAOFT-LIBERO-130"
    [object]="RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-object"
    [spatial]="RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-spatial"
    [goal]="RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-goal"
    [long]="RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-long"
    [pi05-sft]="RLinf/RLinf-Pi05-LIBERO-SFT"
)

download_model() {
    local repo="$1"
    local name="${repo#*/}"
    local target="${DEST}/${name}"

    if [[ -d "$target" ]]; then
        echo "[skip] ${repo} already exists at ${target}"
        return
    fi

    echo "[download] ${repo} -> ${target}"
    huggingface-cli download "$repo" --local-dir "$target"
}

if [[ $# -ge 1 ]]; then
    key="$1"
    if [[ -z "${MODELS[$key]+x}" ]]; then
        echo "Unknown model: $key"
        echo "Available: ${!MODELS[*]}"
        exit 1
    fi
    mkdir -p "$DEST"
    download_model "${MODELS[$key]}"
    exit 0
fi

# Default: download all
mkdir -p "$DEST"
for key in base 130 object spatial goal long pi05-sft; do
    download_model "${MODELS[$key]}"
done

echo "Done. All models saved to ${DEST}/"
