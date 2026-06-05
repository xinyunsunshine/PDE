#!/usr/bin/env bash
# Download Openvla-oft-SFT models from HuggingFace.
#
# Usage:
#   bash scripts/download_models.sh              # download all 8 models
#   bash scripts/download_models.sh spatial all   # download one: libero-spatial, trajall

set -euo pipefail

DEST="./models"
ORG="Haozhan72"
PREFIX="Openvla-oft-SFT"

SUITES=("libero10" "libero-object" "libero-goal" "libero-spatial")
TRAJS=("traj1" "trajall")

download_model() {
    local suite="$1" traj="$2"
    local repo="${ORG}/${PREFIX}-${suite}-${traj}"
    local target="${DEST}/${PREFIX}-${suite}-${traj}"

    if [[ -d "$target" ]]; then
        echo "[skip] ${repo} already exists at ${target}"
        return
    fi

    echo "[download] ${repo} -> ${target}"
    huggingface-cli download "$repo" --local-dir "$target"
}

# If arguments are provided, download a single model
if [[ $# -ge 2 ]]; then
    suite_arg="$1"
    traj_arg="$2"

    # Normalise suite arg: "10" -> "libero10", "spatial" -> "libero-spatial", etc.
    case "$suite_arg" in
        10|libero10)       suite="libero10" ;;
        object|libero-object)   suite="libero-object" ;;
        goal|libero-goal)       suite="libero-goal" ;;
        spatial|libero-spatial) suite="libero-spatial" ;;
        *) echo "Unknown suite: $suite_arg (expected: 10, object, goal, spatial)"; exit 1 ;;
    esac

    # Normalise traj arg: "1" -> "traj1", "all" -> "trajall", etc.
    case "$traj_arg" in
        1|traj1)     traj="traj1" ;;
        all|trajall) traj="trajall" ;;
        *) echo "Unknown traj: $traj_arg (expected: 1, all)"; exit 1 ;;
    esac

    download_model "$suite" "$traj"
    exit 0
fi

# Default: download all combinations
mkdir -p "$DEST"
for suite in "${SUITES[@]}"; do
    for traj in "${TRAJS[@]}"; do
        download_model "$suite" "$traj"
    done
done

echo "Done. All models saved to ${DEST}/"
