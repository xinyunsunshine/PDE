#!/bin/bash
#
# Submit multiple seed variants of a config via Hydra CLI overrides (no temp files).
#
# Usage:
#   bash scripts/submit_seed_sweep.sh <config_name> <bv_script> [seed_start] [seed_end] \
#       [--dry-run] [--experiment-name <name>] [--experiment-suffix <suffix>] \
#       [--overrides <key=val> ...]
#
# Example:
#   # Basic seed sweep (reads experiment_name from config, shared across seeds):
#   bash scripts/submit_seed_sweep.sh \
#       libero_object/openvlaoft/libero_object_openvlaoft_her_512batch_1ue_32micro_256step_anchorfix \
#       submit-job-preemptable.sh 1 4
#
#   # With suffix and Hydra overrides (builds on config's experiment_name):
#   bash scripts/submit_seed_sweep.sh \
#       libero_object/openvlaoft/libero_object_openvlaoft_her_512batch_1ue_32micro_256step_anchorfix \
#       submit-job.sh 1 4 \
#       --experiment-suffix 'lr5em6_warmup3' \
#       --overrides 'actor.optim.lr=5e-6' '+actor.optim.lr_warmup_steps=3'
#   # -> experiment_name: libero_object_openvlaoft_her_512batch_1ue_32micro_256step_anchorfix_lr5em6_warmup3
#   # -> logs dir per seed: logs/<config_basename>_seed1, logs/<config_basename>_seed2, ...
#
# All seeds share the same experiment_name in W&B. Each seed gets its own logs/
# directory with a _seed<N> suffix.
#
# --experiment-name: explicit experiment name (shared across all seeds).
# --experiment-suffix: appended to config's experiment_name as _{suffix}.
# If neither is given, reads experiment_name from config and uses it as-is.
# --overrides: additional Hydra CLI overrides appended to every seed's command.

# by default you shoule use submit-job-preemptable.sh to submit

set -euo pipefail

RLINF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    echo "Usage: $0 <config_name> <bv_script> [seed_start] [seed_end] [--dry-run] [--experiment-name <template>] [--experiment-suffix <suffix>] [--overrides <key=val> ...]"
    echo ""
    echo "  config_name         - relative config path"
    echo "  bv_script           - cluster submission script name (e.g. submit-job-u2.sh)"
    echo "  seed_start          - first seed (default: 1)"
    echo "  seed_end            - last seed (default: 4)"
    echo "  --dry-run           - print commands without submitting"
    echo "  --experiment-name   - explicit experiment name (shared across all seeds)"
    echo "  --experiment-suffix - appended to config's experiment_name as _{suffix}"
    echo "  --overrides         - additional Hydra CLI overrides (e.g. 'actor.optim.lr=5e-6')"
    exit 1
}

if [ "$#" -lt 2 ]; then
    usage
fi

CONFIG_NAME="$1"
BV_SCRIPT="$2"
shift 2

SEED_START=""
SEED_END=""
DRY_RUN=0
EXPERIMENT_TEMPLATE=""
EXPERIMENT_SUFFIX=""
EXTRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --experiment-name)
            EXPERIMENT_TEMPLATE="$2"
            shift 2
            ;;
        --experiment-suffix)
            EXPERIMENT_SUFFIX="$2"
            shift 2
            ;;
        --overrides)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                EXTRA_OVERRIDES+=("$1")
                shift
            done
            ;;
        *)
            if [ -z "$SEED_START" ]; then
                SEED_START="$1"
            elif [ -z "$SEED_END" ]; then
                SEED_END="$1"
            fi
            shift
            ;;
    esac
done

SEED_START="${SEED_START:-1}"
SEED_END="${SEED_END:-4}"

CONFIG_FILE="${RLINF_DIR}/config/${CONFIG_NAME}.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

BV_SCRIPT_PATH="${RLINF_DIR}/../cluster_scripts/${BV_SCRIPT}"
if [ ! -f "$BV_SCRIPT_PATH" ]; then
    echo "ERROR: bv script not found: $BV_SCRIPT_PATH"
    exit 1
fi

CONFIG_BASENAME="$(basename "$CONFIG_NAME")"

if [ -z "$EXPERIMENT_TEMPLATE" ]; then
    BASE_EXPERIMENT_NAME=$(grep -oP 'experiment_name:\s*\K\S+' "$CONFIG_FILE")
    if [ -z "$BASE_EXPERIMENT_NAME" ]; then
        echo "ERROR: Could not find experiment_name in $CONFIG_FILE"
        exit 1
    fi
fi

# Determine the shared experiment name (same for all seeds).
if [ -n "$EXPERIMENT_TEMPLATE" ]; then
    SHARED_EXPERIMENT_NAME="$EXPERIMENT_TEMPLATE"
elif [ -n "$EXPERIMENT_SUFFIX" ]; then
    SHARED_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_${EXPERIMENT_SUFFIX}"
else
    SHARED_EXPERIMENT_NAME="$BASE_EXPERIMENT_NAME"
fi

for SEED in $(seq "$SEED_START" "$SEED_END"); do
    SEED_LOG_DIR="${RLINF_DIR}/logs/${SHARED_EXPERIMENT_NAME}_seed${SEED}"

    CMD=(
        bash "$BV_SCRIPT_PATH"
        scripts/run_embodiment.sh
        "$CONFIG_NAME" LIBERO --no-timestamp-prefix
        "actor.seed=${SEED}"
        "runner.logger.experiment_name=${SHARED_EXPERIMENT_NAME}"
        "runner.logger.log_path=${SEED_LOG_DIR}"
        ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}
    )

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[DRY-RUN] ${CMD[*]}"
    else
        echo "Submitting seed=$SEED: ${CMD[*]}"
        (cd "$RLINF_DIR" && "${CMD[@]}")
    fi
done

echo ""
echo "Done. Submitted seeds ${SEED_START}-${SEED_END} for ${CONFIG_BASENAME}."
