#!/bin/bash
# submit_from_checkpoint.sh — Submit a training job initialized from a prior model or checkpoint.
#
# Two usage modes:
#
#   1. HuggingFace model directory (e.g. from models/):
#        --model <path>   path to an HF model dir (has config.json + safetensors)
#        --config <name>  target training config
#
#      The script overrides actor/rollout model_path to <model>, and auto-merges
#      any missing dataset_statistics into that model dir so the target config's
#      unnorm_key resolves correctly.
#
#   2. FSDP training checkpoint (e.g. from logs/):
#        --from <source_config>   config name whose latest checkpoint to use
#        --to   <target_config>   target training config
#        [--step <N>]             use global_step_N instead of latest
#        [--save-as <name>]       convert checkpoint to HF and save to models/<name>
#                                 before submitting; required for multi-task chaining
#
#      With --save-as: converts full_weights.pt -> HF safetensors in models/<name>,
#      then submits using that saved model (same as mode 1).
#      Without --save-as: passes runner.ckpt_path directly (no model saved).
#
# Common flags:
#   --platform <LIBERO|BRIDGE>   default: LIBERO
#   --queue <normal|gpu_partition|preemptable>  default: normal
#   --logs-dir <path>            override logs search dir (default: $RLINF_LOGS_DIR or ./logs)
#   --dry-run                    print command without submitting
#
# Examples:
#   # HF model -> spatial task
#   bash scripts/submit_from_checkpoint.sh \
#       --model  models/Openvla-oft-grpo-object-step50 \
#       --config libero_pro_spatial/openvlaoft/libero_pro_spatial_openvlaoft_inplace_her_sym_rits
#
#   # FSDP checkpoint -> goal task, save HF model, then submit (recommended for chaining)
#   bash scripts/submit_from_checkpoint.sh \
#       --from     libero_pro_object_openvlaoft_inplace_her_sym_rits \
#       --to       libero_pro_goal_openvlaoft_inplace_her_sym_rits \
#       --save-as  Openvla-oft-grpo-pro-object-step500
#
#   # FSDP checkpoint -> goal task, specific step, no save
#   bash scripts/submit_from_checkpoint.sh \
#       --from libero_pro_object_openvlaoft_inplace_her_sym_rits \
#       --to   libero_pro_goal_openvlaoft_inplace_her_sym_rits \
#       --step 500

set -euo pipefail

RLINF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Default logs dir; override with --logs-dir or RLINF_LOGS_DIR env var.
LOGS_DIR="${RLINF_LOGS_DIR:-${RLINF_DIR}/logs}"
BV_SCRIPTS_DIR="${RLINF_DIR}/../cluster_scripts"

# ── argument parsing ─────────────────────────────────────────────────────────
MODEL_PATH=""       # mode 1: HF model dir
TARGET_CONFIG=""    # --config / --to
SOURCE_CONFIG=""    # mode 2: --from
STEP=""             # mode 2: --step
SAVE_AS=""          # mode 2: --save-as (convert to HF before submitting)
PLATFORM="LIBERO"
QUEUE="normal"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)    MODEL_PATH="$2";    shift 2 ;;
        --config)   TARGET_CONFIG="$2"; shift 2 ;;
        --from)     SOURCE_CONFIG="$2"; shift 2 ;;
        --to)       TARGET_CONFIG="$2"; shift 2 ;;
        --step)     STEP="$2";          shift 2 ;;
        --save-as)  SAVE_AS="$2";       shift 2 ;;
        --logs-dir) LOGS_DIR="$2";      shift 2 ;;
        --platform) PLATFORM="$2";      shift 2 ;;
        --queue)    QUEUE="$2";         shift 2 ;;
        --dry-run)  DRY_RUN=1;          shift ;;
        *) echo "Error: unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$TARGET_CONFIG" ]]; then
    echo "Error: --config (or --to) is required."
    exit 1
fi
if [[ -z "$MODEL_PATH" && -z "$SOURCE_CONFIG" ]]; then
    echo "Error: either --model <hf_dir> or --from <source_config> is required."
    exit 1
fi

# ── resolve target config file ────────────────────────────────────────────────
BARE_TARGET="$(basename "$TARGET_CONFIG")"
TARGET_CONFIG_FILE=""
for search_dir in "${RLINF_DIR}/config" "${RLINF_DIR}/examples/embodiment/config"; do
    if [[ -f "${search_dir}/${TARGET_CONFIG}.yaml" ]]; then
        TARGET_CONFIG_FILE="${search_dir}/${TARGET_CONFIG}.yaml"
        break
    fi
done
if [[ -z "$TARGET_CONFIG_FILE" ]]; then
    echo "Error: config file not found for '${TARGET_CONFIG}'"
    exit 1
fi

# ── mode 1: HF model directory ───────────────────────────────────────────────
if [[ -n "$MODEL_PATH" ]]; then
    # Resolve to absolute path (done again below after --save-as may set MODEL_PATH)
    if [[ "$MODEL_PATH" != /* ]]; then
        MODEL_PATH="${RLINF_DIR}/${MODEL_PATH}"
    fi

    if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
        echo "Error: '${MODEL_PATH}' does not look like an HF model dir (no config.json)"
        exit 1
    fi

    EXTRA_HYDRA_ARGS=(
        "actor.model.model_path=${MODEL_PATH}"
        "rollout.model.model_path=${MODEL_PATH}"
    )
    DISPLAY_SOURCE="$MODEL_PATH"
    CKPT_ARG=""

# ── mode 2: FSDP training checkpoint ─────────────────────────────────────────
else
    BARE_SOURCE="$(basename "$SOURCE_CONFIG")"

    find_checkpoint() {
        local config="$1"
        local step="$2"
        local best_ckpt=""
        for dir in "${LOGS_DIR}/${config}" "${LOGS_DIR}/"*"-${config}"; do
            [[ -d "$dir" ]] || continue
            if [[ -n "$step" ]]; then
                local ckpt="${dir}/${config}/checkpoints/global_step_${step}"
                [[ -d "$ckpt" ]] && best_ckpt="$ckpt" && break
            else
                local ckpt
                ckpt=$(ls -d "${dir}/${config}/checkpoints/global_step_"* 2>/dev/null | sort -V | tail -1)
                [[ -n "$ckpt" && -d "$ckpt" ]] && best_ckpt="$ckpt"
            fi
        done
        echo "$best_ckpt"
    }

    CKPT_DIR="$(find_checkpoint "$BARE_SOURCE" "$STEP")"
    if [[ -z "$CKPT_DIR" ]]; then
        echo "Error: no checkpoint found for '${BARE_SOURCE}' under ${LOGS_DIR}/"
        echo "Available dirs:"
        ls -d "${LOGS_DIR}/"*"-${BARE_SOURCE}" "${LOGS_DIR}/${BARE_SOURCE}" 2>/dev/null || echo "  (none)"
        exit 1
    fi

    WEIGHTS_FILE="${CKPT_DIR}/actor/model_state_dict/full_weights.pt"
    if [[ ! -f "$WEIGHTS_FILE" ]]; then
        echo "Error: weights file not found: ${WEIGHTS_FILE}"
        exit 1
    fi

    echo "Found checkpoint : ${CKPT_DIR}"

    if [[ -n "$SAVE_AS" ]]; then
        # Convert to HF format in models/ then use as mode 1
        MODEL_DEST="${RLINF_DIR}/models/${SAVE_AS}"
        if [[ -d "$MODEL_DEST" ]]; then
            echo "Model already exists at ${MODEL_DEST}, skipping conversion."
        else
            # Determine base model from source config's actor.model.model_path
            BASE_MODEL_PATH=$(grep -E '^\s+model_path:' "$TARGET_CONFIG_FILE" | head -1 \
                | sed 's/.*model_path:\s*["'"'"']\?//' | sed 's/["'"'"']\s*$//')
            if [[ "$BASE_MODEL_PATH" != /* ]]; then
                BASE_MODEL_PATH="${RLINF_DIR}/${BASE_MODEL_PATH}"
            fi
            echo "Converting checkpoint to HF format..."
            echo "  weights : ${WEIGHTS_FILE}"
            echo "  base    : ${BASE_MODEL_PATH}"
            echo "  output  : ${MODEL_DEST}"
            if [[ "$DRY_RUN" == "0" ]]; then
                bash "${RLINF_DIR}/scripts/load_checkpoint.sh" \
                    --checkpoint "${CKPT_DIR}" \
                    --name "${SAVE_AS}" \
                    --base-model "$(basename "$BASE_MODEL_PATH")"
            else
                echo "[dry-run] Would run: bash scripts/load_checkpoint.sh --checkpoint ${CKPT_DIR} --name ${SAVE_AS} --base-model $(basename "$BASE_MODEL_PATH")"
            fi
        fi
        # Switch to mode 1 using the saved model
        MODEL_PATH="$MODEL_DEST"
        EXTRA_HYDRA_ARGS=(
            "actor.model.model_path=${MODEL_PATH}"
            "rollout.model.model_path=${MODEL_PATH}"
        )
        CKPT_ARG=""
        DISPLAY_SOURCE="${MODEL_DEST} (converted from ${CKPT_DIR})"
    else
        EXTRA_HYDRA_ARGS=()
        CKPT_ARG="runner.ckpt_path=${WEIGHTS_FILE}"
        DISPLAY_SOURCE="$CKPT_DIR"
    fi
fi

# ── dataset_statistics merge (mode 1 path, including --save-as converted models) ──
if [[ -n "$MODEL_PATH" ]]; then
    if [[ "$MODEL_PATH" != /* ]]; then
        MODEL_PATH="${RLINF_DIR}/${MODEL_PATH}"
    fi

    CURRENT_MODEL_PATH=$(grep -E '^\s+model_path:' "$TARGET_CONFIG_FILE" | head -1 \
        | sed 's/.*model_path:\s*["'"'"']\?//' | sed 's/["'"'"']\s*$//')
    if [[ "$CURRENT_MODEL_PATH" != /* ]]; then
        CURRENT_MODEL_PATH="${RLINF_DIR}/${CURRENT_MODEL_PATH}"
    fi

    SRC_STATS="${MODEL_PATH}/dataset_statistics.json"
    REF_STATS="${CURRENT_MODEL_PATH}/dataset_statistics.json"

    if [[ -f "$REF_STATS" && "$MODEL_PATH" != "$CURRENT_MODEL_PATH" && -d "$MODEL_PATH" ]]; then
        python3 - <<PYEOF
import json, os

src_path = "${SRC_STATS}"
ref_path = "${REF_STATS}"

with open(ref_path) as f:
    ref = json.load(f)

src = {}
if os.path.isfile(src_path):
    with open(src_path) as f:
        src = json.load(f)

added = [key for key, val in ref.items() if key not in src and not src.update({key: val})]
if added:
    with open(src_path, 'w') as f:
        json.dump(src, f, indent=2)
    print(f"Merged dataset_statistics keys into ${MODEL_PATH}: {added}")
else:
    print(f"dataset_statistics already up to date in ${MODEL_PATH}")
PYEOF
    fi
fi

# ── pick bv submit script ─────────────────────────────────────────────────────
case "$QUEUE" in
    normal)      BV_SCRIPT="${BV_SCRIPTS_DIR}/submit-job.sh" ;;
    gpu_partition)  BV_SCRIPT="${BV_SCRIPTS_DIR}/submit-job-gpu.sh" ;;
    preemptable) BV_SCRIPT="${BV_SCRIPTS_DIR}/submit-job-preemptable.sh" ;;
    *)
        echo "Error: --queue must be 'normal', 'gpu_partition', or 'preemptable'"
        exit 1
        ;;
esac
[[ -f "$BV_SCRIPT" ]] || { echo "Error: bv script not found: ${BV_SCRIPT}"; exit 1; }

# ── build and submit the job ──────────────────────────────────────────────────
RUN_SCRIPT="${RLINF_DIR}/scripts/run_embodiment.sh"

CMD=(
    bash "$BV_SCRIPT"
    "$RUN_SCRIPT"
    --fresh
    "$TARGET_CONFIG"
    "$PLATFORM"
    "${EXTRA_HYDRA_ARGS[@]}"
)
[[ -n "$CKPT_ARG" ]] && CMD+=("$CKPT_ARG")

echo ""
echo "Submitting job:"
printf '  %q ' "${CMD[@]}"
echo ""
echo ""
echo "  Source model : ${DISPLAY_SOURCE}"
echo "  Target config: ${TARGET_CONFIG}"
echo "  Platform     : ${PLATFORM}"
echo "  Queue        : ${QUEUE}"
echo ""

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] Not submitting."
else
    cd "$RLINF_DIR"
    "${CMD[@]}"
fi
