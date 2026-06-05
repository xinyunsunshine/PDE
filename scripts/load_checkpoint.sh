#! /bin/bash
# Convert an FSDP checkpoint (full_weights.pt) to HF format and save it to models/.
#
# Usage:
#   bash scripts/load_checkpoint.sh \
#       --checkpoint /path/to/global_step_N \
#       --base-model Openvla-oft-SFT-libero10-traj1
#
# --checkpoint  Path to the checkpoint directory. Searches for full_weights.pt at:
#               <path>/model_state_dict/full_weights.pt
#               <path>/actor/model_state_dict/full_weights.pt
#               <path>/full_weights.pt
# --base-model  Name of the base HF model in models/ used to load the architecture.
#
# Output model name is derived from the checkpoint path as <experiment>_<step>,
# e.g. libero_spatial_openvlaoft_grpo_global_step_100.

export RLINF_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd )"
cd "$RLINF_DIR"

source "$RLINF_DIR/.venv_openvla_oft/bin/activate"

export REPO_PATH="${RLINF_DIR}"
export PYTHONPATH="${RLINF_DIR}:$PYTHONPATH"

CKPT_PATH=""
BASE_MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            CKPT_PATH="$2"; shift 2 ;;
        --checkpoint=*)
            CKPT_PATH="${1#*=}"; shift ;;
        --base-model)
            BASE_MODEL="$2"; shift 2 ;;
        --base-model=*)
            BASE_MODEL="${1#*=}"; shift ;;
        --name|--name=*)
            echo "ERROR: --name is not allowed; model name is derived from the checkpoint path automatically."
            exit 1 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "${CKPT_PATH}" ] || [ -z "${BASE_MODEL}" ]; then
    echo "Usage: bash scripts/load_checkpoint.sh --checkpoint <path> --base-model <base>"
    exit 1
fi

STEP_NAME="$(basename "${CKPT_PATH}")"
EXPERIMENT_NAME="$(basename "$(dirname "$(dirname "${CKPT_PATH}")")")"
MODEL_NAME="${EXPERIMENT_NAME}_${STEP_NAME}"
MODEL_DEST="${RLINF_DIR}/models/${MODEL_NAME}"

if [ -d "${MODEL_DEST}" ]; then
    echo "ERROR: Model already exists at ${MODEL_DEST}. Remove it first or use a different checkpoint."
    exit 1
fi

FULL_WEIGHTS=""
for subpath in \
    "model_state_dict/full_weights.pt" \
    "actor/model_state_dict/full_weights.pt" \
    "full_weights.pt"; do
    if [ -f "${CKPT_PATH}/${subpath}" ]; then
        FULL_WEIGHTS="${CKPT_PATH}/${subpath}"
        break
    fi
done

if [ -z "${FULL_WEIGHTS}" ]; then
    echo "ERROR: Could not find full_weights.pt under ${CKPT_PATH}"
    exit 1
fi

echo "Converting checkpoint to HF format..."
echo "  weights : ${FULL_WEIGHTS}"
echo "  base    : ${RLINF_DIR}/models/${BASE_MODEL}"
echo "  output  : ${MODEL_DEST}"

python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf \
    --config-path "${RLINF_DIR}/rlinf/utils/ckpt_convertor/fsdp_convertor/config" \
    --config-name fsdp_model_convertor \
    "convertor.ckpt_path=${FULL_WEIGHTS}" \
    "convertor.save_path=${MODEL_DEST}" \
    "model.model_path=${RLINF_DIR}/models/${BASE_MODEL}" \
    "model.is_lora=False"

if [ $? -ne 0 ]; then
    echo "ERROR: Conversion failed"
    exit 1
fi

echo "Done. Model saved to ${MODEL_DEST}"
