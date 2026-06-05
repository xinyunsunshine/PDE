#! /bin/bash

export RLINF_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd )"
cd "$RLINF_DIR"
if [ -f "$RLINF_DIR/.venv_libero/bin/activate" ]; then
    source "$RLINF_DIR/.venv_libero/bin/activate"
fi

if [ -f "$RLINF_DIR/.env.wandb" ]; then
    echo "Loading W&B config from .env.wandb"
    source "$RLINF_DIR/.env.wandb"
else
    echo "WARNING: .env.wandb not found"
fi

if [ -f "$RLINF_DIR/.env" ]; then
    echo "Loading API keys from .env"
    source "$RLINF_DIR/.env"
fi

CACHE_BASE=/PROJECT_ROOT/.cache
mkdir -p "$CACHE_BASE"/{huggingface,transformers,torch,xdg}

export TOKENIZERS_PARALLELISM=false

export HOME=/PROJECT_ROOT
export HF_HOME=$CACHE_BASE/huggingface
export HF_HUB_CACHE=$CACHE_BASE/huggingface/hub
export TRANSFORMERS_CACHE=$CACHE_BASE/transformers
export TORCH_HOME=$CACHE_BASE/torch
export XDG_CACHE_HOME=$CACHE_BASE/xdg
export HF_HUB_DISABLE_TELEMETRY=1
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_${USER}}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export ROBOT_PLATFORM=BRIDGE

export VULKAN_SDK=/PROJECT_ROOT/vulkan/1.4.341.1/x86_64
export PATH=$VULKAN_SDK/bin:$PATH
export LD_LIBRARY_PATH=$VULKAN_SDK/lib:$LD_LIBRARY_PATH
export VK_LAYER_PATH=$VULKAN_SDK/share/vulkan/explicit_layer.d
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json}
export VK_DRIVER_FILES=$VK_ICD_FILENAMES

export EMBODIED_PATH="${RLINF_DIR}/examples/embodiment"
export REPO_PATH="${RLINF_DIR}"
export SRC_FILE="${EMBODIED_PATH}/collect_rollout_videos.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

export RAY_DEBUG=legacy
python -m ray stop --force >/dev/null 2>&1 || true

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <config_name> [ROBOT_PLATFORM] [extra_hydra_args...]"
    exit 1
fi

CONFIG_NAME=$1
BARE_CONFIG_NAME="$(basename "${CONFIG_NAME}")"
CONFIG_FILE=""
RESOLVED_CONFIG_PATH="${RLINF_DIR}/config"
RESOLVED_CONFIG_NAME="${BARE_CONFIG_NAME}"
for search_dir in "${RLINF_DIR}/config" "${RLINF_DIR}/examples/embodiment/config"; do
    if [ -f "${search_dir}/${CONFIG_NAME}.yaml" ]; then
        CONFIG_FILE="${search_dir}/${CONFIG_NAME}.yaml"
        RESOLVED_CONFIG_PATH="$(dirname "${CONFIG_FILE}")"
        RESOLVED_CONFIG_NAME="${BARE_CONFIG_NAME}"
        break
    fi
done

# Auto-detect LIBERO_TYPE
if [ -n "$CONFIG_FILE" ] && [ -z "${LIBERO_TYPE:-}" ]; then
    if grep -qP '^\s*-\s*env/libero_pro' "$CONFIG_FILE"; then
        LIBERO_TYPE=pro
    elif grep -qP 'libero_variant:\s*pro|libero_pro_' "$CONFIG_FILE"; then
        LIBERO_TYPE=pro
    elif grep -qP '^\s*-\s*env/libero_plus' "$CONFIG_FILE"; then
        LIBERO_TYPE=plus
    elif grep -qP 'libero_variant:\s*plus|libero_plus_' "$CONFIG_FILE"; then
        LIBERO_TYPE=plus
    fi
fi
if [ -n "${LIBERO_TYPE:-}" ]; then
    export LIBERO_TYPE
    echo "Using LIBERO_TYPE=$LIBERO_TYPE"
fi

# Auto-activate the correct venv
if [[ "${AUTO_SWITCH_VENV:-1}" == "1" ]] && [ -n "$CONFIG_FILE" ]; then
    MODEL_NAME=$(grep -oP '^\s*-\s*model/\K[^@]+' "$CONFIG_FILE" | head -1)
    if [ -z "$MODEL_NAME" ]; then
        # Infer from defaults chain (e.g. /libero_goal_grpo_openpi_pi05@_global_)
        if grep -qP 'openpi|pi0_5|pi05' "$CONFIG_FILE"; then
            MODEL_NAME="pi0_5"
        elif grep -qP 'openvla_oft|openvlaoft' "$CONFIG_FILE"; then
            MODEL_NAME="openvla_oft"
        elif grep -qP 'openvla' "$CONFIG_FILE"; then
            MODEL_NAME="openvla"
        fi
    fi
    case "$MODEL_NAME" in
        openvla_oft)
            if [[ "${LIBERO_TYPE:-}" == "pro" || "${LIBERO_TYPE:-}" == "plus" ]]; then
                VENV_PATH="${RLINF_DIR}/.venv_liberopro_openvlaoft"
            else
                VENV_PATH="${RLINF_DIR}/.venv_openvla_oft"
            fi
            ;;
        openvla)
            if [[ "${LIBERO_TYPE:-}" == "pro" || "${LIBERO_TYPE:-}" == "plus" ]]; then
                VENV_PATH="${RLINF_DIR}/.venv_liberopro_openvla"
            else
                VENV_PATH="${RLINF_DIR}/.venv_openvla_maniskill"
            fi
            ;;
        pi0|pi0_5)
            if [[ "${LIBERO_TYPE:-}" == "pro" || "${LIBERO_TYPE:-}" == "plus" ]]; then
                VENV_PATH="${RLINF_DIR}/.venv_liberopro_openpi"
            else
                VENV_PATH="${RLINF_DIR}/.venv_openpi_maniskill_libero"
            fi
            ;;
    esac
    if [ -n "${VENV_PATH:-}" ] && [ -d "$VENV_PATH" ]; then
        echo "Auto-activating venv for model '${MODEL_NAME}': ${VENV_PATH}"
        source "${VENV_PATH}/bin/activate"
    fi
fi

ROBOT_PLATFORM=${2:-${ROBOT_PLATFORM:-"BRIDGE"}}
EXTRA_HYDRA_ARGS=("${@:3}")

export ROBOT_PLATFORM
echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"
echo "Using Python at $(which python)"

LOG_DIR="${REPO_PATH}/logs/${BARE_CONFIG_NAME}-collect"
mkdir -p "${LOG_DIR}"

CMD=(
    python "${SRC_FILE}"
    --config-path "${RESOLVED_CONFIG_PATH}/"
    --config-name "${RESOLVED_CONFIG_NAME}"
    "hydra.searchpath=[file://${EMBODIED_PATH}/config]"
    "runner.logger.log_path=${LOG_DIR}"
    "++video_collection.output_dir=${LOG_DIR}/videos"
    "++video_collection.num_steps=3"
    "++video_collection.max_videos_per_step=0"
    "++video_collection.video_fps=15"
)
# Extra hydra args — auto-prefix ++ for video_collection keys
for arg in "${@:3}"; do
    if [[ "$arg" == video_collection.* ]]; then
        CMD+=("++${arg}")
    else
        CMD+=("$arg")
    fi
done

MEGA_LOG_FILE="${LOG_DIR}/run_collect_videos.log"
printf '%q ' "${CMD[@]}" > "${MEGA_LOG_FILE}"
echo >> "${MEGA_LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
