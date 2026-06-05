#! /bin/bash

export RLINF_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd )"
cd "$RLINF_DIR"
# Default venv; may be overridden below based on model type in config
source "$RLINF_DIR/.venv_libero_openvlaoft/bin/activate"

# Load W&B credentials from .env file if it exists
if [ -f "$RLINF_DIR/.env.wandb" ]; then
    echo "Loading W&B config from .env.wandb"
    source "$RLINF_DIR/.env.wandb"
else
    echo "WARNING: .env.wandb not found"
fi

# Load API keys (RITS_API_KEY, OPENAI_API_KEY, etc.) from .env if it exists
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

# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Robot platform for OpenVLA (uses bridge_orig unnorm_key)
export ROBOT_PLATFORM=BRIDGE

# Vulkan SDK setup for ManiSkill rendering
export VULKAN_SDK=/PROJECT_ROOT/vulkan/1.4.341.1/x86_64
export PATH=$VULKAN_SDK/bin:$PATH
export LD_LIBRARY_PATH=$VULKAN_SDK/lib:$LD_LIBRARY_PATH
export VK_LAYER_PATH=$VULKAN_SDK/share/vulkan/explicit_layer.d
# SAPIEN checks VK_ICD_FILENAMES; system typically has nvidia_icd.x86_64.json
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json}
export VK_DRIVER_FILES=$VK_ICD_FILENAMES


# Keep training entrypoint and defaults rooted in examples/embodiment.
export EMBODIED_PATH="${RLINF_DIR}/examples/embodiment"
export REPO_PATH="${RLINF_DIR}"
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

# Base path to the BEHAVIOR dataset, which is the BEHAVIOR-1k repo's dataset folder
# Only required when running the behavior experiment.
export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
# Base path to Isaac Sim, only required when running the behavior experiment.
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

# Ensure Ray dashboard/state API is available for actor health checks.
export RAY_DEBUG=legacy
python -m ray stop --force >/dev/null 2>&1 || true

# Resume from latest checkpoint for this config by default. Use --no-resume / --fresh / -n
# to start a new run (new log dir, no runner.resume_dir). Use --no-timestamp-prefix to
# create fresh runs under logs/<config_name> (instead of logs/<timestamp>-<config_name>).
# Flags may appear anywhere; remaining args are positionals in order: [config_name] [robot_platform].
RESUME_FROM_LAST=1
USE_TIMESTAMP_PREFIX=1
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-resume|--fresh|-n)
            RESUME_FROM_LAST=0
            shift
            ;;
        --resume|-r)
            RESUME_FROM_LAST=1
            shift
            ;;
        --no-timestamp-prefix|--no-timestamp)
            USE_TIMESTAMP_PREFIX=0
            shift
            ;;
        --timestamp-prefix|--timestamp)
            USE_TIMESTAMP_PREFIX=1
            shift
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

if [ -z "${1:-}" ]; then
    CONFIG_NAME="maniskill_grpo_openvla"
else
    CONFIG_NAME=$1
fi

# Resolve config file location. CONFIG_NAME may include a subdirectory prefix
# (e.g. "maniskill_configs/maniskill_openvla_grpo"). Searches config/ then
# examples/embodiment/config/. Sets CONFIG_FILE, RESOLVED_CONFIG_PATH,
# RESOLVED_CONFIG_NAME (bare filename, no subdir) for Hydra invocation.
BARE_CONFIG_NAME="$(basename "${CONFIG_NAME}")"
CONFIG_FILE=""
RESOLVED_CONFIG_PATH="${RLINF_DIR}/config"
RESOLVED_CONFIG_NAME="${BARE_CONFIG_NAME}"
for search_dir in "${RLINF_DIR}/config" "${RLINF_DIR}/examples/embodiment/config"; do
    if [ -f "${search_dir}/${CONFIG_NAME}.yaml" ]; then
        CONFIG_FILE="${search_dir}/${CONFIG_NAME}.yaml"
        RESOLVED_CONFIG_PATH="${search_dir}"
        RESOLVED_CONFIG_NAME="${CONFIG_NAME}"
        break
    fi
done

# Auto-detect LIBERO_TYPE and model from the config path.
# Config directory convention: config/{suite}/{model}/{config_name}.yaml
# Extract suite (1st component) and model (2nd component) from CONFIG_NAME.
PATH_SUITE=$(echo "$CONFIG_NAME" | awk -F/ '{if(NF>=2) print $1}')
PATH_MODEL=$(echo "$CONFIG_NAME" | awk -F/ '{if(NF>=2) print $(NF-1)}')

# LIBERO_TYPE: pro/plus variants use different benchmark packages.
# Explicit LIBERO_TYPE env var takes precedence.
if [ -z "${LIBERO_TYPE:-}" ]; then
    case "$PATH_SUITE" in
        libero_pro_*) LIBERO_TYPE=pro ;;
        libero_plus_*) LIBERO_TYPE=plus ;;
    esac
fi
if [ -n "${LIBERO_TYPE:-}" ]; then
    export LIBERO_TYPE
    echo "Using LIBERO_TYPE=$LIBERO_TYPE"
fi

# Standard (non-pro, non-plus) libero detection
IS_LIBERO_STD=0
if [[ -z "${LIBERO_TYPE:-}" ]] && [[ "${PATH_SUITE}" == libero_* ]]; then
    IS_LIBERO_STD=1
fi

# Auto-activate the correct venv based on model type and environment in the config YAML.
# Looks for the "- model/<name>@actor.model" line in defaults.
# Venv mapping (model x env):
#   openvla_oft + pro/plus -> .venv_liberopro_openvlaoft  (libero, liberopro, openvla-oft/prismatic)
#   openvla_oft            -> .venv_openvla_oft
#   openvla     + pro/plus -> .venv_liberopro_openvla     (libero, liberopro, openvla/prismatic)
#   openvla                -> .venv_openvla_maniskill
#   pi0/pi0_5   + pro/plus -> .venv_liberopro_openpi
#   pi0/pi0_5              -> .venv_openpi_maniskill_libero
# Set AUTO_SWITCH_VENV=0 to disable.
if [[ "${AUTO_SWITCH_VENV:-1}" == "1" ]]; then
    if [ -n "$CONFIG_FILE" ]; then
        # Model from config path (2nd directory component), mapped to model config names
        case "$PATH_MODEL" in
            openvlaoft) MODEL_NAME=openvla_oft ;;
            openvla|openvla_warmup) MODEL_NAME=openvla ;;
            pi05) MODEL_NAME=pi0_5 ;;
            gr00t) MODEL_NAME=gr00t ;;
            *) MODEL_NAME="$PATH_MODEL" ;;
        esac
        case "$MODEL_NAME" in
            openvla_oft)
                if [[ "${LIBERO_TYPE:-}" == "pro" || "${LIBERO_TYPE:-}" == "plus" ]]; then
                    VENV_PATH="${RLINF_DIR}/.venv_liberopro_openvlaoft"
                elif [[ "${IS_LIBERO_STD}" == "1" ]]; then
                    VENV_PATH="${RLINF_DIR}/.venv_libero_openvlaoft"
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
                elif [ -d "${RLINF_DIR}/.venv_openpi_maniskill_libero" ]; then
                    VENV_PATH="${RLINF_DIR}/.venv_openpi_maniskill_libero"
                else
                    VENV_PATH="${RLINF_DIR}/.venv_liberopro_openpi"
                fi
                ;;
            gr00t)
                if [[ "${LIBERO_TYPE:-}" == "pro" || "${LIBERO_TYPE:-}" == "plus" ]]; then
                    VENV_PATH="${RLINF_DIR}/.venv_liberopro_gr00t"
                else
                    VENV_PATH="${RLINF_DIR}/.venv_gr00t_maniskill_libero"
                fi
                ;;
        esac
        if [ -n "${VENV_PATH:-}" ] && [ -d "$VENV_PATH" ]; then
            echo "Auto-activating venv for model '${MODEL_NAME}': ${VENV_PATH}"
            source "${VENV_PATH}/bin/activate"
        fi
    fi
fi

# NOTE: Set the active robot platform (required for correct action dimension and normalization), supported platforms are LIBERO, ALOHA, BRIDGE, default is LIBERO
ROBOT_PLATFORM=${2:-${ROBOT_PLATFORM:-"BRIDGE"}}
EXTRA_HYDRA_ARGS=("${@:3}")

export ROBOT_PLATFORM
if [[ -n "$CONFIG_FILE" ]]; then
    TRAIN_DATASET=$(grep -oP '^\s*-\s*env/\K[^@]+(?=@env\.train)' "$CONFIG_FILE" | head -1 \
        | sed 's/_/ /g')
    echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM | dataset: ${TRAIN_DATASET:-unknown}"
else
    echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"
fi

echo "Using Python at $(which python)"

# Auto-detect latest checkpoint from a previous run of the same config (unless --no-resume).
# Supports both logs/*-${BARE_CONFIG_NAME}/ and logs/${BARE_CONFIG_NAME}/.
LATEST_CKPT=""
PREV_LOG_DIR=""
if [[ "${RESUME_FROM_LAST}" == "1" ]]; then
    for dir in "${REPO_PATH}/logs/${BARE_CONFIG_NAME}" "${REPO_PATH}/logs/"*"-${BARE_CONFIG_NAME}"; do
        [ -d "$dir" ] || continue
        ckpt=$(ls -d "${dir}/"*"/checkpoints/global_step_"* 2>/dev/null | sort -V | tail -1)
        if [ -n "$ckpt" ] && [ -d "$ckpt" ]; then
            LATEST_CKPT="$ckpt"
            PREV_LOG_DIR="$dir"
        fi
    done
fi

if [[ "${RESUME_FROM_LAST}" == "1" ]] && [ -n "$LATEST_CKPT" ]; then
    LOG_DIR="$PREV_LOG_DIR"
    RESUME_ARG="runner.resume_dir=${LATEST_CKPT}"
    echo "Resuming from checkpoint: ${LATEST_CKPT}"
else
    if [[ "${RESUME_FROM_LAST}" == "0" ]]; then
        echo "Starting fresh (--no-resume): not loading a previous checkpoint."
    fi
    if [[ "${USE_TIMESTAMP_PREFIX}" == "1" ]]; then
        LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${BARE_CONFIG_NAME}"
    else
        LOG_DIR="${REPO_PATH}/logs/${BARE_CONFIG_NAME}"
        echo "Timestamp prefix disabled: using LOG_DIR=${LOG_DIR}"
    fi
    RESUME_ARG=""
fi

MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD=(
    python "${SRC_FILE}"
    --config-path "${RESOLVED_CONFIG_PATH}/"
    --config-name "${RESOLVED_CONFIG_NAME}"
    "hydra.searchpath=[file://${EMBODIED_PATH}/config]"
    "runner.logger.log_path=${LOG_DIR}"
)
if [[ -n "${RESUME_ARG}" ]]; then
    CMD+=("${RESUME_ARG}")
fi
if [[ ${#EXTRA_HYDRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_HYDRA_ARGS[@]}")
fi

printf '%q ' "${CMD[@]}" > "${MEGA_LOG_FILE}"
echo >> "${MEGA_LOG_FILE}"

python keep_gpus_warm.py &

"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"

kill 0
