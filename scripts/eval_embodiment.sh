#! /bin/bash

export RLINF_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd )"
cd "$RLINF_DIR"
# Default venv; may be overridden below based on model type in config
# source "$RLINF_DIR/.venv_maniskill_libero/bin/activate"

# Load W&B credentials from .env file if it exists
if [ -f "$RLINF_DIR/.env.wandb" ]; then
    echo "Loading W&B config from .env.wandb"
    source "$RLINF_DIR/.env.wandb"
else
    echo "WARNING: .env.wandb not found"
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

# Keep eval entrypoint and defaults rooted in examples/embodiment.
export EMBODIED_PATH="${RLINF_DIR}/examples/embodiment"
export REPO_PATH="${RLINF_DIR}"
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

# Base path to the BEHAVIOR dataset; only required when running the behavior experiment.
export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
# Base path to Isaac Sim; only required when running the behavior experiment.
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

export HYDRA_FULL_ERROR=1
export RAY_DEBUG=legacy
python -m ray stop --force >/dev/null 2>&1 || true

# Flags may appear anywhere; remaining args are positionals in order: [config_name] [robot_platform].
LIBERO_TYPE_ARG=""
MODEL_NAME_ARG=""
EVAL_ROLLOUT_EPOCH=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --libero-type)
            LIBERO_TYPE_ARG="$2"
            shift 2
            ;;
        --libero-type=*)
            LIBERO_TYPE_ARG="${1#*=}"
            shift
            ;;
        --pro)
            LIBERO_TYPE_ARG="pro"
            shift
            ;;
        --perturbation)
            LIBERO_PERTURBATION="$2"
            shift 2
            ;;
        --perturbation=*)
            LIBERO_PERTURBATION="${1#*=}"
            shift
            ;;
        --model)
            MODEL_NAME_ARG="$2"
            shift 2
            ;;
        --model=*)
            MODEL_NAME_ARG="${1#*=}"
            shift
            ;;

        --eval-epochs)
            EVAL_ROLLOUT_EPOCH="$2"
            shift 2
            ;;
        --eval-epochs=*)
            EVAL_ROLLOUT_EPOCH="${1#*=}"
            shift
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

# CLI flag takes precedence over env var
if [ -n "$LIBERO_TYPE_ARG" ]; then
    LIBERO_TYPE="$LIBERO_TYPE_ARG"
fi

if [ -z "${1:-}" ]; then
    CONFIG_NAME="libero_10_grpo_openvlaoft_eval"
else
    CONFIG_NAME=$1
fi

# Resolve config file location. CONFIG_NAME may include a subdirectory prefix.
# Searches config/ then examples/embodiment/config/.
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

if [ -n "${LIBERO_TYPE:-}" ]; then
    export LIBERO_TYPE
    if [ "$LIBERO_TYPE" == "pro" ]; then
        export LIBERO_PERTURBATION="${LIBERO_PERTURBATION:-all}"
        echo "Evaluation Mode: LIBERO-PRO | Perturbation: $LIBERO_PERTURBATION"
    elif [ "$LIBERO_TYPE" == "plus" ]; then
        export LIBERO_SUFFIX="${LIBERO_SUFFIX:-all}"
        echo "Evaluation Mode: LIBERO-PLUS | Suffix: $LIBERO_SUFFIX"
    fi
else
    echo "Evaluation Mode: Standard LIBERO"
fi

# Auto-activate the correct venv based on model type in the config YAML.
# openvla_oft with LIBERO_TYPE=pro uses .venv_liberopro_openvlaoft.
# Set AUTO_SWITCH_VENV=0 to disable.
if [[ "${AUTO_SWITCH_VENV:-1}" == "1" ]]; then
    if [ -n "$CONFIG_FILE" ]; then
        MODEL_NAME=$(grep -oP '^\s*-\s*model/\K[^@]+' "$CONFIG_FILE" | head -1)
        case "$MODEL_NAME" in
            openvla_oft)
                if [[ "${LIBERO_TYPE:-}" == "pro" ]]; then
                    VENV_PATH="${RLINF_DIR}/.venv_liberopro_openvlaoft"
                elif [[ "${LIBERO_TYPE:-}" == "plus" ]]; then
                    VENV_PATH="${RLINF_DIR}/.venv_openvlaoft_liberoplus"
                else
                    VENV_PATH="${RLINF_DIR}/.venv_openvla_oft"
                fi
                ;;
            openvla)
                if [[ "${LIBERO_TYPE:-}" == "pro" || "${LIBERO_TYPE:-}" == "plus" ]]; then
                    VENV_PATH="${RLINF_DIR}/.venv_liberopro_openvla"
                else
                    VENV_PATH="${RLINF_DIR}/.venv_maniskill_libero"
                fi
                ;;
            pi0|pi0_5)
                VENV_PATH="${RLINF_DIR}/.venv_openpi_maniskill_libero"
                ;;
        esac
        if [ -n "${VENV_PATH:-}" ] && [ -d "$VENV_PATH" ]; then
            echo "Auto-activating venv for model '${MODEL_NAME}': ${VENV_PATH}"
            source "${VENV_PATH}/bin/activate"
        fi
    fi
fi

# NOTE: Set the active robot platform (required for correct action dimension and normalization)
ROBOT_PLATFORM=${2:-${ROBOT_PLATFORM:-"LIBERO"}}
EXTRA_HYDRA_ARGS=("${@:3}")

export ROBOT_PLATFORM
echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"
echo "Using Python at $(which python)"

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${BARE_CONFIG_NAME}-eval"
MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"
mkdir -p "${LOG_DIR}"

# Auto-generate experiment name from model, libero type, and perturbation.
# Use model name as the base when provided; fall back to config name.
if [ -n "${MODEL_NAME_ARG}" ]; then
    EXPERIMENT_NAME="${MODEL_NAME_ARG}"
else
    EXPERIMENT_NAME="${BARE_CONFIG_NAME}"
fi
if [ -n "${LIBERO_TYPE:-}" ]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}-${LIBERO_TYPE}"
fi
if [ -n "${LIBERO_PERTURBATION:-}" ]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}-${LIBERO_PERTURBATION}"
fi
echo "Experiment name: ${EXPERIMENT_NAME}"

CMD=(
    python "${SRC_FILE}"
    --config-path "${RESOLVED_CONFIG_PATH}/"
    --config-name "${RESOLVED_CONFIG_NAME}"
    "runner.logger.log_path=${LOG_DIR}"
    "runner.logger.experiment_name=${EXPERIMENT_NAME}"
)
if [ -n "${MODEL_NAME_ARG}" ]; then
    MODEL_PATH="${RLINF_DIR}/models/${MODEL_NAME_ARG}"
    CMD+=("rollout.model.model_path=${MODEL_PATH}")
    CMD+=("actor.model.model_path=${MODEL_PATH}")
    echo "Using model: ${MODEL_PATH}"
fi
if [ -n "$CONFIG_FILE" ]; then
    ORIG_PROJECT=$(grep -E '^\s+project_name:' "$CONFIG_FILE" | head -1 | sed 's/.*project_name:\s*//')
    if [ -n "$ORIG_PROJECT" ]; then
        CMD+=("runner.logger.project_name=eval_${ORIG_PROJECT}")
        echo "W&B project: eval_${ORIG_PROJECT}"
    fi
fi
# LIBERO-Pro/Plus tasks have more objects in the scene, so saved standard init states
# have wrong MuJoCo state dimensions — disable fixed init states for these variants.
if [[ "${LIBERO_TYPE:-}" == "pro" || "${LIBERO_TYPE:-}" == "plus" ]]; then
    # Pro/Plus tasks add extra objects → saved standard init states have wrong MuJoCo
    # state dimensions, so skip set_init_state and do a plain reset instead.
    CMD+=("env.eval.use_fixed_reset_state_ids=False")
    CMD+=("env.eval.use_ordered_reset_state_ids=False")
    # Without set_init_state the gripper position isn't restored from saved state,
    # so explicitly open it during the reset phase.
    CMD+=("env.eval.reset_gripper_open=True")
fi
if [ -n "${EVAL_ROLLOUT_EPOCH}" ]; then
    CMD+=("algorithm.eval_rollout_epoch=${EVAL_ROLLOUT_EPOCH}")
    echo "Using eval_rollout_epoch: ${EVAL_ROLLOUT_EPOCH}"
fi
if [[ ${#EXTRA_HYDRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_HYDRA_ARGS[@]}")
fi

printf '%q ' "${CMD[@]}" > "${MEGA_LOG_FILE}"
echo >> "${MEGA_LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
