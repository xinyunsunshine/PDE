#! /bin/bash

export RLINF_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd )"
cd "$RLINF_DIR"
# Default venv; may be overridden below based on model type in config
source "$RLINF_DIR/.venv_maniskill_libero/bin/activate"

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

xport startTime=$(date +%s) #mark the start of job

export JOB_ID=${LSB_JOBID}

export MASTER_ADDR=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | head -n 1 | grep -f- /etc/hosts | head -1 | cut -d' ' -f1)
export MASTER_PORT=28442

export NNODES=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | wc -w)

export LOCAL_NODE=$(echo "${LSB_MCPU_HOSTS}" | tr ' ' '\n' | sed 'n; d' | grep -m1  ${HOSTNAME} | grep -f- /etc/hosts | head -1 | cut -d' ' -f1)

export NODE_RANK=$(($(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | grep -n -m1 $HOSTNAME | cut -d':' -f1)-1))

export NCCL_IB_HCA="^=mlx5_1,mlx5_6"
export NCCL_SOCKET_IFNAME="=ibp26s0,ibp60s0,ibp77s0,ibp94s0,ibp156s0,ibp188s0,ibp204s0,ibp220s0"

export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_IB_SPLIT_DATA_ON_QPS=0
export NCCL_IB_TIMEOUT=30
export NCCL_IB_RETRY_CNT=20
export NCCL_CROSS_NIC=2
export NCCL_IB_DISABLE=0

export HOME=/PROJECT_ROOT

export TMPDIR=/tmp/job

export LOCAL_DIR=/tmp/job
export HF_HOME=$LOCAL_DIR
export HF_DATASETS_CACHE=$LOCAL_DIR/.cache
export VLLM_CACHE_ROOT=$LOCAL_DIR/.cache
export XDG_CACHE_HOME=$LOCAL_DIR/.cache
export XDG_CONFIG_HOME=$LOCAL_DIR/

export runtime=$(date "+%Y.%m.%d-%H.%M")
export TRITON_HOME=$LOCAL_DIR/.cache
export TRITON_CACHE_DIR=$LOCAL_DIR/.cache
export RAY_TMPDIR=$TMPDIR/$JOB_ID

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NNODES: $NNODES"


echo "JOB ID = ${JOB_ID}"
echo "Run Dir: ${run_dir} "
echo "Local node name is: ${LOCAL_NODE}"

port=$MASTER_PORT

if [ -z "$1" ]; then
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
        RESOLVED_CONFIG_PATH="$(dirname "${CONFIG_FILE}")"
        RESOLVED_CONFIG_NAME="${BARE_CONFIG_NAME}"
        break
    fi
done

# Auto-activate the correct venv based on model type in the config YAML.
# Looks for the "- model/<name>@actor.model" line in defaults.
# Set AUTO_SWITCH_VENV=0 to disable.
if [[ "${AUTO_SWITCH_VENV:-1}" == "1" ]]; then
    if [ -n "$CONFIG_FILE" ]; then
        MODEL_NAME=$(grep -oP '^\s*-\s*model/\K[^@]+' "$CONFIG_FILE" | head -1)
        case "$MODEL_NAME" in
            openvla_oft)
                VENV_PATH="${RLINF_DIR}/.venv_openvla_oft"
                ;;
            openvla)
                VENV_PATH="${RLINF_DIR}/.venv_maniskill_libero"
                ;;
            pi0|pi0_5)
                VENV_PATH="${RLINF_DIR}/.venv_openpi_libero_maniskill"
                ;;
        esac
        if [ -n "${VENV_PATH:-}" ] && [ -d "$VENV_PATH" ]; then
            echo "Auto-activating venv for model '${MODEL_NAME}': ${VENV_PATH}"
            source "${VENV_PATH}/bin/activate"
        fi
    fi
fi

# Ensure Ray dashboard/state API is available for actor health checks.
# export RAY_DEBUG=legacy  # disabled: causes remote PDB sessions on errors
python -m ray stop --force >/dev/null 2>&1 || true

# Bootstrap Ray cluster.
# Head node (NODE_RANK=0) starts the Ray head; workers connect to it.
# Workers sleep indefinitely after joining — the head drives all training.
# Venv must be activated before ray start so workers inherit the right packages.
RAY_CLI="python -m ray.scripts.scripts"
if [ "${NODE_RANK}" -eq 0 ]; then
    echo "NODE_RANK=0: starting Ray head on ${MASTER_ADDR}:${MASTER_PORT}"
    ${RAY_CLI} start --head --port=${MASTER_PORT} --node-ip-address=${MASTER_ADDR} --num-gpus=8
    sleep 5
else
    echo "NODE_RANK=${NODE_RANK}: waiting for Ray head at ${MASTER_ADDR}:${MASTER_PORT}..."
    sleep 15
    ${RAY_CLI} start --address=${MASTER_ADDR}:${MASTER_PORT} --num-gpus=8
    echo "Ray worker started. Waiting for job to complete..."
    sleep infinity
fi

# NOTE: Set the active robot platform (required for correct action dimension and normalization), supported platforms are LIBERO, ALOHA, BRIDGE, default is LIBERO
ROBOT_PLATFORM=${2:-${ROBOT_PLATFORM:-"BRIDGE"}}

export ROBOT_PLATFORM
echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${BARE_CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${RESOLVED_CONFIG_PATH}/ --config-name ${RESOLVED_CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
