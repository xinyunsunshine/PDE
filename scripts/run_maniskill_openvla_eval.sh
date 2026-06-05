#!/bin/bash

cd /PROJECT_ROOT
source .venv_maniskill_libero/bin/activate

# Load W&B credentials from .env file if it exists
if [ -f ".env.wandb" ]; then
    echo "Loading W&B config from .env.wandb"
    source .env.wandb
else
    echo "WARNING: .env.wandb not found"
fi

CACHE_BASE=/PROJECT_ROOT/.cache
mkdir -p "$CACHE_BASE"/{huggingface,transformers,torch,xdg}

export HF_HOME=$CACHE_BASE/huggingface
export HF_HUB_CACHE=$CACHE_BASE/huggingface/hub
export TRANSFORMERS_CACHE=$CACHE_BASE/transformers
export TORCH_HOME=$CACHE_BASE/torch
export XDG_CACHE_HOME=$CACHE_BASE/xdg
export HF_HUB_DISABLE_TELEMETRY=1

# Robot platform for OpenVLA (uses bridge_orig unnorm_key)
export ROBOT_PLATFORM=BRIDGE

bash examples/embodiment/eval_embodiment.sh maniskill_openvla_eval
