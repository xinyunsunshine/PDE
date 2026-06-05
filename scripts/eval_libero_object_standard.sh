#!/bin/bash
# Evaluate OpenVLA-OFT on standard LIBERO-object (10 tasks, no perturbations).
#
# Usage:
#   bash scripts/eval_libero_object_standard.sh
#
# Optional env overrides:
#   MODEL_PATH          - path to checkpoint (default below)
#   NUM_TRIALS_PER_TASK - rollouts per task (default: 3, use 50 for full eval)

set -euo pipefail

RLINF_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd )"
cd "$RLINF_DIR"

# source "$RLINF_DIR/.venv_openvla_oft/bin/activate"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export VULKAN_SDK=/PROJECT_ROOT/vulkan/1.4.341.1/x86_64
export PATH=$VULKAN_SDK/bin:$PATH
export LD_LIBRARY_PATH=$VULKAN_SDK/lib:$LD_LIBRARY_PATH
export VK_LAYER_PATH=$VULKAN_SDK/share/vulkan/explicit_layer.d
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json

EVAL_SCRIPT="$RLINF_DIR/.venv_libero/lib/python3.11/site-packages/experiments/robot/libero/run_libero_eval.py"
MODEL_PATH="${MODEL_PATH:-/PROJECT_ROOT/models/Openvla-oft-SFT-libero-object-traj1}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-3}"
LOG_DIR="$RLINF_DIR/logs/libero_object_standard_eval_$(date +'%Y%m%d-%H%M%S')"
mkdir -p "$LOG_DIR"

echo "Model:              $MODEL_PATH"
echo "Trials per task:    $NUM_TRIALS_PER_TASK"
echo "Logs:               $LOG_DIR"

python "$EVAL_SCRIPT" \
    --pretrained_checkpoint "$MODEL_PATH" \
    --task_suite_name libero_object \
    --num_trials_per_task "$NUM_TRIALS_PER_TASK" \
    --use_l1_regression True \
    --use_proprio True \
    --num_images_in_input 2 \
    --center_crop True \
    --local_log_dir "$LOG_DIR" \
    2>&1 | tee "$LOG_DIR/eval.log"
