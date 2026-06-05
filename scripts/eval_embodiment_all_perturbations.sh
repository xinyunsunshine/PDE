#! /bin/bash
# Run eval across multiple perturbations and log all results to a single W&B run.
# Each perturbation's metrics are logged under a separate prefix (eval_task/, eval_swap/, etc.)
#
# Usage mirrors eval_embodiment.sh but without --perturbation:
#   bash scripts/eval_embodiment_all_perturbations.sh \
#       libero_object/openvlaoft/libero_object_openvlaoft_inplace_her_sym_rits LIBERO \
#       --pro --eval-epochs 6 --model Openvla-oft-her-sym-object
#
# Override the perturbation list with --perturbations "task swap object lan"
#
# Note: libero_spatial does not have a swap perturbation dataset. The default
# perturbation list excludes swap when the config path contains "spatial".

export RLINF_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd )"

PERTURBATIONS=""  # resolved below after arg parsing
PASS_THROUGH=()
MODEL_NAME_ARG=""
EXPLICIT_PERTURBATIONS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --perturbations)
            EXPLICIT_PERTURBATIONS="$2"; shift 2 ;;
        --perturbations=*)
            EXPLICIT_PERTURBATIONS="${1#*=}"; shift ;;
        --model)
            MODEL_NAME_ARG="$2"; PASS_THROUGH+=("$1" "$2"); shift 2 ;;
        --model=*)
            MODEL_NAME_ARG="${1#*=}"; PASS_THROUGH+=("$1"); shift ;;
        *)
            PASS_THROUGH+=("$1"); shift ;;
    esac
done

# Resolve perturbation list. spatial suites have no swap dataset; default excludes it.
if [[ -n "$EXPLICIT_PERTURBATIONS" ]]; then
    PERTURBATIONS="$EXPLICIT_PERTURBATIONS"
elif [[ "${PASS_THROUGH[*]}" == *spatial* ]]; then
    PERTURBATIONS="task object lan"
    echo "Note: 'swap' excluded — libero_spatial has no swap perturbation dataset."
else
    PERTURBATIONS="task swap object lan"
fi

# Generate a unique W&B run ID shared across all perturbation evals
WANDB_RUN_ID=$(tr -dc 'a-z0-9' < /dev/urandom | head -c 8)
# Use model name as the W&B run name (no perturbation suffix)
EXPERIMENT_NAME="${MODEL_NAME_ARG}-pro"
echo "W&B run ID: ${WANDB_RUN_ID}"
echo "W&B experiment name: ${EXPERIMENT_NAME}"
echo "Perturbations: ${PERTURBATIONS}"

for perturbation in $PERTURBATIONS; do
    echo ""
    echo "========================================"
    echo "Running perturbation: ${perturbation}"
    echo "========================================"
    bash "${RLINF_DIR}/scripts/eval_embodiment.sh" \
        "${PASS_THROUGH[@]}" \
        --perturbation "$perturbation" \
        "+runner.logger.wandb_run_id=${WANDB_RUN_ID}" \
        "+runner.logger.metric_prefix=eval_${perturbation}" \
        "runner.logger.experiment_name=${EXPERIMENT_NAME}"
done
