#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RLINF_DIR="$( dirname "$SCRIPT_DIR" )"
cd "$RLINF_DIR"

# Keep Ray and training in the same runtime environment.
source "${RLINF_DIR}/.venv_maniskill_libero/bin/activate"

NUM_NODES="${NUM_NODES:-2}"
DONE_FILE="${RLINF_DIR}/.lsf_grpo_${LSB_JOBID:-manual}_done"
RAY_HEAD_IP_FILE="${RLINF_DIR}/ray_utils/ray_head_ip.txt"

# Compute a stable node rank from LSF host allocation order.
# LSB_MCPU_HOSTS is "host1 slots1 host2 slots2 ...".
RANK="$(echo "${LSB_MCPU_HOSTS:-}" | awk -v h="${HOSTNAME}" '
  {
    r=0
    for (i=1; i<=NF; i+=2) {
      if (!seen[$i]++) {
        if ($i==h) { print r; exit }
        r++
      }
    }
  }'
)"

if [ -z "${RANK}" ]; then
  echo "Failed to infer RANK from LSF host allocation."
  echo "HOSTNAME=${HOSTNAME}"
  echo "LSB_MCPU_HOSTS=${LSB_MCPU_HOSTS:-}"
  exit 1
fi

echo "[LSF-ENTRY] HOSTNAME=${HOSTNAME} RANK=${RANK} NUM_NODES=${NUM_NODES}"
echo "[LSF-ENTRY] PYTHON=$(python --version 2>&1)"
echo "[LSF-ENTRY] RAY=$(python -m ray.scripts.scripts --version 2>&1)"

cleanup() {
  # Best-effort cleanup; avoid failing the script on shutdown.
  python -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Ensure stale head IP does not leak across runs.
if [ "${RANK}" = "0" ]; then
  rm -f "$RAY_HEAD_IP_FILE" "$DONE_FILE"
fi

RANK="${RANK}" bash "${RLINF_DIR}/ray_utils/start_ray.sh"

if [ "${RANK}" = "0" ]; then
  REQUIRED_GPUS=$((NUM_NODES * 8))
  bash "${RLINF_DIR}/ray_utils/check_ray.sh" "${REQUIRED_GPUS}"
  bash "${RLINF_DIR}/scripts/run_maniskill_openvla_grpo.sh"
  touch "$DONE_FILE"
else
  # Keep worker task alive until the head finishes training.
  while [ ! -f "$DONE_FILE" ]; do
    sleep 10
  done
fi

