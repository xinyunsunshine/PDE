#!/usr/bin/env bash
# Push the sweep winner (lr=2e-6, kl_beta=1e-3, group_size=32) to bifranka
# as iteration000's trained model, so the inference server picks it up.
#
# Bifranka has Duo 2FA, so this needs to run interactively from your shell
# (automation can't pass Duo prompts). Just:
#
#   bash scripts/push_winner_to_bifranka.sh
#
# The script:
#   1. Verifies the local checkpoint exists + sizes/hashes it.
#   2. scp's it to a .tmp filename on bifranka (atomic-rename safe).
#   3. Renames .tmp → final filename in one mv.
#   4. md5sums the remote file and prints both hashes side-by-side.
#
# Destination matches what realworld_embodied_runner.py would have written
# if `bifranka_dest` had been set on the sweep run:
#   user@bifranka:/home/user/real_rl_runs_her/meta_green_bowl_baseline_gs32_nl01/iteration000/train_after_itr0_model.pt
set -euo pipefail

REPO=/PROJECT_ROOT/repo
WINNER=realworld_grpo_gs32_kl1em3_lr2em6
SRC="${REPO}/logs/realworld_grpo_lr_kl_sweep_0515/${WINNER}/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"

DST_HOST=user@robot.example.com
DST_DIR=/home/user/real_rl_runs_her/meta_green_bowl_baseline_gs32_nl01/iteration000
DST_FINAL="${DST_DIR}/train_after_itr0_model.pt"
DST_TMP="${DST_FINAL}.tmp"

SCP_CIPHER=aes128-gcm@openssh.com

# --- Local sanity ---------------------------------------------------------
if [[ ! -f "${SRC}" ]]; then
  echo "ERROR: local checkpoint not found at ${SRC}" >&2
  exit 1
fi

SRC_SIZE=$(stat -c %s "${SRC}")
SRC_SIZE_H=$(numfmt --to=iec "${SRC_SIZE}")
echo "Source:      ${SRC}"
echo "Source size: ${SRC_SIZE_H} (${SRC_SIZE} bytes)"
echo "Computing local md5..."
SRC_MD5=$(md5sum "${SRC}" | awk '{print $1}')
echo "Source md5:  ${SRC_MD5}"
echo

echo "Destination: ${DST_HOST}:${DST_FINAL}"
echo

# --- Upload to .tmp -------------------------------------------------------
echo "[1/3] scp -> ${DST_TMP}"
scp -c "${SCP_CIPHER}" -o ConnectTimeout=30 "${SRC}" "${DST_HOST}:${DST_TMP}"

# --- Verify size on remote ------------------------------------------------
echo "[2/3] verifying remote size"
REMOTE_SIZE=$(ssh -o ConnectTimeout=30 "${DST_HOST}" "stat -c %s ${DST_TMP}")
if [[ "${REMOTE_SIZE}" != "${SRC_SIZE}" ]]; then
  echo "ERROR: remote size ${REMOTE_SIZE} != local ${SRC_SIZE}; leaving .tmp in place for inspection" >&2
  exit 2
fi
echo "ok (${REMOTE_SIZE} bytes)"

# --- Atomic rename + md5 --------------------------------------------------
echo "[3/3] mkdir -p, atomic rename, md5sum on remote"
ssh -o ConnectTimeout=30 "${DST_HOST}" \
  "mkdir -p ${DST_DIR} && mv ${DST_TMP} ${DST_FINAL} && md5sum ${DST_FINAL}"

echo
echo "Push complete."
echo "  local md5:  ${SRC_MD5}"
echo "  expected:   ${SRC_MD5}  ${DST_FINAL}"
echo "  (compare against the remote md5 printed above)"
