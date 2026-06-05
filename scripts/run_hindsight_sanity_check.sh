#!/bin/bash
set -euo pipefail

REPO_ROOT="/PROJECT_ROOT"
cd "${REPO_ROOT}"
export EMBODIED_PATH="${REPO_ROOT}/examples/embodiment"

VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv_maniskill_libero}"
if [ -f "${VENV_PATH}/bin/activate" ]; then
  source "${VENV_PATH}/bin/activate"
fi

echo "[1/3] Generating one rollout video with OpenVLA eval ..."
bash "${REPO_ROOT}/examples/embodiment/eval_embodiment.sh" maniskill_openvla_eval_hindsight_sanity BRIDGE hindsight_sanity
LOG_DIR="$(find "${REPO_ROOT}/logs" -mindepth 1 -maxdepth 1 -type d -name 'hindsight_sanity_*' -print | sort | tail -n 1 || true)"

echo "[2/3] Locating generated rollout videos ..."
mapfile -t VIDEO_PATHS < <(find "${LOG_DIR}/video/eval" -type f -path '*/seed_*/*.mp4' -print | sort)
echo "Found ${#VIDEO_PATHS[@]} videos to relabel."

echo "[3/3] Relabeling with Qwen3-VL ..."
VLLM_JOB_NAME="${VLLM_JOB_NAME:-vllm_qwen3vl}"
VLLM_HOST="${VLLM_HOST:-$(bjobs -J "${VLLM_JOB_NAME}" -noheader -o exec_host 2>/dev/null | awk 'NR==1{print $1}')}"
if [ -z "${VLLM_HOST}" ]; then
  VLLM_HOST="$(bjobs -J "${VLLM_JOB_NAME}" -noheader 2>/dev/null | awk 'NR==1{print $6}')"
fi
if [ -z "${VLLM_HOST}" ]; then
  BASE_URL="${BASE_URL:-http://127.0.0.1:22002/v1}"
else
  BASE_URL="${BASE_URL:-http://${VLLM_HOST}:22002/v1}"
fi
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-235B-A22B-Thinking-FP8}"
echo "Using BASE_URL=${BASE_URL}"
TMP_JSON_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_JSON_DIR}"' EXIT
JSON_FILES=()
IDX=0
for VIDEO_PATH in "${VIDEO_PATHS[@]}"; do
  SEED_NAME="$(basename "$(dirname "${VIDEO_PATH}")")"
  VIDEO_STEM="$(basename "${VIDEO_PATH}" .mp4)"
  CUR_OUTPUT_JSON="${TMP_JSON_DIR}/relabel_${IDX}_${SEED_NAME}_${VIDEO_STEM}.json"
  IDX=$((IDX + 1))
  echo "Relabeling ${VIDEO_PATH}"
  python "${REPO_ROOT}/toolkits/hindsight/relabel_video_with_qwen.py" \
    --video-path "${VIDEO_PATH}" \
    --base-url "${BASE_URL}" \
    --model "${MODEL_NAME}" \
    --enable-thinking \
    --output-json "${CUR_OUTPUT_JSON}"
  JSON_FILES+=("${CUR_OUTPUT_JSON}")
done

OUTPUT_JSON="${OUTPUT_JSON:-${LOG_DIR}/hindsight_relabel_all.json}"
python - "${OUTPUT_JSON}" "${JSON_FILES[@]}" <<'PY'
import json
import sys

output_json = sys.argv[1]
json_files = sys.argv[2:]
items = []
for p in json_files:
    with open(p, "r", encoding="utf-8") as f:
        items.append(json.load(f))
payload = {"num_videos": len(items), "items": items}
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
print(f"Saved merged relabel JSON to {output_json}")
PY

echo "Done. Output json: ${OUTPUT_JSON}"
