#!/bin/bash
set -euo pipefail

REPO_ROOT="/PROJECT_ROOT"
cd "${REPO_ROOT}"

VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv_maniskill_libero}"
LOG_DIR="${LOG_DIR:-$(find "${REPO_ROOT}/logs" -mindepth 1 -maxdepth 1 -type d -name 'hindsight_sanity_*' -print | sort | tail -n 1 || true)}"
if [ -f "${VENV_PATH}/bin/activate" ]; then
  source "${VENV_PATH}/bin/activate"
fi

if [ -n "${VIDEO_PATH:-}" ]; then
  mapfile -t VIDEO_PATHS < <(printf '%s\n' "${VIDEO_PATH}")
else
  mapfile -t VIDEO_PATHS < <(find "${LOG_DIR}/video/eval" -type f -path '*/seed_*/*.mp4' -print | sort)
fi
if [ "${#VIDEO_PATHS[@]}" -eq 0 ]; then
  echo "No videos found for relabeling under ${LOG_DIR}/video/eval/seed_*"
  exit 1
fi

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
OUTPUT_JSON="${OUTPUT_JSON:-${LOG_DIR}/hindsight_relabel_all.json}"

echo "Using LOG_DIR=${LOG_DIR}"
echo "Found ${#VIDEO_PATHS[@]} videos to relabel."
echo "Using BASE_URL=${BASE_URL}"
echo "Merged output: ${OUTPUT_JSON}"

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
