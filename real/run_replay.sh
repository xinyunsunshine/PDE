#!/usr/bin/env bash
# Replay a LeRobot episode through the same control path deploy.py uses.
# Useful for (a) sanity-checking that the controller + env can reproduce
# recorded teleop, and (b) verifying that the VLA server's normalization
# chain is self-inverse (round_trip mode).
#
# Requires:
#   - OSC server is running (real/run_osc_server.sh). No VLA server needed.
#   - For round_trip mode, a checkpoint dir with <asset_id>/norm_stats.json.
#     Model weights are NOT loaded.
#
# Usage:
#   real/run_replay.sh \
#     dataset.root=/abs/path/to/lerobot_datasets \
#     dataset.repo_id=pickcube_v1 \
#     inference.checkpoint=/abs/path/to/ckpt_stage_dir \
#     [replay.mode=round_trip|raw] \
#     [replay.episode_idx=0]
#
# Env vars:
#   OSC_ZMQ_URL  (default: tcp://localhost:5555)
#   REPLAY_MODE  (default: round_trip) — overridden by replay.mode=... on CLI.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

ZMQ_URL=${OSC_ZMQ_URL:-tcp://localhost:5555}
MODE=${REPLAY_MODE:-round_trip}

echo "[run_replay] expecting OSC server at ${ZMQ_URL} — start it with real/run_osc_server.sh if not running."
echo "[run_replay] default mode=${MODE}; override with replay.mode=... on the CLI."

exec python -m real.replay_episode \
    zmq.url="$ZMQ_URL" \
    replay.mode="$MODE" \
    "$@"
