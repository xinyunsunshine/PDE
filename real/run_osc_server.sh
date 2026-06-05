#!/usr/bin/env bash
# Launch the OSC robot server. Owns the in-process aiofranka FrankaController
# and the 1 kHz libfranka loop; exposes set_ee_pose / get_ee_pose / reset over
# a ZMQ REP socket (default tcp://localhost:5555).
#
# This matches the mdpo server pattern (real/server/osc_server.py): a single
# Python process with one control loop, no multiprocessing subprocess, so
# there is no extra thread/process jitter on CPU 31 beyond libfranka's own.
#
# Usage:
#   real/run_osc_server.sh
#
# Env vars:
#   OSC_ZMQ_URL     (default: tcp://localhost:5555)
#
# Prereqs:
#   - FR3 FCI must be unlocked (Desk UI → unlock brakes).
#   - Run inside the conda env that has the in-process aiofranka
#     (FrankaController, NOT FrankaRemoteController).
#
# Then in a separate terminal, run real/run_deploy.sh or real/collect_sft_data.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

ZMQ_URL=${OSC_ZMQ_URL:-tcp://localhost:5555}

echo "[run_osc_server] launching OSC server at ${ZMQ_URL}"
echo "[run_osc_server] make sure FR3 FCI is unlocked (Desk UI) before proceeding."

exec python -m real.server.osc_server \
    zmq.url="$ZMQ_URL"
