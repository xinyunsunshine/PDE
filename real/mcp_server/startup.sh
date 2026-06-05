#!/usr/bin/env bash
# Start the MCP server on the robot workstation.
#
# Usage (on robot):
#   ./startup.sh [port]
#
# Example:
#   ./startup.sh 8766
#
# Then on the cluster, open a forward tunnel:
#   ssh -L 8766:localhost:8766 user@robot.example.com
#
# A remote client connects to http://localhost:8766/sse
# which forwards through the tunnel to the robot's MCP server.

set -euo pipefail

PORT="${1:-8766}"

export MCP_PORT="${PORT}"

echo "[startup] Starting MCP server on port ${PORT}..."
echo "[startup] On the cluster, run:"
echo "    ssh -N -L ${PORT}:localhost:${PORT} user@robot.example.com"
echo ""

cd "$(dirname "$0")/../.."
python -m real.mcp_server.server --port "${PORT}"
