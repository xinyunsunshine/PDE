import os

from real.round_task import base_dir as _round_task_base_dir

# Single source of truth for the prompt-opt data tree. Defined in real.round_task
# so deploy.py and the MCP server agree on the layout.
BASE_DIR = _round_task_base_dir()

NUM_FRAMES = 16
FRAME_SIZE: tuple[int, int] | None = None  # None = keep original resolution

# How the client (e.g. MacBook) reaches this robot via SSH
ROBOT_SSH_HOST = os.environ.get("ROBOT_SSH_HOST", "user@robot.example.com")

# Where the client wants synced files to land (mirror of BASE_DIR layout).
CLIENT_SYNC_PATH = os.environ.get("CLIENT_SYNC_PATH", "~/real_prompt_opt")

MCP_PORT = int(os.environ.get("MCP_PORT", "8766"))
