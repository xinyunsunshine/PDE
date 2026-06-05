import json
from datetime import datetime
from pathlib import Path

from real.mcp_server.config import CLIENT_SYNC_PATH, ROBOT_SSH_HOST
from real.round_task import (
    base_dir,
    iter_prompt_dirs,
    processed_dir,
    round_dir,
)


def is_same_machine(client_path: str = CLIENT_SYNC_PATH, base: Path | None = None) -> bool:
    """True when the agent's view of the data tree is the robot's view.

    On a single-machine setup (A client running on the same workstation that
    records videos), `BASE_DIR` and `CLIENT_SYNC_PATH` resolve to the same
    directory, so rsync has nothing to do.
    """
    try:
        local = (base or base_dir()).expanduser().resolve()
        remote = Path(client_path).expanduser().resolve()
    except OSError:
        return False
    return local == remote


def _client_round_path(task_id: int, round_num: int, client_path: str = CLIENT_SYNC_PATH) -> str:
    return f"{client_path}/task{task_id}/round{round_num}"


def build_round_sync_command(
    task_id: int,
    round_num: int,
    robot_host: str = ROBOT_SSH_HOST,
    client_path: str = CLIENT_SYNC_PATH,
) -> str:
    """rsync command to pull a whole round dir (all prompts) onto the client."""
    rd = round_dir(task_id, round_num)
    source = f"{robot_host}:{rd}/"
    dest = f"{_client_round_path(task_id, round_num, client_path)}/"
    return f"rsync -avz {source} {dest}"


def mark_round_synced(
    task_id: int,
    round_num: int,
    client_path: str = CLIENT_SYNC_PATH,
) -> list[Path]:
    """Flip `synced=true` on every per-prompt manifest under this round."""
    updated: list[Path] = []
    now = datetime.now().isoformat()
    client_round = _client_round_path(task_id, round_num, client_path)
    for pd in iter_prompt_dirs(task_id, round_num):
        manifest_file = processed_dir(pd) / "manifest.json"
        if not manifest_file.exists():
            continue
        with open(manifest_file) as f:
            manifest = json.load(f)
        manifest["synced"] = True
        manifest["last_sync"] = now
        manifest["client_path"] = f"{client_round}/{pd.name}"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)
        updated.append(manifest_file)
    return updated


def get_local_episode_paths(
    task_id: int,
    round_num: int,
    prompt_idx: int,
    client_path: str = CLIENT_SYNC_PATH,
    base: Path | None = None,
) -> dict[str, dict]:
    """Return a `{episode_name: {frames, prompt, label, client_dir}}` map.

    Paths are rewritten to the client side so the consuming process (e.g. the
    the cluster session after rsync) can read them directly.
    """
    from real.round_task import prompt_dir as _prompt_dir

    pd = _prompt_dir(task_id, round_num, prompt_idx, base or base_dir())
    proc = processed_dir(pd)
    client_round = _client_round_path(task_id, round_num, client_path)
    client_prompt = f"{client_round}/{pd.name}/processed"
    episodes: dict[str, dict] = {}

    if not proc.exists():
        return episodes

    for ep_dir in sorted(proc.iterdir()):
        if not ep_dir.is_dir() or not ep_dir.name.startswith("ep"):
            continue
        frames = sorted(ep_dir.glob("frame_*.png"))
        prompt_file = ep_dir / "prompt.txt"
        prompt = prompt_file.read_text().strip() if prompt_file.exists() else ""

        label = "unknown"
        meta_file = ep_dir / "metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                ep_meta = json.load(f)
            label = ep_meta.get("label", "unknown")

        episodes[ep_dir.name] = {
            "frames": [f"{client_prompt}/{ep_dir.name}/{f.name}" for f in frames],
            "prompt": prompt,
            "label": label,
            "success": 1 if label == "success" else 0,
            "client_dir": f"{client_prompt}/{ep_dir.name}",
        }

    return episodes
