"""Replay a LeRobot episode on the FR3 + AgileX rig.

Picks one episode from a dataset produced by ``real/collect_sft_data.py`` and
plays its stored 10-d absolute actions through ``env.step()`` — the exact
same control path ``real/deploy.py`` uses. Two modes:

* ``raw`` — execute the recorded actions verbatim. Baseline sanity check
  that the controller + env + robot can reproduce recorded teleop at all.
  No transforms, no checkpoint needed.
* ``round_trip`` — for each action, apply the VLA server's input transforms
  (``FR3VLAInputs`` + ``DeltaActions`` + ``Normalize``) and then immediately
  their inverses (``Unnormalize`` + ``AbsoluteActions`` + ``FR3VLAOutputs``).
  If the chain is self-inverse, the round-tripped action equals the
  original and the robot moves identically to raw replay. If not, the
  per-dim numerical diff (logged offline before execution) and the robot
  divergence (visible at runtime) localize the bug.

Note that ``real/deploy.py`` itself does no normalization — the server's
openpi ``Policy`` owns all transforms. Verifying the round trip here
exercises exactly the chain the server applies.

Prereqs (same as deploy):
  * OSC server is up (``real/run_osc_server.sh``).
  * For ``round_trip`` mode, a checkpoint dir with ``<asset_id>/norm_stats.json``
    (the same one ``real/run_vla_server.sh`` would serve). Model weights
    are NOT loaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig

from real.env import PRIMARY_CAMERA_NAME, FR3RealEnv

# LeRobot v3 renamed the module path (``lerobot.datasets``); v2 kept it under
# ``lerobot.common.datasets``. Try v3 first since the datasets this repo
# currently produces are v3, fall back to v2 if the installed package is older.
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
except ModuleNotFoundError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore


_GLOBAL_IMAGE_KEY = f"observation.images.{PRIMARY_CAMERA_NAME}"
_ACTION_KEY = "action"
_STATE_KEY = "observation.state"


def _prompt_episode_idx(total: int, prefilled: int | None) -> int:
    if prefilled is not None:
        idx = int(prefilled)
        if not 0 <= idx < total:
            raise ValueError(
                f"replay.episode_idx={idx} out of range (dataset has {total} episodes)."
            )
        return idx

    while True:
        raw = input(f"[replay] Pick an episode index (0..{total - 1}): ").strip()
        try:
            idx = int(raw)
        except ValueError:
            print(f"[replay] '{raw}' is not an integer.")
            continue
        if 0 <= idx < total:
            return idx
        print(f"[replay] {idx} out of range (0..{total - 1}).")


def _episode_range(dataset: LeRobotDataset, episode_idx: int) -> tuple[int, int]:
    """Return ``(start, end_exclusive)`` frame indices for episode ``episode_idx``.

    LeRobot v2 exposes this as ``dataset.episode_data_index[k]``; v3 removed
    that attribute and routes episode boundaries through the underlying
    HuggingFace ``hf_dataset["episode_index"]`` column instead.
    """
    if hasattr(dataset, "episode_data_index") and dataset.episode_data_index is not None:
        start, end = dataset.episode_data_index[episode_idx]
        return int(start), int(end)
    ep_idx = np.asarray(dataset.hf_dataset["episode_index"])
    mask = ep_idx == episode_idx
    where = np.where(mask)[0]
    if where.size == 0:
        raise ValueError(
            f"No frames found for episode {episode_idx} in dataset (is it deleted?)."
        )
    return int(where[0]), int(where[-1]) + 1


def _load_episode(
    dataset: LeRobotDataset, episode_idx: int, *, load_images: bool
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], str]:
    """Return (states, actions, images, prompt) for one episode.

    When ``load_images`` is False, each images entry is a 1x1 uint8 placeholder.
    Raw replay never touches images, so loading them just wastes ~1-2 GB for a
    500-frame episode.
    """
    start, end = _episode_range(dataset, episode_idx)
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    images: list[np.ndarray] = []
    prompt = ""
    zero_img = np.zeros((1, 1, 3), np.uint8)
    for i in range(start, end):
        sample = dataset[i]
        states.append(np.asarray(sample[_STATE_KEY], dtype=np.float32))
        actions.append(np.asarray(sample[_ACTION_KEY], dtype=np.float32))
        if load_images:
            # openpi transforms tolerate both HWC (uint8) and CHW (float32)
            # layouts via ``_parse_image`` in the dataconfig; just hand them
            # the raw decoded frame.
            img = sample.get(_GLOBAL_IMAGE_KEY)
            images.append(np.asarray(img) if img is not None else zero_img)
        else:
            images.append(zero_img)
        if not prompt:
            task = sample.get("task")
            if isinstance(task, bytes):
                prompt = task.decode("utf-8")
            elif task is not None:
                prompt = str(task)
    return states, actions, images, prompt


def _round_trip_actions(
    states: list[np.ndarray],
    actions: list[np.ndarray],
    images: list[np.ndarray],
    cfg: DictConfig,
) -> list[np.ndarray]:
    # Lazy import so ``raw`` mode doesn't need openpi installed.
    from real.replay_transforms import build_round_trip_fn

    checkpoint = cfg.inference.get("checkpoint")
    if checkpoint in (None, "", "???"):
        raise ValueError(
            "replay.mode=round_trip requires inference.checkpoint=<path-to-checkpoint-with-assets>"
        )
    round_trip = build_round_trip_fn(
        checkpoint_dir=str(checkpoint),
        config_name=str(cfg.inference.config_name),
    )

    out = [round_trip(s, a, img) for s, a, img in zip(states, actions, images)]
    raw = np.stack(actions, axis=0)
    rtr = np.stack(out, axis=0)
    diff = np.abs(rtr - raw)
    per_dim_max = diff.max(axis=0)
    per_dim_rms = np.sqrt((diff ** 2).mean(axis=0))
    max_abs = float(diff.max())

    axis_names = [
        "x", "y", "z",
        "r6d_0", "r6d_1", "r6d_2", "r6d_3", "r6d_4", "r6d_5",
        "gripper",
    ]
    print("[replay] Round-trip diff (|raw - denorm(norm(raw))|):")
    for name, mx, rms in zip(axis_names, per_dim_max, per_dim_rms):
        print(f"         {name:8s}  max={mx:.6e}  rms={rms:.6e}")
    print(f"[replay] overall max |diff| = {max_abs:.6e}")

    tolerance = float(cfg.replay.tolerance)
    if max_abs > tolerance:
        bad = [axis_names[i] for i, v in enumerate(per_dim_max) if v > tolerance]
        raise RuntimeError(
            f"Round-trip diverges beyond tolerance={tolerance:.1e} on dims {bad}. "
            "The openpi transform chain is not self-inverse for this checkpoint "
            "+ dataset combination. Aborting before touching the robot."
        )
    return out


def _append_log(log_path: Path, entry: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _run_one(cfg: DictConfig, env: FR3RealEnv, dataset: LeRobotDataset) -> bool:
    """Run one replay rollout. Returns True if the user wants to continue."""
    total = int(dataset.meta.total_episodes)
    if total == 0:
        raise RuntimeError(f"Dataset {cfg.dataset.repo_id!r} is empty.")

    episode_idx = _prompt_episode_idx(total, cfg.replay.get("episode_idx"))
    mode = str(cfg.replay.mode)
    states, actions, images, prompt = _load_episode(
        dataset, episode_idx, load_images=(mode == "round_trip")
    )
    n_steps = len(actions)
    print(
        f"[replay] Episode {episode_idx}: {n_steps} frames, prompt={prompt!r}."
    )

    if mode == "raw":
        actions_to_play = actions
    elif mode == "round_trip":
        actions_to_play = _round_trip_actions(states, actions, images, cfg)
    else:
        raise ValueError(
            f"replay.mode={mode!r} not understood. Use 'raw' or 'round_trip'."
        )

    print(f"\n[replay] Resetting robot for episode {episode_idx}...")
    env.reset()
    input(f"[replay] Press [Enter] to start the {mode} rollout... ")

    for t, action in enumerate(actions_to_play):
        env.step(np.asarray(action, dtype=np.float32))
        if t % 50 == 0:
            print(f"[replay]   step {t}/{n_steps}")

    # Label success
    while True:
        label = input("[replay] Label rollout (s)uccess / (f)ailure: ").strip().lower()
        if label in ("s", "f"):
            break

    _append_log(
        Path(cfg.replay.log_path),
        {
            "episode_idx": episode_idx,
            "mode": mode,
            "steps": n_steps,
            "label": "success" if label == "s" else "failure",
            "repo_id": str(cfg.dataset.repo_id),
            "checkpoint": str(cfg.inference.get("checkpoint", "")),
        },
    )
    print(f"[replay] Labeled episode {episode_idx} as {label!r}.")

    if not bool(cfg.replay.get("interactive", False)):
        return False
    again = input("[replay] Replay another episode? (y/N): ").strip().lower()
    return again == "y"


@hydra.main(config_path="config", config_name="replay", version_base=None)
def main(cfg: DictConfig) -> None:
    root = Path(str(cfg.dataset.root)).expanduser().resolve()
    repo_id = str(cfg.dataset.repo_id)
    ds_root = root / repo_id
    if not (ds_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"No LeRobot dataset at {ds_root}. Set dataset.root and dataset.repo_id on the CLI."
        )
    dataset = LeRobotDataset(repo_id=repo_id, root=ds_root)
    print(f"[replay] Loaded dataset {repo_id!r} with {dataset.meta.total_episodes} episodes.")

    task_prompt = ""
    if dataset.meta.total_episodes > 0:
        first_start, _ = _episode_range(dataset, 0)
        task0 = dataset[int(first_start)].get("task")
        if task0 is not None:
            task_prompt = task0.decode("utf-8") if isinstance(task0, bytes) else str(task0)

    env = FR3RealEnv(cfg, task=task_prompt)
    env.start()

    try:
        qpos = env.controller.get_qpos()
    except Exception as e:
        raise RuntimeError(
            "Robot health check failed right after env.start(). The OSC server "
            "at cfg.zmq.url is not responding — start real/run_osc_server.sh."
        ) from e
    print(f"[replay] Robot health check OK (qpos={qpos}).")

    try:
        while _run_one(cfg, env, dataset):
            continue
    finally:
        env.close()


if __name__ == "__main__":
    main()
