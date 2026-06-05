"""Offline xyz prediction-error benchmark for the FR3 + AgileX VLA stack.

Drives the same VLA inference server that ``real/deploy.py`` connects to,
but feeds it observations from a LeRobot dataset (``real/collect_sft_data.py``
output) instead of live cameras + robot state. At every dataset transition
we run one forward pass and compare the predicted action chunk to the
recorded actions on the EE-translation channels (xyz, dims 0..2 of the
10-d action), reported in meters.

Successive forward passes overlap: the chunk inferred at frame ``t`` covers
frames ``t..t+execute_k-1`` and the chunk at ``t+1`` covers
``t+1..t+execute_k``, so each frame receives up to ``execute_k`` predictions
at offsets 0..execute_k-1 — useful for seeing how prediction error grows
with horizon.

This is purely diagnostic — no robot, no env, no controller. The openpi
``Policy`` on the server side owns all transforms (normalization, delta-
to-absolute, etc.), so the chunk returned by ``client.infer`` is already
in the same 10-d absolute space as ``dataset.action``.

Prereqs:
  * VLA inference server up:
      python -m real.vla_server \\
          inference.weights=/path/to/full_weights.pt \\
          inference.norm_stats=/path/to/norm_stats.json

Launch:
  python -m real.compute_mse \\
    dataset.root=/path/to/datasets \\
    dataset.repo_id=<repo_id>
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from real.env import PRIMARY_CAMERA_NAME
from real.vla_client import connect_vla_client

# LeRobot v3 renamed the module path; v2 kept it under ``lerobot.common``.
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
except ModuleNotFoundError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore


_GLOBAL_IMAGE_KEY = f"observation.images.{PRIMARY_CAMERA_NAME}"
_ACTION_KEY = "action"
_STATE_KEY = "observation.state"


def _episode_range(dataset: LeRobotDataset, episode_idx: int) -> tuple[int, int]:
    """Return ``(start, end_exclusive)`` frame indices for episode ``episode_idx``."""
    if hasattr(dataset, "episode_data_index") and dataset.episode_data_index is not None:
        start, end = dataset.episode_data_index[episode_idx]
        return int(start), int(end)
    ep_idx = np.asarray(dataset.hf_dataset["episode_index"])
    where = np.where(ep_idx == episode_idx)[0]
    if where.size == 0:
        raise ValueError(f"No frames found for episode {episode_idx}.")
    return int(where[0]), int(where[-1]) + 1


def _load_episode(
    dataset: LeRobotDataset, episode_idx: int
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], str]:
    """Return ``(states, actions, images, prompt)`` for one episode."""
    start, end = _episode_range(dataset, episode_idx)
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    images: list[np.ndarray] = []
    prompt = ""
    for i in range(start, end):
        sample = dataset[i]
        states.append(np.asarray(sample[_STATE_KEY], dtype=np.float32))
        actions.append(np.asarray(sample[_ACTION_KEY], dtype=np.float32))
        img = sample.get(_GLOBAL_IMAGE_KEY)
        if img is None:
            raise RuntimeError(
                f"Frame {i} (ep {episode_idx}) has no image at "
                f"{_GLOBAL_IMAGE_KEY!r}. Dataset must include the global camera."
            )
        images.append(np.asarray(img))
        if not prompt:
            task = sample.get("task")
            if isinstance(task, bytes):
                prompt = task.decode("utf-8")
            elif task is not None:
                prompt = str(task)
    return np.stack(states), np.stack(actions), images, prompt


def _dataset_to_openpi_input(
    image: np.ndarray, state: np.ndarray, prompt: str
) -> dict:
    """Pack one dataset frame into pi0's expected input keys.

    Same shape as ``_obs_to_openpi_input`` in ``real/deploy.py`` — only the
    source differs (dataset arrays vs. ``FR3RealEnv`` observation).
    """
    return {
        "observation/image": np.asarray(image, dtype=np.uint8),
        "observation/state": np.asarray(state, dtype=np.float32),
        "prompt": prompt,
    }


class _Accum:
    """Per-offset running buffers for xyz prediction error (in meters).

    ``count[k]`` is the number of frames whose chunk contributed at offset k
    (so frames in the last execute_k-1 of an episode contribute to fewer
    offsets). Sums are over (frame, offset) pairs at fixed offset k.
    """

    def __init__(self, K: int):
        self.K = K
        self.sum_sq_axis = np.zeros((K, 3), dtype=np.float64)  # MSE per axis
        self.sum_sq_l2 = np.zeros(K, dtype=np.float64)         # MSE of 3-D L2
        self.sum_abs_axis = np.zeros((K, 3), dtype=np.float64) # MAE per axis
        self.count = np.zeros(K, dtype=np.int64)

    def update(self, diff_xyz: np.ndarray) -> None:
        """``diff_xyz`` is (k_avail, 3) of (pred - target) in meters."""
        k = diff_xyz.shape[0]
        sq = diff_xyz.astype(np.float64) ** 2
        self.sum_sq_axis[:k] += sq
        self.sum_sq_l2[:k] += sq.sum(axis=1)
        self.sum_abs_axis[:k] += np.abs(diff_xyz)
        self.count[:k] += 1

    def aggregate_rmse(self) -> tuple[np.ndarray, float, int]:
        """Return (rmse_per_axis(3,), rmse_l2 scalar, n_samples) over all offsets."""
        total = int(self.count.sum())
        if total == 0:
            return np.zeros(3), 0.0, 0
        rmse_axis = np.sqrt(self.sum_sq_axis.sum(axis=0) / total)
        rmse_l2 = float(np.sqrt(self.sum_sq_l2.sum() / total))
        return rmse_axis, rmse_l2, total

    def per_offset_table(self) -> list[tuple[int, int, np.ndarray, float, np.ndarray]]:
        """Return ``[(offset, count, rmse_axis(3,), rmse_l2, mae_axis(3,))]``."""
        rows = []
        for k in range(self.K):
            n = int(self.count[k])
            if n == 0:
                rows.append((k, 0, np.zeros(3), 0.0, np.zeros(3)))
                continue
            rmse_axis = np.sqrt(self.sum_sq_axis[k] / n)
            rmse_l2 = float(np.sqrt(self.sum_sq_l2[k] / n))
            mae_axis = self.sum_abs_axis[k] / n
            rows.append((k, n, rmse_axis, rmse_l2, mae_axis))
        return rows


def _fmt_rmse(rmse_axis: np.ndarray, rmse_l2: float) -> str:
    return (
        f"x={rmse_axis[0]:.4f} y={rmse_axis[1]:.4f} z={rmse_axis[2]:.4f} "
        f"|xyz|={rmse_l2:.4f}"
    )


@hydra.main(config_path="config", config_name="compute_mse", version_base=None)
def main(cfg: DictConfig) -> None:
    root = Path(str(cfg.dataset.root)).expanduser().resolve()
    repo_id = str(cfg.dataset.repo_id)
    ds_root = root / repo_id
    if not (ds_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"No LeRobot dataset at {ds_root}. Set dataset.root and dataset.repo_id."
        )
    dataset = LeRobotDataset(repo_id=repo_id, root=ds_root)
    total_eps = int(dataset.meta.total_episodes)
    print(f"[compute_mse] Loaded dataset {repo_id!r} with {total_eps} episodes.")
    if total_eps == 0:
        raise RuntimeError(f"Dataset {repo_id!r} is empty.")

    execute_k = int(cfg.inference.get("execute_k", 25))
    print(f"[compute_mse] execute_k = {execute_k} (per-frame inference, overlapping chunks).")

    client = connect_vla_client(
        host=str(cfg.inference.server.host),
        port=int(cfg.inference.server.port),
    )

    max_episodes = cfg.get("max_episodes", None)
    n_eps = total_eps if max_episodes is None else min(int(max_episodes), total_eps)

    global_accum = _Accum(execute_k)

    for ep in range(n_eps):
        states, actions, images, prompt = _load_episode(dataset, ep)
        n_frames = len(actions)
        ep_accum = _Accum(execute_k)

        bar = tqdm(range(n_frames), desc=f"ep {ep}/{n_eps - 1}", leave=False)
        for t in bar:
            obs_in = _dataset_to_openpi_input(images[t], states[t], prompt)
            chunk = np.asarray(client.infer(obs_in)["actions"])
            k_avail = min(execute_k, n_frames - t)
            diff = chunk[:k_avail, :3] - actions[t : t + k_avail, :3]
            ep_accum.update(diff)
            global_accum.update(diff)

            rmse_axis, rmse_l2, _ = global_accum.aggregate_rmse()
            bar.set_postfix_str(_fmt_rmse(rmse_axis, rmse_l2))

        rmse_axis, rmse_l2, n = ep_accum.aggregate_rmse()
        print(
            f"[compute_mse] ep={ep:3d} frames={n_frames:4d} samples={n:5d} "
            f"RMSE_xyz(m): {_fmt_rmse(rmse_axis, rmse_l2)}, prompt={prompt!r}"
        )

    rmse_axis, rmse_l2, n = global_accum.aggregate_rmse()
    print(
        f"\n[compute_mse] Aggregate over {n_eps} episodes ({n} (frame, offset) "
        f"samples): RMSE_xyz(m): {_fmt_rmse(rmse_axis, rmse_l2)}"
    )

    print("\n[compute_mse] Per-offset xyz error (meters):")
    print(
        f"  {'k':>3s}  {'count':>6s}  {'RMSE_x':>8s}  {'RMSE_y':>8s}  "
        f"{'RMSE_z':>8s}  {'RMSE_|xyz|':>10s}  "
        f"{'MAE_x':>8s}  {'MAE_y':>8s}  {'MAE_z':>8s}"
    )
    for k, n_k, rmse_axis_k, rmse_l2_k, mae_axis_k in global_accum.per_offset_table():
        print(
            f"  {k:>3d}  {n_k:>6d}  "
            f"{rmse_axis_k[0]:>8.4f}  {rmse_axis_k[1]:>8.4f}  {rmse_axis_k[2]:>8.4f}  "
            f"{rmse_l2_k:>10.4f}  "
            f"{mae_axis_k[0]:>8.4f}  {mae_axis_k[1]:>8.4f}  {mae_axis_k[2]:>8.4f}"
        )


if __name__ == "__main__":
    main()
