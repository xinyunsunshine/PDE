"""HDF5 rollout format reader for the realworld training pipeline.

The bifranka inference server (`real/real_rl.py`) writes one PPO iteration
worth of rollouts as:

    rollout_root/iterationNNN/
        iteration_meta.json
        rollout000.h5
        rollout001.h5
        ...
        rollout{num_envs-1}.h5
        iteration_complete.json     ← only appears when all rollouts atomically committed

Per-rollout HDF5 schema (real/real_rl.py:21-33, with the
terminations/truncations split added in commit 7261b718):

    /state                                 (T, state_dim)              float32
    /images/global                         (T, H_cam, W_cam, 3)        uint8
    /images/wrist                          (T, H_cam, W_cam, 3)        uint8 (optional)
    /actions                               (T, chunk_size, action_dim) float32
    /valid_start                           (T,)                        int32
    /execute_k                             (T,)                        int32
    /forward_inputs/<key>                  (T, ...)                    server-defined
    /rewards                               (T,)                        float32
    /dones                                 (T,)                        uint8
    /terminations                          (T,)                        uint8
    /truncations                           (T,)                        uint8

T is fixed at `rl.horizon` (the writer zero-pads after termination);
chunk_size is the model's predicted-chunk length (typically 50);
action_dim is the *executed* dim count (10 for FR3 + AgileX gripper).

The action_mask the trainer-side `default_forward` consumes is
reconstructed in this module from `valid_start` and `execute_k` —
positions [valid_start, valid_start + execute_k) of the predicted
chunk are the executed window for each chunk-step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REQUIRED_FORWARD_INPUT_KEYS = (
    "chains",
    "denoise_inds",
    "observation/image",
    "observation/state",
    "tokenized_prompt",
    "tokenized_prompt_mask",
)

REQUIRED_TOPLEVEL_DATASETS = (
    "state",
    "actions",
    "valid_start",
    "execute_k",
    "rewards",
    "dones",
    "terminations",
    "truncations",
    "forward_inputs",
)


def is_iteration_complete(iter_dir: Path) -> bool:
    """True iff the upstream writer has committed every rollout in this iter."""
    return (Path(iter_dir) / "iteration_complete.json").is_file()


def load_rollout(h5_path: Path) -> dict[str, Any]:
    """Load one rolloutNNN.h5 into a dict of numpy arrays.

    Returns:
        {
            "horizon": int T,
            "state":          (T, state_dim) f32,
            "actions":        (T, chunk_size, action_dim) f32,
            "valid_start":    (T,) int32,
            "execute_k":      (T,) int32,
            "rewards":        (T,) f32,
            "dones":          (T,) bool,
            "terminations":   (T,) bool,
            "truncations":    (T,) bool,
            "forward_inputs": {<key>: (T, ...)},
        }
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        for k in REQUIRED_TOPLEVEL_DATASETS:
            if k not in f:
                raise ValueError(f"{h5_path}: missing required dataset '{k}'")

        out: dict[str, Any] = {
            "state":        f["state"][:],
            "actions":      f["actions"][:],
            "valid_start":  f["valid_start"][:].astype(np.int32),
            "execute_k":    f["execute_k"][:].astype(np.int32),
            "rewards":      f["rewards"][:].astype(np.float32),
            "dones":        f["dones"][:].astype(bool),
            "terminations": f["terminations"][:].astype(bool),
            "truncations":  f["truncations"][:].astype(bool),
        }

        # forward_inputs is an h5py group whose keys may contain '/' segments
        # (e.g., 'observation/image'); h5py represents those as nested groups.
        forward_inputs: dict[str, np.ndarray] = {}
        _collect_h5_leaves(f["forward_inputs"], "", forward_inputs)
        for k in REQUIRED_FORWARD_INPUT_KEYS:
            if k not in forward_inputs:
                raise ValueError(
                    f"{h5_path}: forward_inputs missing required key '{k}'. "
                    f"Got: {sorted(forward_inputs.keys())}"
                )
        out["forward_inputs"] = forward_inputs

    # Trim to actual episode length. The inference server buffers each
    # rollout into a fixed T (e.g., 128 chunk-steps) but real episodes
    # are shorter and variable. Past first_done the action_mask is
    # already all-False (execute_k=0), so the math is identical with or
    # without trim, but skipping the padded tail saves ~5x compute on
    # the no-grad logprob recompute.
    dones_tl = out["dones"]
    T_actual = (
        int(np.argmax(dones_tl)) + 1 if dones_tl.any() else dones_tl.shape[0]
    )
    if T_actual < dones_tl.shape[0]:
        for k in (
            "state",
            "actions",
            "valid_start",
            "execute_k",
            "rewards",
            "dones",
            "terminations",
            "truncations",
        ):
            out[k] = out[k][:T_actual]
        out["forward_inputs"] = {
            k: v[:T_actual] for k, v in out["forward_inputs"].items()
        }

    out["horizon"] = int(out["state"].shape[0])
    return out


def _collect_h5_leaves(group, prefix: str, out: dict[str, np.ndarray]) -> None:
    """Recursively walk an h5py group, collecting datasets keyed by '/'-joined path."""
    for name, item in group.items():
        path = f"{prefix}{name}" if not prefix else f"{prefix}/{name}"
        if isinstance(item, h5py.Group):
            _collect_h5_leaves(item, path, out)
        else:
            out[path] = item[:]


def load_iteration(iter_dir: Path) -> list[dict[str, Any]]:
    """Load every rollout*.h5 in an iteration dir, in deterministic order.

    Returns a list of length num_envs; each entry is one rollout dict.
    Caller should check is_iteration_complete(iter_dir) before calling.
    """
    iter_dir = Path(iter_dir)
    if not is_iteration_complete(iter_dir):
        raise ValueError(
            f"{iter_dir}: iteration_complete.json missing — refusing to load "
            "a partially-committed iteration."
        )
    rollouts: list[dict[str, Any]] = []
    for path in sorted(iter_dir.glob("rollout*.h5")):
        ep = load_rollout(path)
        ep["__source_path"] = str(path)
        rollouts.append(ep)
    if not rollouts:
        raise ValueError(f"{iter_dir}: no rollout*.h5 files found")
    return rollouts


def load_iteration_meta(iter_dir: Path) -> dict[str, Any]:
    """Read iteration_meta.json (prompts, indices, started_at, etc.)."""
    iter_dir = Path(iter_dir)
    meta_path = iter_dir / "iteration_meta.json"
    if not meta_path.is_file():
        return {}
    with open(meta_path) as f:
        return json.load(f)


def reconstruct_action_mask(
    valid_start: np.ndarray,
    execute_k: np.ndarray,
    chunk_size: int,
    action_dim_full: int,
    action_env_dim: int,
) -> np.ndarray:
    """Build a (T, chunk_size, action_dim_full) bool mask from the sliding-window encoding.

    Position i is "executed" iff valid_start[t] <= i < valid_start[t] + execute_k[t].
    Action dim j is "executed" iff j < action_env_dim (rest is cross-embodiment padding).

    The mask is what `OpenPi0ForRLActionPrediction.default_forward` consumes
    via forward_inputs["action_mask"] when the masked-sum logprob path is
    active. See the trainer-side proof in inference_server_episode_writer.md.
    """
    valid_start = np.asarray(valid_start, dtype=np.int64)
    execute_k = np.asarray(execute_k, dtype=np.int64)
    if valid_start.shape != execute_k.shape:
        raise ValueError(
            f"valid_start {valid_start.shape} and execute_k {execute_k.shape} must match"
        )
    T = valid_start.shape[0]
    positions = np.arange(chunk_size)                         # (chunk_size,)
    pos_mask = (
        (positions[None, :] >= valid_start[:, None])
        & (positions[None, :] < (valid_start + execute_k)[:, None])
    )                                                          # (T, chunk_size)
    dim_mask = np.arange(action_dim_full) < action_env_dim    # (action_dim_full,)
    mask = pos_mask[:, :, None] & dim_mask[None, None, :]     # (T, chunk_size, action_dim_full)
    assert mask.shape == (T, chunk_size, action_dim_full)
    return mask


def horizon_of(rollout: dict[str, Any]) -> int:
    return int(rollout["horizon"])
