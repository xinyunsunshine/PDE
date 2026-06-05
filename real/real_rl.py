"""Collect one PPO iteration worth of real-world rollouts on FR3 + AgileX.

Each invocation collects ``rl.num_envs`` sequential rollouts (one PPO iteration)
under ``rl.run_root/iterationNNN/`` as per-rollout HDF5 files, then exits. The
trainer (out of scope) loads these, updates the VLA, and the user manually
relaunches this script for iteration N+1.

Layout (one PPO iteration per invocation):

    rl.run_root/
        iteration000/
            iteration_meta.json        # written on first rollout, read on resume
            rollout000.h5
            rollout001.h5
            ...
            rollout{num_envs-1}.h5
            iteration_complete.json    # written when all num_envs rollouts done
        iteration001/
            ...

Per-rollout HDF5 schema (T = horizon, zero-padded after termination):

    /state                                 (T, 10)                     float32
    /images/global                         (T, H_cam, W_cam, 3)        uint8
    /images/wrist                          (T, H_cam, W_cam, 3)        uint8
    /actions                               (T, chunk_size, action_dim) float32
    /valid_start                           (T,)                        int32
    /execute_k                             (T,)                        int32
    /forward_inputs/<key>                  (T, ...)                    server-defined
    /rewards                               (T,)                        float32
    /dones                                 (T,)                        uint8
    /terminations                          (T,)                        uint8
    /truncations                           (T,)                        uint8

External dependency:

    real/vla_server.py must return a `forward_inputs` dict alongside `actions`
    so PPO can recompute current-policy logprobs. Validated at first inference;
    fails fast with a pointed error if the contract is missing.

Trainer-side: per-chunk action-position mask
--------------------------------------------

The VLA emits a 50-action chunk at every inference, but only positions
``[valid_start, valid_start + execute_k)`` of that chunk actually moved the
robot (the first ``valid_start`` positions are skipped to compensate for
inference latency, and ``execute_k`` is set per ``inference.execute_k`` in
the config). PPO should reweight the per-position logprob by an
``is_executed`` mask before reducing over the chunk axis, so positions the
policy proposed but never executed don't influence the gradient.

The mask is fully reconstructible from ``valid_start`` and ``execute_k``,
which are stored per-step in every rollout HDF5. One-liner (numpy / torch):

    # rollout_h5["valid_start"]: (T,) int32   ; rollout_h5["execute_k"]: (T,) int32
    # rollout_h5["actions"].shape == (T, chunk_size, action_dim)
    chunk_size = rollout_h5["actions"].shape[1]
    positions = np.arange(chunk_size)                         # (chunk_size,)
    valid_start = rollout_h5["valid_start"][:]                # (T,)
    execute_k   = rollout_h5["execute_k"][:]                  # (T,)
    is_executed = (
        (positions[None, :] >= valid_start[:, None])
        & (positions[None, :] < (valid_start + execute_k)[:, None])
    )                                                         # (T, chunk_size) bool

Stack across N rollouts to get ``(N, T, chunk_size)`` and broadcast against
the per-position logprob tensor in the policy loss.

Launch:

    # 1) OSC server (in a separate terminal):
    python -m real.server.osc_server

    # 2) VLA inference server (separate terminal):
    python -m real.vla_server \\
        inference.weights=/path/to/full_weights.pt \\
        inference.norm_stats=/path/to/norm_stats.json

    # 3) This script:
    python -m real.real_rl \\
        rl.run_root=/path/to/run_root \\
        rl.num_envs=8 rl.horizon=64 \\
        rl.prompts_path=/path/to/prompts.json

    The prompts JSON must be a per-env assignment file:

        {
            "n_envs": 8,
            "assignments": [
                {"env_idx": 0, "prompt": "..."},
                {"env_idx": 1, "prompt": "..."},
                ...
            ]
        }

    Each rollout uses ``assignments[env_idx].prompt`` directly — no
    deduplication, no separate index list.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any, Optional

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from real.env import FR3RealEnv
from real.iteration_paths import (
    next_iteration_to_run,
    prompts_path_for_iteration,
)
from real.vla_client import connect_vla_client

GLOBAL_CAMERA_NAME = "global"
WRIST_CAMERA_NAME = "wrist"

REQUIRED_FORWARD_INPUTS = {
    "chains": np.float32,
    "denoise_inds": np.int64,
    "observation/image": np.uint8,
    "observation/state": np.float32,
    "tokenized_prompt": np.int32,
    "tokenized_prompt_mask": np.bool_,
}


@contextmanager
def raw_stdin():
    """Put stdin into raw mode so single keypresses are available immediately."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def key_pressed() -> Optional[str]:
    """Return the key if one is waiting on stdin, else None. Non-blocking."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def _obs_to_openpi_input(obs: dict, prompt: str) -> dict:
    """Pack an FR3RealEnv observation into pi0's expected input keys."""
    return {
        "observation/image": np.asarray(obs["images"][GLOBAL_CAMERA_NAME], dtype=np.uint8),
        "observation/wrist_image": np.asarray(obs["images"][WRIST_CAMERA_NAME], dtype=np.uint8),
        "observation/state": np.asarray(obs["state"], dtype=np.float32),
        "prompt": prompt,
    }


def _cfg_get(cfg, key: str):
    """Read a key from a dict-like or attribute-like config object."""
    if cfg is None:
        return None
    if hasattr(cfg, "get"):
        return cfg.get(key)
    return getattr(cfg, key, None)


def clip_action_to_state(
    action: np.ndarray,
    state: np.ndarray,
    clip_cfg,
) -> np.ndarray:
    """Rescale a 10-dim FR3 action against the current robot state.
    """
    if clip_cfg is None:
        return action
    out = np.asarray(action, dtype=np.float32).copy()
    state = np.asarray(state, dtype=np.float32)

    xyz_norm_m = _cfg_get(clip_cfg, "xyz_norm_m")
    rot_d = _cfg_get(clip_cfg, "rot6d_per_dim")
    grip_d = _cfg_get(clip_cfg, "grip")

    if xyz_norm_m is not None:
        delta = out[0:3] - state[0:3]
        norm = float(np.linalg.norm(delta))
        bound = float(xyz_norm_m)
        if norm > bound and norm > 1e-9:
            out[0:3] = state[0:3] + delta * (bound / norm)
    if rot_d is not None:
        out[3:9] = np.clip(out[3:9], state[3:9] - float(rot_d), state[3:9] + float(rot_d))
    if grip_d is not None:
        out[9:10] = np.clip(out[9:10], state[9:10] - float(grip_d), state[9:10] + float(grip_d))
    return out


def load_prompts_json(path: str | os.PathLike) -> list[str]:
    """Parse a prompts JSON file and return a per-env prompt list.

    Required format:
        {
            "n_envs": 32,
            "assignments": [
                {"env_idx": 0, "prompt": "..."},
                {"env_idx": 1, "prompt": "..."},
                ...
            ]
        }

    Returns a ``list[str]`` of length ``n_envs`` where ``out[env_idx]`` is
    the prompt for that env. Each rollout reads its prompt directly from
    this list — no deduplication, no index indirection.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"prompts_path does not exist: {path}")
    with open(path) as f:
        payload = json.load(f)

    if not isinstance(payload, dict) or "assignments" not in payload:
        raise ValueError(
            f"{path}: expected an object with an 'assignments' list; "
            f"got top-level {type(payload).__name__}. See load_prompts_json "
            "docstring for the required schema."
        )
    assignments = payload["assignments"]
    if not isinstance(assignments, list) or not assignments:
        raise ValueError(f"{path}: 'assignments' must be a non-empty list.")

    n_envs = len(assignments)
    per_env_prompts: list[str | None] = [None] * n_envs
    for i, entry in enumerate(assignments):
        if not isinstance(entry, dict) or "env_idx" not in entry or "prompt" not in entry:
            raise ValueError(
                f"{path}: assignments[{i}] must be an object with 'env_idx' and 'prompt'."
            )
        env_idx = int(entry["env_idx"])
        if env_idx < 0 or env_idx >= n_envs:
            raise ValueError(
                f"{path}: assignments[{i}].env_idx={env_idx} out of range "
                f"[0, {n_envs})."
            )
        if per_env_prompts[env_idx] is not None:
            raise ValueError(f"{path}: duplicate env_idx={env_idx} in assignments.")
        per_env_prompts[env_idx] = str(entry["prompt"])

    missing = [i for i, p in enumerate(per_env_prompts) if p is None]
    if missing:
        raise ValueError(f"{path}: missing assignments for env_idx(s) {missing}.")
    if "n_envs" in payload and int(payload["n_envs"]) != n_envs:
        raise ValueError(
            f"{path}: n_envs={payload['n_envs']} disagrees with "
            f"len(assignments)={n_envs}."
        )
    return [str(p) for p in per_env_prompts]


def validate_forward_inputs_schema(fwd_inputs: Any) -> None:
    """Sanity-check the server's forward_inputs response on the first inference.

    Fail fast (before any disk write) if a key is missing or the dtype is off.
    Extra keys are allowed and stored verbatim under /forward_inputs/<key>.
    """
    if not isinstance(fwd_inputs, dict):
        raise ValueError(
            "VLA server response is missing 'forward_inputs' or it is not a dict. "
            "real_rl.py expects client.infer(obs) to return "
            "{'actions': ..., 'forward_inputs': {chains, denoise_inds, "
            "observation/image, observation/state, tokenized_prompt, "
            "tokenized_prompt_mask, ...}}. Update real/vla_server.py to use "
            "OpenPi0ForRLActionPrediction.predict_action_batch(env_obs, mode='eval')."
        )
    missing = [k for k in REQUIRED_FORWARD_INPUTS if k not in fwd_inputs]
    if missing:
        raise ValueError(
            f"VLA server's forward_inputs is missing required keys: {missing}. "
            f"Got keys: {sorted(fwd_inputs.keys())}."
        )
    for key, want_dtype in REQUIRED_FORWARD_INPUTS.items():
        arr = np.asarray(fwd_inputs[key])
        if arr.dtype.kind != np.dtype(want_dtype).kind:
            raise ValueError(
                f"forward_inputs[{key!r}] has dtype {arr.dtype}, expected "
                f"{np.dtype(want_dtype)} (kind {np.dtype(want_dtype).kind})."
            )
        if arr.ndim < 1 or arr.shape[0] != 1:
            raise ValueError(
                f"forward_inputs[{key!r}] has shape {arr.shape}; expected a "
                "leading batch dim of 1."
            )


@dataclasses.dataclass
class RolloutBuffer:
    """Per-rollout buffer of fixed time-axis length T = horizon.

    Robot-side fields are explicit numpy arrays. forward_inputs is a dict whose
    schema is determined dynamically on the first step (so we don't have to
    hard-code every key the server might add).
    """
    horizon: int
    chunk_size: int
    action_dim: int                                  # of executed action (typically 10)
    state: np.ndarray                                # (T, state_dim)
    images: dict[str, np.ndarray]                    # {camera_name: (T, H, W, 3) uint8}
    actions: np.ndarray                              # (T, chunk_size, action_dim)
    valid_start: np.ndarray                          # (T,) int32
    execute_k: np.ndarray                            # (T,) int32
    rewards: np.ndarray                              # (T,) float32
    dones: np.ndarray                                # (T,) uint8
    terminations: np.ndarray                         # (T,) uint8
    truncations: np.ndarray                          # (T,) uint8
    forward_inputs: dict[str, np.ndarray]            # {key: (T, ...) <dtype>}

    @classmethod
    def allocate(
        cls,
        horizon: int,
        executed_chunk_shape: tuple[int, ...],
        state_shape: tuple[int, ...],
        cam_image_shapes: dict[str, tuple[int, ...]],
        forward_inputs_shapes: dict[str, tuple[int, ...]],
        forward_inputs_dtypes: dict[str, np.dtype],
    ) -> "RolloutBuffer":
        chunk_size, action_dim = executed_chunk_shape
        return cls(
            horizon=horizon,
            chunk_size=chunk_size,
            action_dim=action_dim,
            state=np.zeros((horizon,) + state_shape, dtype=np.float32),
            images={
                name: np.zeros((horizon,) + shape, dtype=np.uint8)
                for name, shape in cam_image_shapes.items()
            },
            actions=np.zeros((horizon, chunk_size, action_dim), dtype=np.float32),
            valid_start=np.zeros(horizon, dtype=np.int32),
            execute_k=np.zeros(horizon, dtype=np.int32),
            rewards=np.zeros(horizon, dtype=np.float32),
            dones=np.zeros(horizon, dtype=np.uint8),
            terminations=np.zeros(horizon, dtype=np.uint8),
            truncations=np.zeros(horizon, dtype=np.uint8),
            forward_inputs={
                key: np.zeros((horizon,) + shape, dtype=forward_inputs_dtypes[key])
                for key, shape in forward_inputs_shapes.items()
            },
        )

    def record_step(
        self,
        t: int,
        obs: dict,
        chunk: np.ndarray,
        valid_start: int,
        execute_k: int,
        forward_inputs: dict[str, np.ndarray],
    ) -> None:
        self.state[t] = np.asarray(obs["state"], dtype=np.float32)
        for name, frame in obs["images"].items():
            if name in self.images:
                self.images[name][t] = np.asarray(frame, dtype=np.uint8)
        self.actions[t] = np.asarray(chunk, dtype=np.float32)
        self.valid_start[t] = int(valid_start)
        self.execute_k[t] = int(execute_k)
        for key, val in forward_inputs.items():
            arr = np.asarray(val)
            # squeeze the leading batch dim (= 1 for single-env real-world rollouts)
            if arr.shape[0] == 1:
                arr = arr[0]
            if key in self.forward_inputs:
                self.forward_inputs[key][t] = arr

    def set_terminal(self, t_idx: int, success: bool, truncated: bool) -> None:
        """Set the terminal reward, done, termination, and truncation flags.

        ``truncated`` should be True when the episode ended because the time
        limit (horizon) was hit, and False for any other end (success, user
        failure, or crash). ``dones`` is set unconditionally; exactly one of
        ``terminations`` / ``truncations`` is set, mirroring Gymnasium's
        five-tuple convention.
        """
        if t_idx < 0 or t_idx >= self.horizon:
            raise ValueError(f"t_idx={t_idx} out of range [0, {self.horizon}).")
        self.rewards[t_idx] = 1.0 if success else 0.0
        self.dones[t_idx] = 1
        if truncated:
            self.truncations[t_idx] = 1
        else:
            self.terminations[t_idx] = 1



def write_rollout_h5_atomic(
    path: Path,
    buf: RolloutBuffer,
    attrs: dict[str, Any],
    compression: str,
    compression_level: int,
    store_images: bool,
) -> None:
    """Write a rollout buffer to an HDF5 file atomically (tempfile + rename)."""
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    # If a stale tmp file exists from a previous crash, remove it.
    if tmp_path.exists():
        tmp_path.unlink()

    use_gzip = compression == "gzip"
    image_kw = {"compression": "gzip", "compression_opts": compression_level} if use_gzip else {}
    chains_kw = image_kw  # chains is also large; reuse the same compression

    try:
        with h5py.File(tmp_path, "w") as f:
            # Robot-side, top-level
            f.create_dataset("state", data=buf.state)
            if store_images:
                grp_imgs = f.create_group("images")
                for name, arr in buf.images.items():
                    grp_imgs.create_dataset(name, data=arr, **image_kw)
            f.create_dataset("actions", data=buf.actions)
            f.create_dataset("valid_start", data=buf.valid_start)
            f.create_dataset("execute_k", data=buf.execute_k)
            f.create_dataset("rewards", data=buf.rewards)
            f.create_dataset("dones", data=buf.dones)
            f.create_dataset("terminations", data=buf.terminations)
            f.create_dataset("truncations", data=buf.truncations)

            # forward_inputs (server-defined keys; key may contain '/' as in
            # 'observation/image' which h5py turns into a nested group, fine).
            grp_fi = f.create_group("forward_inputs")
            for key, arr in buf.forward_inputs.items():
                # Apply gzip to large arrays (chains, images, state). Cheap call;
                # h5py only compresses contiguous datasets so small ones are no-ops.
                ds_kw = chains_kw if (key == "chains" or "image" in key) else {}
                grp_fi.create_dataset(key, data=arr, **ds_kw)

            for k, v in attrs.items():
                # h5py doesn't accept None / Path; cast appropriately.
                if v is None:
                    continue
                if isinstance(v, Path):
                    v = str(v)
                f.attrs[k] = v
        os.replace(tmp_path, path)
    except BaseException:
        # On any error (including KeyboardInterrupt), drop the tmp so it doesn't
        # mislead future resumes.
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def write_iteration_meta(
    iter_dir: Path,
    iteration: int,
    cfg: DictConfig,
    per_env_prompts: list[str],
) -> None:
    iter_dir.mkdir(parents=True, exist_ok=True)
    meta_path = iter_dir / "iteration_meta.json"
    payload = {
        "iteration": int(iteration),
        "num_envs": int(cfg.rl.num_envs),
        "horizon": int(cfg.rl.horizon),
        "execute_k_cfg": int(cfg.inference.execute_k),
        "prompts_path_at_first_run": str(Path(cfg.rl.prompts_path).expanduser().resolve()),
        "per_env_prompts": list(per_env_prompts),
        "started_at_unix": time.time(),
        "started_at_iso": datetime.now(timezone.utc).isoformat(),
        "control_freq_hz": float(cfg.robot.freq),
    }
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, meta_path)


def write_iteration_complete(iter_dir: Path, iteration: int, num_envs: int, success_rate: float) -> None:
    marker = iter_dir / "iteration_complete.json"
    payload = {
        "iteration": int(iteration),
        "num_envs": int(num_envs),
        "completed_at_unix": time.time(),
        "completed_at_iso": datetime.now(timezone.utc).isoformat(),
        "success_rate": float(success_rate),
    }
    tmp = marker.with_suffix(marker.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, marker)


_ITER_DIR_RE = re.compile(r"^iteration(\d+)$")
_ROLLOUT_FILE_RE = re.compile(r"^rollout(\d+)\.h5$")


def _scan_iterations(run_root: Path) -> list[int]:
    """Return the sorted list of iteration numbers present under run_root."""
    if not run_root.is_dir():
        return []
    out: list[int] = []
    for entry in run_root.iterdir():
        if not entry.is_dir():
            continue
        m = _ITER_DIR_RE.match(entry.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def _compute_success_rate_from_disk(iter_dir: Path) -> float:
    """Success rate from on-disk rollout `success` attrs.

    Reading from disk (instead of an in-memory counter) keeps the rate correct
    across resumed sessions — successes from earlier runs are counted too.
    """
    paths = sorted(iter_dir.glob("rollout*.h5"))
    if not paths:
        return 0.0
    n_success = 0
    for path in paths:
        with h5py.File(path, "r") as f:
            if bool(f.attrs.get("success", False)):
                n_success += 1
    return n_success / len(paths)


def _count_completed_rollouts(iter_dir: Path) -> int:
    """Count rolloutMMM.h5 files (ignoring .tmp), checking for sequential numbering.

    Returns the largest M+1 such that rollout000.h5 .. rolloutMMM.h5 all exist.
    Gaps abort with an error (we don't quietly skip them).
    """
    if not iter_dir.is_dir():
        return 0
    indices: list[int] = []
    for entry in iter_dir.iterdir():
        if not entry.is_file():
            continue
        m = _ROLLOUT_FILE_RE.match(entry.name)
        if m:
            indices.append(int(m.group(1)))
    if not indices:
        return 0
    indices.sort()
    expected = list(range(len(indices)))
    if indices != expected:
        raise RuntimeError(
            f"Rollout files in {iter_dir} are not contiguous from 0: found {indices}. "
            "Refuse to resume; manually clean up the iteration before retrying."
        )
    return len(indices)


def resolve_iteration_and_resume(
    cfg: DictConfig,
    per_env_prompts: list[str],
) -> tuple[int, int, Path]:
    """Determine target iteration K and resume index M.

    Validates / writes iteration_meta.json. Returns (K, M, iter_dir).
    """
    run_root = Path(str(cfg.rl.run_root)).expanduser()
    run_root.mkdir(parents=True, exist_ok=True)

    existing = _scan_iterations(run_root)

    if cfg.rl.iteration is not None:
        K = int(cfg.rl.iteration)
    elif not existing:
        K = 0
    else:
        last = existing[-1]
        if (run_root / f"iteration{last:03d}" / "iteration_complete.json").is_file():
            K = last + 1
        else:
            K = last
            print(f"### Resuming rollout iteration {K} that ended early")

    iter_dir = run_root / f"iteration{K:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    M = _count_completed_rollouts(iter_dir)

    meta_path = iter_dir / "iteration_meta.json"
    if meta_path.is_file():
        with open(meta_path) as f:
            stored = json.load(f)
        if list(stored.get("per_env_prompts", [])) != list(per_env_prompts):
            raise RuntimeError(
                f"per_env_prompts mismatch on resume of iteration {K}.\n"
                f"  stored ({meta_path}):\n    {stored.get('per_env_prompts')}\n"
                f"  current (rl.prompts_path={cfg.rl.prompts_path}):\n    {per_env_prompts}\n"
                "Refuse to mix rollouts from different prompt assignments within an iteration."
            )
        if int(stored.get("num_envs", -1)) != int(cfg.rl.num_envs):
            raise RuntimeError(
                f"num_envs mismatch on resume of iteration {K}: "
                f"stored={stored.get('num_envs')}, current={cfg.rl.num_envs}."
            )
        if int(stored.get("horizon", -1)) != int(cfg.rl.horizon):
            raise RuntimeError(
                f"horizon mismatch on resume of iteration {K}: "
                f"stored={stored.get('horizon')}, current={cfg.rl.horizon}."
            )
    else:
        if M > 0:
            raise RuntimeError(
                f"iteration {K} has {M} rollouts but no iteration_meta.json. "
                "Refuse to resume without canonical prompt record. "
                "Either restore the meta file or pick a fresh rl.iteration."
            )
        write_iteration_meta(iter_dir, K, cfg, per_env_prompts)

    return K, M, iter_dir


def print_prompt_assignment_banner(
    per_env_prompts: list[str],
    M: int,
) -> None:
    """Print one line per env with its prompt + done/next/todo tag."""
    print(f"[real_rl] Per-env prompts ({len(per_env_prompts)} envs):")
    for env_idx, prompt in enumerate(per_env_prompts):
        if env_idx < M:
            tag = "done"
        elif env_idx == M:
            tag = "next"
        else:
            tag = "todo"
        print(f"[real_rl]   env {env_idx:>3d} [{tag:>4s}]: {prompt!r}")


def prompt_for_label_blocking() -> str:
    while True:
        label = input(
            "[real_rl] Label rollout (s)uccess / (f)ailure / (r)eset: "
        ).strip().lower()
        if label in ("s", "f", "r"):
            return {"s": "success", "f": "failure", "r": "reset"}[label]


def prompt_for_horizon_success_blocking() -> bool:
    """At horizon, ask whether the rollout succeeded.

    True → true termination (V(s_T)=0). False → time-limit truncation
    (bootstrap V(s_T)).
    """
    while True:
        label = input(
            "[real_rl] Horizon hit. Success? (y)es / (n)o: "
        ).strip().lower()
        if label in ("y", "yes"):
            return True
        if label in ("n", "no"):
            return False


def health_check_qpos(env: FR3RealEnv) -> None:
    try:
        qpos = env.controller.get_qpos()
    except Exception as e:
        raise RuntimeError(
            "Robot health check failed right after env.start(). The OSC server "
            "at cfg.zmq.url is not responding, or the 1 kHz libfranka loop died "
            "(check the server's terminal for 'communication_constraints_violation'). "
            "Re-activate FCI from Desk UI and relaunch real/run_osc_server.sh, "
            f"then retry. Underlying: {e!r}"
        ) from e
    print(f"[real_rl] Robot health check OK (qpos={qpos}).")



def validate_cfg(cfg: DictConfig) -> list[str]:
    """Validate cfg, load per-env prompt list, return it.

    The prompts.json file's `assignments` list is the single source of
    truth: ``per_env_prompts[env_idx]`` is the prompt that env will run.

    Auto-resolve: if ``rl.prompts_path`` is null, resolve it from
    ``rl.run_root`` using the trainer-side convention
    ``<run_root>/iteration{K:03d}/itr{K:02d}_prompt.json`` where K is the
    next iteration to run (determined the same way as
    ``resolve_iteration_and_resume`` so they always agree).
    """
    if cfg.rl.prompts_path is None:
        # Auto-resolve from run_root + next iteration index.
        run_root = Path(str(cfg.rl.run_root)).expanduser()
        K = int(cfg.rl.iteration) if cfg.rl.iteration is not None else next_iteration_to_run(run_root)
        auto_path = prompts_path_for_iteration(run_root, K)
        if not auto_path.is_file():
            raise FileNotFoundError(
                f"Auto-resolved prompts path does not exist: {auto_path}\n"
                f"(rl.run_root={run_root}, next-iteration K={K})\n"
                "Drop the per-env prompts file at that path, or pass an explicit "
                "rl.prompts_path=/abs/path."
            )
        print(f"[real_rl] Auto-resolved rl.prompts_path = {auto_path} (iter K={K})")
        cfg.rl.prompts_path = str(auto_path)
    if int(cfg.rl.num_envs) <= 0:
        raise ValueError(f"rl.num_envs must be > 0; got {cfg.rl.num_envs}.")
    if int(cfg.rl.horizon) <= 0:
        raise ValueError(f"rl.horizon must be > 0; got {cfg.rl.horizon}.")
    if int(cfg.inference.execute_k) <= 0:
        raise ValueError(f"inference.execute_k must be > 0; got {cfg.inference.execute_k}.")

    per_env_prompts = load_prompts_json(str(cfg.rl.prompts_path))
    if len(per_env_prompts) != int(cfg.rl.num_envs):
        raise ValueError(
            f"prompts.json has {len(per_env_prompts)} assignments but "
            f"rl.num_envs={cfg.rl.num_envs}. Either update num_envs to match "
            f"the prompts file or fix the assignments list."
        )
    return per_env_prompts


@hydra.main(config_path="config", config_name="real_rl", version_base=None)
def main(cfg: DictConfig) -> None:
    per_env_prompts = validate_cfg(cfg)
    K, M, iter_dir = resolve_iteration_and_resume(cfg, per_env_prompts)
    rollouts_remaining = int(cfg.rl.num_envs) - M

    run_root = Path(str(cfg.rl.run_root)).expanduser().resolve()
    print(f"[real_rl] Run root: {run_root}")
    print(f"[real_rl] iter={K}  num_envs={cfg.rl.num_envs}  horizon={cfg.rl.horizon}  "
          f"completed={M}/{cfg.rl.num_envs}  remaining={rollouts_remaining}")
    print_prompt_assignment_banner(per_env_prompts, M)

    if rollouts_remaining <= 0:
        # if not (iter_dir / "iteration_complete.json").is_file():
        #     write_iteration_complete(iter_dir, K, int(cfg.rl.num_envs))
        #     print(f"[real_rl] Iteration {K} already had {M} rollouts; wrote completion marker.")
        # else:
        print(f"[real_rl] Iteration {K} already complete; nothing to do.")
        return

    # Hardware / server setup BEFORE the initial Enter gate so misconfigs
    # fail fast without making the user wait at a prompt for nothing.
    env = FR3RealEnv(cfg, task=per_env_prompts[M])
    env.start()
    obs, _ = env.reset()
    health_check_qpos(env)

    client = connect_vla_client(
        host=str(cfg.inference.server.host),
        port=int(cfg.inference.server.port),
    )

    input(f"[real_rl] Press <<Enter>> to begin iteration {K} ({rollouts_remaining} rollouts)... ")

    dt = 1.0 / float(cfg.robot.freq)
    horizon = int(cfg.rl.horizon)
    execute_k_cfg = int(cfg.inference.execute_k)
    rl_compression = str(cfg.rl.image_compression)
    rl_compression_level = int(cfg.rl.image_compression_level)
    rl_store_images = bool(cfg.rl.store_images)
    print(f"### image compression: {rl_compression} | level: {rl_compression_level}\n### ")
    if not rl_store_images:
        print("[Warn] Images are not being stored!!")

    rollout_idx = M
    pbar = tqdm(total=int(cfg.rl.num_envs), initial=M, desc=f"iter{K} rollouts")
    while rollout_idx < int(cfg.rl.num_envs):
        rollout_prompt = per_env_prompts[rollout_idx]

        print(f"\n[real_rl] iter {K} rollout {rollout_idx}/{int(cfg.rl.num_envs) - 1}: "
              f"resetting robot (prompt={rollout_prompt!r})...")
        try:
            obs, _ = env.reset()
        except KeyboardInterrupt:
            raise
        except RuntimeError as e:
            print(f"[real_rl] RuntimeError during env.reset(): {e!r}. "
                  "Discarding rollout (no h5 written). Fix the issue and re-run to resume.")
            return
        except Exception as e:
            print(f"[real_rl] env.reset() failed: {e!r}. Aborting; rerun to resume.")
            return

        input(
            f"[real_rl] Press <<Enter>> to start rollout {rollout_idx} "
            "(mid-rollout: s=success, f=failure, r=reset)... "
        )

        buf: Optional[RolloutBuffer] = None
        T_used = 0
        mid_label: Optional[str] = None
        crash_reason: Optional[str] = None
        start_ts = time.time()

        try:
            with raw_stdin():
                hit_horizon = True
                for t in range(horizon):
                    k = key_pressed()
                    if k in ("s", "f", "r"):
                        mid_label = {"s": "success", "f": "failure", "r": "reset"}[k]
                        print(f"\n[real_rl] '{k}' pressed before inference {t} — "
                              f"ending as {mid_label.upper()}.")
                        hit_horizon = False
                        break

                    f0 = time.monotonic()
                    out = client.infer(_obs_to_openpi_input(obs, rollout_prompt))
                    chunk = np.asarray(out["actions"])
                    fwd_inputs = out.get("forward_inputs")
                    elapsed = time.monotonic() - f0

                    print(f"elapsed: {elapsed}")

                    if buf is None:
                        validate_forward_inputs_schema(fwd_inputs)
                        buf = RolloutBuffer.allocate(
                            horizon=horizon,
                            executed_chunk_shape=chunk.shape,
                            state_shape=np.asarray(obs["state"]).shape,
                            cam_image_shapes={
                                name: np.asarray(frame).shape
                                for name, frame in obs["images"].items()
                            },
                            forward_inputs_shapes={
                                key: np.asarray(val).shape[1:]
                                for key, val in fwd_inputs.items()
                            },
                            forward_inputs_dtypes={
                                key: np.asarray(val).dtype
                                for key, val in fwd_inputs.items()
                            },
                        )

                    valid_start = 0 * max(1, ceil(elapsed / dt))
                    # wait_for = valid_start * dt - elapsed
                    # if wait_for > 0:
                    #     ts = time.monotonic()
                    #     while time.monotonic() - ts < wait_for:
                    #         time.sleep(0.001)

                    # import pdb;pdb.set_trace()
                    actions_to_execute = np.asarray(
                        chunk[valid_start : valid_start + execute_k_cfg]
                    )
                    actual_execute_k = int(actions_to_execute.shape[0])

                    buf.record_step(
                        t=t,
                        obs=obs,
                        chunk=chunk,
                        valid_start=valid_start,
                        execute_k=actual_execute_k,
                        forward_inputs=fwd_inputs,
                    )
                    T_used = t + 1

                    interrupted_during_execute = False
                    action_clip_cfg = cfg.inference.get("action_clip", None)
                    for a in actions_to_execute:
                        k2 = key_pressed()
                        if k2 in ("s", "f", "r"):
                            mid_label = {"s": "success", "f": "failure", "r": "reset"}[k2]
                            print(f"\n[real_rl] '{k2}' pressed mid-execute (inference {t}) — "
                                  f"ending as {mid_label.upper()}.")
                            interrupted_during_execute = True
                            break
                        a_clipped = clip_action_to_state(
                            np.asarray(a, dtype=np.float32),
                            np.asarray(obs["state"], dtype=np.float32),
                            action_clip_cfg,
                        )
                        obs, _, _, _, _ = env.step(np.array(a_clipped))
                    if interrupted_during_execute:
                        hit_horizon = False
                        break
            if hit_horizon:
                mid_label = (
                    "success"
                    if prompt_for_horizon_success_blocking()
                    else "horizon"
                )
        except KeyboardInterrupt:
            raise
        except RuntimeError as e:
            print(f"\n[real_rl] RuntimeError mid-rollout after {T_used} "
                  f"inference steps: {e!r}. Discarding rollout {rollout_idx} "
                  "(no h5 written). Fix the issue and re-run to resume.")
            return
        except Exception as e:
            crash_reason = repr(e)
            print(f"\n[real_rl] Rollout {rollout_idx} crashed mid-loop after "
                  f"{T_used} inference steps: {crash_reason}")
            mid_label = "failure"

        if mid_label is None:
            mid_label = prompt_for_label_blocking()

        if mid_label == "reset":
            print("[real_rl] Reset → discarding rollout, retrying same rollout_idx.")
            continue

        if buf is None:
            # Crash before the first inference; nothing to write usefully.
            print("[real_rl] No data captured (crash before first inference). "
                  "Aborting iteration without writing this rollout. Re-run to retry.")
            return

        success = (mid_label == "success")
        if crash_reason:
            termination_reason = "crash"
        elif mid_label == "success":
            termination_reason = "user_success"
        elif mid_label == "failure":
            termination_reason = "user_failure"
        elif mid_label == "horizon":
            termination_reason = "horizon"
        else:
            termination_reason = mid_label

        T_used = max(1, min(T_used, horizon))
        truncated = (termination_reason == "horizon")
        buf.set_terminal(t_idx=T_used - 1, success=success, truncated=truncated)

        attrs = {
            "prompt": rollout_prompt,
            "prompt_text_sha1": hashlib.sha1(rollout_prompt.encode("utf-8")).hexdigest(),
            "prompts_path": str(Path(str(cfg.rl.prompts_path)).expanduser().resolve()),
            "success": bool(success),
            "termination_reason": termination_reason,
            "iteration": int(K),
            "rollout_idx": int(rollout_idx),
            "num_inference_steps": int(T_used),
            "horizon": int(horizon),
            "chunk_size": int(buf.chunk_size),
            "action_dim": int(buf.action_dim),
            "control_freq_hz": float(cfg.robot.freq),
            "execute_k_cfg": int(execute_k_cfg),
            "start_ts_unix": float(start_ts),
            "end_ts_unix": time.time(),
            "vla_server_host": str(cfg.inference.server.host),
            "vla_server_port": int(cfg.inference.server.port),
        }

        write_rollout_h5_atomic(
            path=iter_dir / f"rollout{rollout_idx:03d}.h5",
            buf=buf,
            attrs=attrs,
            compression=rl_compression,
            compression_level=rl_compression_level,
            store_images=rl_store_images,
        )
        print(f"[real_rl] Wrote rollout {rollout_idx} "
              f"({termination_reason}, T={T_used}/{horizon}, success={success}).")

        if crash_reason:
            print("[real_rl] Crash recorded; aborting iteration. Re-run to resume.")
            return

        rollout_idx += 1
        pbar.update(1)

    pbar.close()
    success_rate = _compute_success_rate_from_disk(iter_dir)
    write_iteration_complete(iter_dir, K, int(cfg.rl.num_envs), success_rate)
    print(f"[real_rl] Iteration {K} complete. {int(cfg.rl.num_envs)} rollouts written. Success rate: {success_rate:.2%}")


if __name__ == "__main__":
    main()