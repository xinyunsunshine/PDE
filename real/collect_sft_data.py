"""Teleop SFT data collector for pi0(.5) on FR3 + AgileX.

Records absolute 10-dim end-effector + gripper trajectories to a LeRobot v2
dataset so the RLinf OpenPI SFT runner can consume it via HF_LEROBOT_HOME.

Control contract
----------------
    state  : (10,) float32   [xyz(3), rot6d(6), gripper_width(1)]
    action : (10,) float32   same layout, absolute (not delta)
    image  : (H, W, 3) uint8 rgb image from the primary RealSense camera


Keyboard (in both "idle" and "recording" modes unless noted):
    s    start recording a new episode (from idle only)
    e    end + save current episode (from recording only)
    r    discard the current episode and reset the robot
    q    quit the collector

Usage
-----
    ```
    python -m real.collect_sft_data \
        collect.dataset_root=/abs/path/lerobot_datasets \
        collect.repo_id=pickcube_v1 \
        collect.task_prompt="pick up the red block"
    ```

The collector is resume-safe.
"""

from __future__ import annotations

import enum
import json
import queue
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.video_utils import get_safe_default_codec
from omegaconf import DictConfig

from real.env import FR3RealEnv
from real.rotation_utils import homogeneous_to_pose_10d
from real.spacemouse import SpaceMouseController


class _KeyboardListener:
    """Non-blocking stdin key listener for SSH sessions.

    Uses termios + tty + select so it works without an X display,
    and runs in a daemon thread so the main teleop loop never blocks on key
    I/O. Presses are pushed to a thread-safe queue that the main loop drains.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._old_settings: list | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError(
                "Keyboard listener requires a TTY on stdin (run in a terminal)."
            )
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if rlist:
                ch = sys.stdin.read(1)
                if ch:
                    self._queue.put(ch.lower())

    def poll(self) -> str | None:
        """Return the next pressed key or ``None`` if the queue is empty."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None


class _GripperToggle:
    """Edge-triggered toggle between an open and closed gripper target.

    A press on any SpaceMouse button flips the current target. Re-triggering
    is suppressed until every button has been released, which avoids
    chattering when the user holds the button down.
    """

    def __init__(self, open_width: float, closed_width: float) -> None:
        self._open_width = float(open_width)
        self._closed_width = float(closed_width)
        self._target = self._open_width
        self._armed = True  # True iff all buttons have been released

    def reset(self) -> None:
        self._target = self._open_width
        self._armed = True

    def update(self, buttons: tuple[int, ...]) -> float:
        any_pressed = any(int(b) for b in buttons)
        if any_pressed and self._armed:
            self._target = (
                self._closed_width if self._target == self._open_width else self._open_width
            )
            self._armed = False
        elif not any_pressed:
            self._armed = True
        return self._target

    @property
    def target(self) -> float:
        return self._target


class _State(enum.Enum):
    IDLE = "idle"          # waiting for 's' to start recording
    RECORDING = "recording"  # teleop in progress, frames buffered
    SAVING = "saving"      # flush buffer to dataset, then -> IDLE
    DISCARDING = "discarding"  # drop buffer, reset, then -> IDLE
    QUITTING = "quitting"


STATE_AXIS_NAMES = [
    "x", "y", "z",
    "r6d_0", "r6d_1", "r6d_2", "r6d_3", "r6d_4", "r6d_5",
    "gripper",
]
ACTION_AXIS_NAMES = list(STATE_AXIS_NAMES)

STATE_KEY = "observation.state"
# LeRobot convention is singular "action"; the OpenPI dataconfig overrides
# action_sequence_keys to match and renames to plural "actions" in the
# repack transform (see fr3_vla_dataconfig.py).
ACTION_KEY = "action"


def _image_key(camera_name: str) -> str:
    """LeRobot feature key for a given camera name."""
    return f"observation.images.{camera_name}"


def _build_features(
    camera_shapes: dict[str, tuple[int, int, int]],
    use_videos: bool,
) -> dict:
    """LeRobot v2 feature spec for the 10-dim FR3 layout + N cameras."""
    features: dict = {}
    for cam_name, shape in camera_shapes.items():
        features[_image_key(cam_name)] = {
            "dtype": "video" if use_videos else "image",
            "shape": tuple(shape),  # (H, W, 3) uint8
            "names": ["height", "width", "channels"],
        }
    features[STATE_KEY] = {
        "dtype": "float32",
        "shape": (10,),
        "names": STATE_AXIS_NAMES,
    }
    features[ACTION_KEY] = {
        "dtype": "float32",
        "shape": (10,),
        "names": ACTION_AXIS_NAMES,
    }
    return features


def _validate_existing_info(
    info_path: Path,
    camera_shapes: dict[str, tuple[int, int, int]],
    fps: int,
    use_videos: bool,
) -> None:
    """Sanity-check an existing dataset before resuming."""
    with info_path.open() as f:
        info = json.load(f)

    existing_fps = int(info.get("fps", -1))
    if existing_fps != fps:
        raise ValueError(
            f"Existing dataset fps={existing_fps} does not match requested "
            f"fps={fps}. Pick a different repo_id or change collect.fps to match."
        )

    features = info.get("features", {})
    expected_dtype = "video" if use_videos else "image"

    for cam_name, cam_shape in camera_shapes.items():
        key = _image_key(cam_name)
        if key not in features:
            raise ValueError(
                f"Existing dataset is missing expected image key '{key}'. "
                f"Available features: {sorted(features.keys())}"
            )

        existing_dtype = features[key].get("dtype")
        if existing_dtype != expected_dtype:
            flag = "true" if existing_dtype == "video" else "false"
            raise ValueError(
                f"Existing dataset stores '{key}' with dtype='{existing_dtype}' "
                f"but collect.use_videos={use_videos} would write as "
                f"'{expected_dtype}'. Set collect.use_videos={flag} to match, "
                f"or pick a new repo_id."
            )

        existing_shape = tuple(features[key].get("shape", ()))
        if existing_shape != tuple(cam_shape):
            raise ValueError(
                f"Camera resolution mismatch: existing dataset has '{key}' "
                f"shape={existing_shape}, but the current camera produces "
                f"shape={tuple(cam_shape)}. Reconfigure the camera or pick a "
                f"new repo_id."
            )

    for key, expected_shape in ((STATE_KEY, (10,)), (ACTION_KEY, (10,))):
        if key not in features:
            raise ValueError(
                f"Existing dataset is missing required feature '{key}'."
            )
        shape = tuple(features[key].get("shape", ()))
        if shape != expected_shape:
            raise ValueError(
                f"Feature '{key}' shape mismatch: existing={shape}, "
                f"expected={expected_shape}. The 10-dim FR3 layout is fixed "
                f"in real/collect_sft_data.py."
            )


def _open_or_create_dataset(
    *,
    dataset_root: Path,
    repo_id: str,
    fps: int,
    camera_shapes: dict[str, tuple[int, int, int]],
    use_videos: bool,
    image_writer_threads: int,
) -> tuple[LeRobotDataset, bool]:
    """Open an existing LeRobot dataset for append, or create a fresh one."""
    ds_root = dataset_root / repo_id
    info_path = ds_root / "meta" / "info.json"

    if info_path.exists():
        _validate_existing_info(info_path, camera_shapes, fps, use_videos)
        dataset = LeRobotDataset.__new__(LeRobotDataset)
        dataset.meta = LeRobotDatasetMetadata(repo_id=repo_id, root=ds_root)
        dataset.repo_id = dataset.meta.repo_id
        dataset.root = dataset.meta.root
        dataset.revision = None
        dataset.tolerance_s = 1e-4
        dataset.image_writer = None
        dataset.batch_encoding_size = 1
        dataset.episodes_since_last_encoding = 0
        dataset.episodes = None
        dataset.hf_dataset = dataset.create_hf_dataset()
        dataset.image_transforms = None
        dataset.delta_timestamps = None
        dataset.delta_indices = None
        dataset.episode_data_index = None
        dataset.video_backend = get_safe_default_codec()
        if image_writer_threads:
            dataset.start_image_writer(num_processes=0, num_threads=image_writer_threads)
        dataset.episode_buffer = dataset.create_episode_buffer()
        return dataset, True

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=_build_features(camera_shapes, use_videos),
        root=ds_root,
        robot_type="fr3_agilex",
        use_videos=use_videos,
        image_writer_threads=image_writer_threads,
    )
    return dataset, False


def _print_help(state: _State) -> None:
    keys = {
        _State.IDLE: "[s] start episode   [q] quit",
        _State.RECORDING: "[e] end+save   [r] discard+reset   [q] quit",
    }.get(state, "")
    print(f"\r[{state.value}] {keys}", end="", flush=True)


def _reset_desired_pose(env: FR3RealEnv) -> np.ndarray:
    """Re-read the current EE pose as the new teleop setpoint."""
    return env.controller.get_ee_pose()


def run(cfg: DictConfig) -> None:
    cc = cfg.collect
    dataset_root = Path(cc.dataset_root).expanduser().resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    task_prompt = str(cc.task_prompt)

    env = FR3RealEnv(cfg, task=task_prompt)
    env.start()

    # Reset once up front so get_ee_pose() is a sensible initial setpoint.
    obs, _ = env.reset()
    camera_shapes: dict[str, tuple[int, int, int]] = {
        name: img.shape for name, img in obs["images"].items()
    }
    for name, shape in camera_shapes.items():
        print(f"[collect] Camera '{name}' streaming at {shape}")
    print(f"[collect] Control freq = {cfg.robot.freq} Hz; dataset fps = {cc.fps}.")

    dataset, was_resumed = _open_or_create_dataset(
        dataset_root=dataset_root,
        repo_id=str(cc.repo_id),
        fps=int(cc.fps),
        camera_shapes=camera_shapes,
        use_videos=bool(cc.use_videos),
        image_writer_threads=int(cc.image_writer_threads),
    )

    sm = SpaceMouseController(freq=float(cc.spacemouse_freq))
    gripper = _GripperToggle(
        open_width=float(cc.gripper_open_width),
        closed_width=float(cc.gripper_closed_width),
    )

    keys = _KeyboardListener()
    keys.start()

    state = _State.IDLE
    desired_ee_pose = _reset_desired_pose(env)
    episode_frames: list[dict[str, Any]] = []
    total_episodes = int(dataset.meta.total_episodes)
    session_episodes = 0

    verb = "Resumed" if was_resumed else "Created"
    print(
        f"[collect] Task: '{task_prompt}'\n"
        f"[collect] {verb} dataset at {dataset_root / str(cc.repo_id)}\n"
        f"[collect] Existing episodes in dataset: {total_episodes}"
    )

    # Print per-task episode counts on load
    if dataset.meta.episodes:
        task_counts: dict[str, int] = {}
        for ep in dataset.meta.episodes.values():
            for t in ep.get("tasks", []):
                task_counts[t] = task_counts.get(t, 0) + 1
        for t, count in sorted(task_counts.items()):
            print(f"[collect]   {t}: {count} episodes")

    print("[collect] Use [s] to start the first episode.")
    _print_help(state)

    try:
        while state is not _State.QUITTING:
            key = keys.poll()
            if key == "q":
                state = _State.QUITTING
                break
            elif key == "s" and state is _State.IDLE:
                episode_frames = []
                desired_ee_pose = _reset_desired_pose(env)
                gripper.reset()
                state = _State.RECORDING
                print()  # newline after status bar
                print(f"[collect] Recording episode {total_episodes + 1}...")
                _print_help(state)
            elif key == "e" and state is _State.RECORDING:
                state = _State.SAVING
            elif key == "r" and state is _State.RECORDING:
                state = _State.DISCARDING

            if state is _State.IDLE:
                time.sleep(0.02)
                _print_help(state)
                continue

            if state is _State.RECORDING:
                delta = sm.homogeneous_delta
                desired_ee_pose[:3, 3] = desired_ee_pose[:3, 3] + delta[:3, 3]
                desired_ee_pose[:3, :3] = delta[:3, :3] @ desired_ee_pose[:3, :3]

                gripper_target = gripper.update(sm.buttons)

                action_10d = homogeneous_to_pose_10d(desired_ee_pose, gripper_target)

                frame: dict[str, Any] = {
                    STATE_KEY: np.asarray(obs["state"], dtype=np.float32),
                    ACTION_KEY: np.asarray(action_10d, dtype=np.float32),
                }
                for cam_name, cam_img in obs["images"].items():
                    frame[_image_key(cam_name)] = np.ascontiguousarray(
                        cam_img, dtype=np.uint8
                    )
                # Only add frame if the qpos has changed: 
                cur_state_10d = env._controller.get_state_10d()
                if not np.allclose(env._last_state_10d, cur_state_10d, atol=1e-4):
                    episode_frames.append(frame)
                
                env._last_state_10d = cur_state_10d.copy()
                obs, _, _, _, _ = env.step(action_10d)
                continue

            
            # Reset robot first, then flush buffered frames to disk
            if state is _State.SAVING:
                print()
                saved_frames = episode_frames
                episode_frames = []
                env.reset()
                desired_ee_pose = _reset_desired_pose(env)
                if len(saved_frames) == 0:
                    print("[collect] Empty episode — nothing to save.")
                else:
                    in_progress_idx = total_episodes + 1
                    print(
                        f"[collect] Saving episode {in_progress_idx} "
                        f"with {len(saved_frames)} frames..."
                    )
                    for frame in saved_frames:
                        dataset.add_frame(frame, task=task_prompt)
                    print("Saving episode...")
                    dataset.save_episode()
                    total_episodes = int(dataset.meta.total_episodes)
                    session_episodes += 1
                    task_ep_count = sum(
                        1 for ep in dataset.meta.episodes.values()
                        if task_prompt in ep.get("tasks", [])
                    )
                    print(
                        f"[collect] Episode {total_episodes} saved "
                        f"(session: {session_episodes})."
                    )
                    print(f"[collect] {task_prompt}: {task_ep_count} episodes")
                state = _State.IDLE
                _print_help(state)
                continue

            #drop buffer, reset robot, back to IDLE.
            if state is _State.DISCARDING:
                print()
                print(
                    f"[collect] Discarding {len(episode_frames)} frames and resetting."
                )
                episode_frames = []
                if dataset.episode_buffer is not None and dataset.episode_buffer["size"] > 0:
                    dataset.clear_episode_buffer()
                env.reset()
                desired_ee_pose = _reset_desired_pose(env)
                state = _State.IDLE
                _print_help(state)
                continue

    except KeyboardInterrupt:
        print("\n[collect] Interrupted by user.")
    finally:
        print()
        print("[collect] Cleaning up...")
        try:
            keys.stop()
        except Exception as e:
            print(f"[collect] keyboard cleanup error: {e}")
        try:
            sm.close()
        except Exception as e:
            print(f"[collect] spacemouse cleanup error: {e}")
        try:
            env.close()
        except Exception as e:
            print(f"[collect] env cleanup error: {e}")
        try:
            if dataset.episode_buffer is not None and dataset.episode_buffer["size"] > 0:
                dataset.clear_episode_buffer()
        except Exception as e:
            print(f"[collect] dataset finalize error: {e}")
        try:
            final_total = int(dataset.meta.total_episodes)
        except Exception:
            final_total = total_episodes
        print(
            f"[collect] Saved {session_episodes} episode(s) this session "
            f"({final_total} total in dataset) -> {dataset_root / str(cc.repo_id)}"
        )


@hydra.main(version_base=None, config_path="config", config_name="collect")
def _outer_main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    _outer_main()
