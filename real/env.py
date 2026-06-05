"""Gym-style FR3 real-world environment for pi0 VLA data collection and deploy.

Observation (dict):
    image : (H, W, 3) uint8 — RGB from the "global" RealSense camera
    state : (10,) float32  — [xyz(3), rot6d(6), gripper_width(1)]
    task  : str            — natural-language task description

Action:
    (10,) float32 — same layout as state, interpreted as an *absolute*
    desired EE pose + gripper width. ``env.step`` dispatches the setpoint
    in sub-ms (ZMQ write-through to the aiofranka server) and then sleeps
    in ``FR3Controller.wait_for_next_tick`` to the next control deadline,
    so callers should not add their own sleep.

This wrapper is intentionally thin — it owns:
  * FR3Controller    (aiofranka + AgileX gripper)
  * MultiRSDevice    (RealSense cameras)

and nothing policy- or checkpoint-specific. Image preprocessing (resize,
center crop, mean subtract) is left to the downstream model pipeline because
pi0's data loader resizes at training time.
"""
from __future__ import annotations

import cv2
import numpy as np

from typing import Any
from omegaconf import DictConfig

from real.controller import FR3Controller
from real.perception.rs_device import MultiRSDevice


# Name of the primary RGB camera inside cfg.task.perception (used by deploy
# and anywhere a single canonical image is needed).
PRIMARY_CAMERA_NAME = "global"


class FR3RealEnv:
    """Minimal gym-style wrapper around the FR3 + RealSense stack."""

    def __init__(self, cfg: DictConfig, task: str):
        """
        Args:
            cfg: Full hydra config. Must expose ``cfg.robot`` (FR3Controller)
                and ``cfg.task.perception`` (MultiRSDevice).
            task: Natural-language task description for the policy prompt.
        """
        self.cfg = cfg
        self.task = task

        self._controller = FR3Controller(cfg)
        self._cameras = MultiRSDevice(cfg.task.perception)
        self._resize_cfg: dict[str, tuple[int, int]] | None = cfg.task.perception.get("resize", None)
        self._started = False


    def start(self) -> None:
        if self._started:
            return
        self._controller.start()
        self._cameras.start()
        self._started = True

    def close(self) -> None:
        if not self._started:
            return
        try:
            self._controller.close()
        finally:
            self._cameras.stop()
            self._started = False

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the robot to home and return the first observation."""
        del seed, options
        if not self._started:
            self.start()
        self._controller.reset()
        self._last_state_10d = self._controller.get_state_10d()
        return self._get_obs(), {}

    def step(self, action_10d: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Apply a 10-dim absolute EE+gripper action and return next obs.

        Returns the standard gym 5-tuple. Reward, termination, and truncation
        are all zero/False — episode management is external (teleop keyboard
        or deploy-side step counter).
        """
        self._controller.step(action_10d)
        return self._get_obs(), 0.0, False, False, {}


    @property
    def controller(self) -> FR3Controller:
        return self._controller

    @property
    def cameras(self) -> MultiRSDevice:
        return self._cameras

    def _get_obs(self) -> dict[str, Any]:
        all_frames = self._cameras.get_frames()
        images: dict[str, np.ndarray] = {}
        for name, frame in all_frames.items():
            if frame is None:
                raise RuntimeError(
                    f"No frame from camera '{name}'. "
                    "Check that the camera is connected and warmed up."
                )
            images[name] = np.ascontiguousarray(frame.color, dtype=np.uint8)

        state = self._controller.get_state_10d()
        return {"images": images, "state": state, "task": self.task}

    # Context manager suggggie
    def __enter__(self) -> "FR3RealEnv":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
