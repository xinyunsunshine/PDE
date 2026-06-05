"""Threaded SpaceMouse reader producing task-space delta actions."""

from __future__ import annotations

import threading
import time

import numpy as np
import pyspacemouse
from scipy.spatial.transform import Rotation


class SpaceMouseController:
    """Threaded 3Dconnexion SpaceMouse reader.

    Samples the device at ``freq`` Hz in a daemon thread and exposes the
    latest per-tick delta as a 4x4 homogeneous transform. The translation and
    rotation components are clipped to conservative magnitudes so a single
    tick can be applied directly to a current EE pose every control cycle.

    """

    def __init__(self, freq: float = 100.0):
        # pyspacemouse 2.x returns a SpaceMouseDevice from open() (1.x
        # returned a bool and exposed read()/close() at module level).
        self._device = pyspacemouse.open()
        if self._device is None:
            raise RuntimeError("Failed to open spacemouse device")
        self._started = False
        self._freq = freq
        self._lock = threading.Lock()

        # Default state: identity transform (no delta).
        self._state = np.eye(4)
        self._buttons: tuple[int, ...] = tuple()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self._started = True
        while self._started:
            state = self._device.read()
            translation = np.clip(
                np.array([state.x, state.y, -state.z]) * -0.005,
                -0.005,
                0.005,
            )
            rot = np.clip(
                np.array([-0.4 * state.roll, -0.4 * state.pitch, -state.yaw]) * 1.5,
                -1,
                1,
            )
            rot_delta = Rotation.from_euler("XYZ", rot, degrees=True).as_matrix()

            homogeneous_delta = np.eye(4)
            homogeneous_delta[:3, :3] = rot_delta
            homogeneous_delta[:3, 3] = translation

            buttons = tuple(state.buttons) if hasattr(state, "buttons") else tuple()

            with self._lock:
                self._state = homogeneous_delta
                self._buttons = buttons

            time.sleep(1.0 / self._freq)

    @property
    def homogeneous_delta(self) -> np.ndarray:
        """Latest per-tick delta as a 4x4 homogeneous transform."""
        with self._lock:
            return self._state.copy()

    @property
    def buttons(self) -> tuple[int, ...]:
        """Latest button state as a tuple of ints (1 = pressed)."""
        with self._lock:
            return self._buttons

    def close(self) -> None:
        self._started = False
        self._thread.join(timeout=1.0)
        self._device.close()
