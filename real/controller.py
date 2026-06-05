"""Synchronous FR3 controller: ZMQ-client to OSC server + local AgileX gripper.

Arm control (4x4 EE pose) is dispatched via ``OSCRobotClient`` to the OSC
server process (``real/server/osc_server.py``), which owns the in-process
aiofranka ``FrankaController`` and the 1 kHz libfranka loop. The gripper
runs client-side over CAN (``AgileXGripper``) so CAN traffic never competes
with the RT control thread for CPU time.

State dict returned by the server mirrors aiofranka's layout:
    qpos: (7,), ee: (4, 4).

Public API is unchanged from the previous in-process controller so
``real/env.py``, ``real/deploy.py``, and ``real/collect_sft_data.py`` see
the same ``FR3Controller.start/reset/step(action_10d)/get_state_10d/…``.
"""

from __future__ import annotations

import time

import numpy as np
from omegaconf import DictConfig

from real.rotation_utils import homogeneous_to_pose_10d, pose_10d_to_homogeneous
from real.server.osc_server import OSCRobotClient, connect_to_osc_server
from real.server.robot import AgileXGripper


class FR3Controller:
    """Synchronous controller: ZMQ OSC client + local AgileX gripper.

    Lifecycle:
        ctrl = FR3Controller(cfg)
        ctrl.start()                 # connect to OSC server, open gripper if any
        ctrl.reset()                 # server homes the arm, gripper opens
        ctrl.step(action_10d)        # one control tick (dispatch + pace to freq)
        ctrl.get_state_10d()         # read 10-dim state
        ctrl.close()                 # disconnect ZMQ client

    Config keys expected (real/config/osc.yaml + task group):
        robot.ip                    : Franka IP (unused client-side; server uses it)
        robot.freq                  : control frequency (Hz)
        zmq.url                     : e.g. ``tcp://localhost:5555``
        zmq.timeout_ms              : REQ socket send/recv timeout
        gripper.CAN, gripper.torque : AgileX gripper config (optional)
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._client: OSCRobotClient | None = None
        self._gripper: AgileXGripper | None = None
        self._freq = float(cfg.robot.freq)
        self._dt = 1.0 / self._freq

        # Drift-free pacing anchor for ``wait_for_next_tick``. Reset to None
        # whenever the schedule should be re-seeded (start, reset).
        self._next_deadline: float | None = None


    def start(self) -> None:
        """Connect to the OSC server, construct gripper if configured."""
        zmq_url = str(self.cfg.zmq.url)
        timeout_ms = int(self.cfg.zmq.get("timeout_ms", 5000))
        self._client = connect_to_osc_server(zmq_url, timeout_ms=timeout_ms)

        # Gripper is optional — e.g. the camera calibration path uses the
        # controller for EE teleop without needing a gripper. Only construct
        # one if the config has a ``gripper:`` section.
        if "gripper" in self.cfg:
            self._gripper = AgileXGripper(self.cfg)
        else:
            self._gripper = None

        # Re-seed pacing on start; the first wait_for_next_tick anchors the schedule.
        self._next_deadline = None

    def reset(self) -> None:
        """Open gripper (if present) and ask the server to home the arm.

        The server owns the impedance-mode home move and the switch back to
        OSC; we only open the gripper locally and delegate the arm move.
        """
        assert self._client is not None, "FR3Controller.start() must be called first"

        if self._gripper is not None:
            self._gripper.step(0.069)  # fully open

        t0 = time.perf_counter()
        ok = self._client.reset()
        if not ok:
            raise RuntimeError(
                "OSC server rejected reset (timeout or error). Check that the "
                "server is running and the robot is unlocked in the Desk UI."
            )
        print(f"[Reset] Home move completed in {(time.perf_counter() - t0) * 1e3:.1f} ms")

        # Re-seed pacing after the home move; any prior deadline is stale.
        self._next_deadline = None

    def close(self) -> None:
        """Disconnect the ZMQ client. Server keeps running for the next episode."""
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception as e:
                print(f"[FR3Controller] Error during disconnect: {e}")
            self._client = None
        self._gripper = None


    def get_ee_pose(self) -> np.ndarray:
        """Current measured end-effector pose as a 4x4 homogeneous transform."""
        assert self._client is not None
        pose = self._client.get_ee_pose()
        if pose is None:
            raise RuntimeError(
                "OSC server did not return ee_pose. Check the server's terminal "
                "for a '[OSCRobotClient] Server returned error: …' line (exception "
                "inside the handler) or a timeout message (server hung / libfranka "
                "loop died). Re-activate FCI + relaunch real/run_osc_server.sh."
            )
        return pose

    def get_qpos(self) -> np.ndarray:
        """Current measured joint positions (7,)."""
        assert self._client is not None
        qpos = self._client.get_qpos()
        if qpos is None:
            raise RuntimeError(
                "OSC server did not return qpos. Check the server's terminal "
                "for a '[OSCRobotClient] Server returned error: …' line or a "
                "timeout message."
            )
        return qpos

    def get_gripper_width(self) -> float:
        """Current gripper width in meters."""
        assert self._gripper is not None
        return float(self._gripper.get_state())

    def get_state_10d(self) -> np.ndarray:
        """Pack measured EE pose + gripper width into the 10-dim layout."""
        return homogeneous_to_pose_10d(self.get_ee_pose(), self.get_gripper_width())

    def set_ee_pose(self, pose: np.ndarray) -> None:
        """Dispatch a desired EE pose via ZMQ. Round-trip is sub-ms on localhost.

        The server forwards the pose to aiofranka's ``ee_desired`` slot on
        every call; the 1 kHz C++ loop picks up the latest value within one
        tick. Callers that need control-rate enforcement must follow up with
        ``wait_for_next_tick()``, or use ``step()`` which does both.
        """
        assert self._client is not None
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError(f"pose must be 4x4, got {pose.shape}")
        if not self._client.step(pose):
            raise RuntimeError("OSC server rejected set_ee_pose (timeout or error).")

    def wait_for_next_tick(self) -> None:
        """Drift-free sleep to maintain self._freq.

        Advances an internal deadline by exactly self._dt per call so
        transient jitter does not accumulate. Re-anchors if the caller has
        fallen more than one full period behind (e.g. after a long obs read
        or a stalled camera frame), so a single hiccup does not cause a
        catch-up burst on subsequent calls.
        """
        time.sleep(self._dt)
        # now = time.perf_counter()
        # if self._next_deadline is None:
        #     self._next_deadline = now + self._dt
        #     sleep_time = self._next_deadline - time.perf_counter()
        #     if sleep_time > 0:
        #         time.sleep(sleep_time)
        #     return

        # self._next_deadline += self._dt
        # sleep_time = self._next_deadline - time.perf_counter()
        # if sleep_time > 0:
        #     time.sleep(sleep_time)
        # elif sleep_time < -self._dt:
        #     # Fell more than one period behind — re-anchor and keep going.
        #     print("WARN: Fell more than one period behind")
        #     self._next_deadline = time.perf_counter() + self._dt

    def set_gripper(self, width_m: float) -> None:
        """Send a desired gripper width in meters."""
        assert self._gripper is not None
        self._gripper.step(float(width_m))

    def step(self, action_10d: np.ndarray) -> None:
        """Unpack a 10-dim absolute action, dispatch it, and pace to ``self._freq``.

        Dispatch is sub-ms (ZMQ round-trip + CAN write); the rest of the
        control period is spent in ``wait_for_next_tick``. The sleep
        releases the GIL, so background threads (e.g. a VLA inference
        worker) make progress concurrently.
        """
        pose, gripper_width = pose_10d_to_homogeneous(action_10d)
        self.set_gripper(gripper_width)
        self.set_ee_pose(pose)
        self.wait_for_next_tick()

    # Context mgr suggie
    def __enter__(self) -> "FR3Controller":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
