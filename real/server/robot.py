from omegaconf import DictConfig
from aiofranka import RobotInterface, FrankaController
import numpy as np
import time
from piper_sdk import C_PiperInterface_V2


_FACTORY_SENTINEL = object()


class AgileXGripper:
    def __init__(self, cfg: DictConfig):
        try:
            self._gripper = C_PiperInterface_V2(cfg.gripper.CAN)
            self._gripper.ConnectPort()
        except Exception as e:
            raise RuntimeError(f"[AgileXGripper] Failed to connect to gripper via CAN interface: {e}")

        self._torque_nm = cfg.gripper.torque
        print("[AgileXGripper] Connected to gripper")

    def step(self, width_m: float):
        width_m = np.clip(width_m, 0, 0.069) # avoid crashing the gripper
        self._gripper.GripperCtrl(int(width_m * 1_000_000), int(self._torque_nm * 1_000), 0x01, 0)

    def get_state(self) -> float:
        return self._gripper.GetArmGripperMsgs().gripper_state.grippers_angle / 1_000_000.0

if __name__ == "__main__":
    gripper = AgileXGripper(DictConfig(
        {
            "gripper": {"CAN": "can_gripper", "torque": 1}
        }
    ))
    # gripper.move(0.08)
    print(gripper.get_state())
    while True:
        time.sleep(0.25)
        gripper.step(0.069)
        time.sleep(0.25)
        print(gripper.get_state())
        gripper.step(0.059)


class ImpedanceController:
    def __init__(self, cfg: DictConfig, _sentinel=None):
        """
        Args:
            cfg (DictConfig): Configuration for the robot.

        Note: The gripper server must be running in a separate terminal before
        creating this controller. Start it with:
            python -m real.gripper_process --ip <robot_ip>
        """
        if _sentinel is not _FACTORY_SENTINEL:
            raise ValueError("This class is not meant to be instantiated directly. Use the create method instead.")
        self.cfg = cfg
        self._robot = RobotInterface(cfg.ip)
        self._controller = FrankaController(self._robot)

        self._gripper = AgileXGripper(cfg)
        self._home_qpos = np.array(cfg.home_qpos)

    @classmethod
    async def create(cls, cfg: DictConfig):
        """Setup the robot interface and start the low level controller"""
        instance = cls(cfg, _sentinel=_FACTORY_SENTINEL)
        await instance._setup()
        return instance

    async def _setup(self):
        await self._controller.start()
        await self._controller.test_connection()

        await self.reset()

        # Set controller parameters
        self._controller.switch("impedance")
        self._controller.kp = np.asarray(self.cfg.kp)
        self._controller.kd = np.asarray(self.cfg.kd)
        self._controller.set_freq(self.cfg.freq)

    async def reset(self):
        """Reset the robot to the home position"""
        # TODO: i(@jmarangola) implement randomization
        self._gripper.step(0.069)
        self._controller.switch("osc")
        self._controller.set_freq(50)
        await self._controller.move(self._home_qpos.tolist())
        self._controller.switch("impedance")
        self._controller.set_freq(self.cfg.freq)


    async def set_delta(self, delta: np.ndarray):
        """Set the joint delta positions of the robot. Expects gripper as absolute, denormalized to [0, 0.070]"""
        self._gripper.step(delta[-1] * 2)
        current_qpos = self._controller.state['qpos']
        await self._controller.set("q_desired", delta[:7] + current_qpos)

    # async def set_target(self, qpos: np.ndarray):
    #     if self._prev_gripper_unnorm_action_targ is None:
    #         self._prev_gripper_unnorm_action_targ = qpos[-1]

    #     gripper_command_change = abs(qpos[-1] - self._prev_gripper_unnorm_action_targ)
    #     import pdb;pdb.set_trace()
    #     if gripper_command_change > self._gripper_target_change_threshold:
    #         self.gripper.move(qpos[-1], self.cfg.gripper.speed, wait=True)
    #         self._prev_gripper_unnorm_action_targ = qpos[-1]
    #     await self._controller.set("q_desired", qpos[:7])

    def set_home_qpos(self, qpos: np.ndarray):
        self._home_qpos = qpos

    @property
    def qpos(self):
        gripper_qpos = np.array([self._gripper.get_state()] * 2) / 2
        return np.concatenate([self._controller.state['qpos'].copy(), gripper_qpos])

    @property
    def qvel(self):
        return self._controller.state['qvel'].copy()

    @property
    def ee_pose(self):
        return self._controller.state['ee'].copy()

    @property
    def q_d(self):
        return self._controller.state['q_d'].copy()

    @property
    def tau_last(self):
        return self._controller.state['last_torque'].copy()

    async def stop(self):
        # self.gripper.close()
        await self._controller.stop()


class OSCController:
    """Operational Space Controller for end-effector control"""

    def __init__(self, cfg: DictConfig, _sentinel=None):
        """
        Args:
            cfg (DictConfig): Configuration for the robot.
        """
        if _sentinel is not _FACTORY_SENTINEL:
            raise ValueError("This class is not meant to be instantiated directly. Use the create method instead.")
        self.cfg = cfg
        self._robot = RobotInterface(cfg.ip)
        self._controller = FrankaController(self._robot)
        self._home_qpos = np.array(cfg.home_qpos)

    @classmethod
    async def create(cls, cfg: DictConfig):
        """Setup the robot interface and start the low level controller"""
        instance = cls(cfg, _sentinel=_FACTORY_SENTINEL)
        await instance._setup()
        return instance

    async def _setup(self):
        # Start the controller and test the connection
        await self._controller.start()
        await self._controller.test_connection()

        # Reset the robot to the home position
        await self.reset()

        # Set controller parameters. Pacing (set_freq) is done on the ZMQ
        # server boundary using cfg.robot.freq — not on FrankaController,
        # because some aiofranka builds in this workspace do not expose
        # set_freq at all (the `vla` env ships a minimal shim).
        self._controller.switch("osc")
        self._controller.ee_kp = np.asarray(self.cfg.ee_kp)
        self._controller.ee_kd = np.asarray(self.cfg.ee_kd)

    async def reset(self):
        """Reset the robot to the home position"""
        await self._controller.move(self._home_qpos.tolist())
        # move() switches to impedance mode, so switch back to OSC
        self._controller.switch("osc")
        self._controller.ee_kp = np.asarray(self.cfg.ee_kp)
        self._controller.ee_kd = np.asarray(self.cfg.ee_kd)

    async def set_ee_pose(self, pose: np.ndarray):
        """Set the desired end-effector pose (4x4 transformation matrix).

        The ``vla`` env's aiofranka shim does not expose
        ``FrankaController.set()``, so we write ``ee_desired`` directly
        under the controller's state_lock. The 1 kHz loop picks it up on
        its next tick via the OSC step.
        """
        with self._controller.state_lock:
            self._controller.ee_desired = np.asarray(pose, dtype=np.float64)


    # State comes from ``RobotInterface.state`` (a fresh dict read off the
    # live pylibfranka snapshot each call). The minimal aiofranka in the
    # ``vla`` env does NOT cache this on ``FrankaController.state`` — that
    # attribute doesn't exist, so the fuller-API accesses would throw
    # ``AttributeError`` inside the ZMQ handler and surface as a
    # mis-labelled "timeout" on the client side.
    @property
    def qpos(self):
        return self._controller.robot.state['qpos'].copy()

    @property
    def qvel(self):
        return self._controller.robot.state['qvel'].copy()

    @property
    def ee_pose(self):
        return self._controller.robot.state['ee'].copy()

    @property
    def initial_ee(self):
        return self._controller.initial_ee.copy()

    @property
    def tau_last(self):
        return self._controller.robot.state['last_torque'].copy()

    async def stop(self):
        await self._controller.stop()
