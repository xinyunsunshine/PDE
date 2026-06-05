import hydra
import time
import numpy as np
import cv2
import pinocchio as pin
import pupil_apriltags as apriltags
from real.perception.rs_device import MultiRSDevice
from omegaconf import OmegaConf, DictConfig
from scipy.spatial.transform import Rotation
from piper_sdk import *
from real.scripts.agilex_read import _AGILEX_FACTOR
from real.real_util import AgileXPositionController
from real.controller import FR3Controller
from real.spacemouse import SpaceMouseController

"""
Camera calibration tool (spacemouse teleop + AprilTag PnP).

Run directly — no separate server process is needed because
``FR3Controller`` spawns the aiofranka backend itself:

    python -m real.camera_calibration
"""

SQUARE_HALF_LENGTH = 0.050 # 50 Centimeters for 100x100x6mm target (x, y, z)
CORNER_Xi_TAG_FRAME = SQUARE_HALF_LENGTH * np.array( 
    [
        [-1, 1, 0], 
        [1, 1, 0], 
        [1, -1, 0], 
        [-1, -1, 0]
    ]
)


def main(cfg: DictConfig):
    """Camera calibration with spacemouse teleoperation + AprilTag PnP.

    Uses ``FR3Controller`` directly — the legacy ZMQ ``osc_server`` path is
    gone. The controller owns the aiofranka backend subprocess and exposes
    a synchronous OSC-mode API (see ``real/controller.py``).
    """
    print("[Calibration] Starting FR3 controller...")
    robot = FR3Controller(cfg)
    robot.start()

    agilex = AgileXPositionController()
    agilex.move(np.asarray(cfg.agilex_qpos))

    # Initialize the spacemouse controller
    print("[Calibration] Initializing spacemouse...")
    sm = SpaceMouseController(freq=100)

    # Reset robot to home position
    print("[Calibration] Resetting robot to home...")
    robot.reset()

    current_ee_pose = robot.get_ee_pose()
    print(f"[Calibration] Initial EE pose:\n{current_ee_pose}")

    # Initialize the pinocchio model for forward kinematics
    model = pin.buildModelFromUrdf("./real/fr3_calibration.urdf")
    data = model.createData()
    calibration_target_fid = model.getFrameId("calibration_target_frame")

    translations, rotations = [], []
    with MultiRSDevice(OmegaConf.create({"calibration": cfg.calibration})) as cal_cam: 
        # Get the camera factory calibrated intrinsics matrix, K
        K = cal_cam._rget_device("calibration").K
        D = cal_cam._rget_device("calibration").distortion_coefficients
        detector = apriltags.Detector(families="tag36h11")

        try:
            while True:
                frame = cal_cam.get_frames()["calibration"]
                if frame:
                    gray = cv2.cvtColor(frame.color, cv2.COLOR_RGB2GRAY)
    
                    # Get current joint positions for FK
                    qpos = robot.get_qpos()
                    # if qpos is not None:
                    # print("qpos:", qpos)

                    detections = detector.detect(gray)

                    if detections:
                        det = detections[0]
                        for i, corner in enumerate(det.corners):
                            cv2.circle(frame.color, (int(corner[0] + 5), int(corner[1] + 5)), 5, (0, 0, 255), -1)
                            cv2.putText(frame.color, str(i), (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        # Solve PnP to get the pose of the tag
                        success, rvec, tvec = cv2.solvePnP(
                            CORNER_Xi_TAG_FRAME,
                            det.corners[[3, 2, 1, 0]],
                            K,
                            D,
                            flags=cv2.SOLVEPNP_IPPE_SQUARE,                         
                        )
                        
                        # Visualize the pose with 3D axes (X=red, Y=green, Z=blue)
                        if success:
                            # Rotate the rvec from perspective-n-point to urdf tag frame convention 
                            rvec = (
                                Rotation.from_rotvec(rvec.flatten()) * Rotation.from_euler('zyx', [np.pi, 0, np.pi / 2], degrees=False)
                            ).as_rotvec()
                            axis_length = SQUARE_HALF_LENGTH  # Draw axes same size as tag
                            cv2.drawFrameAxes(frame.color, K, D, rvec, tvec, axis_length)

                            # Compute the pose of the tag in the world frame
                            if qpos is not None:
                                pin.forwardKinematics(model, data, qpos)
                                pin.updateFramePlacements(model, data)
                                world_T_tag = data.oMf[calibration_target_fid].homogeneous

                                cam_T_tag = np.eye(4)
                                cam_T_tag[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
                                cam_T_tag[:3, 3] = tvec.flatten()

                                # Compute the pose of camera in the world (base) frame 
                                world_T_cam = world_T_tag @ np.linalg.inv(cam_T_tag)
                    
                    cv2.imshow("Calibration", cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR))
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord('q'):
                        break
                    elif k == ord('r'): 
                        # Record the current trans and rot of world_T_cam
                        print(f"[Recorded] Trans: {world_T_cam[:3, 3]}, Rot: {Rotation.from_matrix(world_T_cam[:3, :3]).as_euler('XYZ', degrees=True)}")
                        translations.append(world_T_cam[:3, 3])
                        rotations.append(Rotation.from_matrix(world_T_cam[:3, :3]))

                # Apply spacemouse delta to current EE pose
                delta = sm.homogeneous_delta

                # Apply translation to end-effector pose in base frame
                current_ee_pose[:3, 3] += delta[:3, 3]

                # Apply global rotation to end-effector orientation in the base frame
                current_ee_pose[:3, :3] = delta[:3, :3] @ current_ee_pose[:3, :3]

                # Send new desired pose to robot. set_ee_pose is non-blocking
                # (sub-ms ZMQ write-through); wait_for_next_tick enforces the
                # control rate.
                robot.set_ee_pose(current_ee_pose)
                robot.wait_for_next_tick()

        except KeyboardInterrupt:
            print("\n[Calibration] Interrupted by user")
        finally:
            print("[Calibration] Cleaning up...")
            sm.close()
            robot.close()
            agilex.move(np.zeros(6))

            # Compute the calibration data and save to file
            trans_median = np.median(np.stack(translations, axis=0), axis=0)
            final_rot = Rotation.concatenate(rotations).mean().as_matrix()
            save_filename = f"calibration_data_{time.strftime('%Y%m%d_%H%M%S')}.npz"
            np.savez(
                save_filename, 
                cam_translation_inbase_frame=trans_median, 
                cam_rotation_inbase_frame=final_rot, 
                N=len(translations), 
                agilex_qpos=np.asarray(cfg.agilex_qpos),
                K=K,
                D=D,
            )
            print(f"[Calibration] Saved calibration data to: {save_filename}")
            cv2.destroyAllWindows()
            print("[Calibration] Completed calibration, exiting...")


@hydra.main(version_base=None, config_path="config", config_name="calibration")
def _outer_main(cfg: DictConfig): 
    main(cfg)


if __name__ == "__main__":
    _outer_main()