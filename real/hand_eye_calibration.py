"""
Hand-eye calibration for the wrist-mounted D405 on the FR3 + AgileX gripper.

Solves AX = XB (cv2.calibrateHandEye) to compute gripper_base_T_cam: the rigid
transform from fr3_gripper_base to the D405 color optical frame.

Unlike real/camera_calibration.py (which assumes the tag is rigidly attached to
the EE and solves for a stationary base camera), this script assumes the tag is
stationary and solves for a wrist-mounted camera.

Run directly — no separate server process is needed because ``FR3Controller``
spawns the aiofranka backend itself:

    python -m real.hand_eye_calibration
"""

import time
from itertools import combinations
from pathlib import Path

import cv2
import hydra
import numpy as np
import pinocchio as pin
import pupil_apriltags as apriltags
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation

from real.camera_calibration import (
    CORNER_Xi_TAG_FRAME,
    SQUARE_HALF_LENGTH,
)
from real.controller import FR3Controller
from real.perception.rs_device import MultiRSDevice
from real.real_util import AgileXPositionController
from real.spacemouse import SpaceMouseController


URDF_PATH = (
    "external/ManiSkill/mani_skill/assets/robots/fr3/fr3_agilex_gripper_wristcam.urdf"
)

_METHOD_NAMES = {
    cv2.CALIB_HAND_EYE_TSAI: "TSAI",
    cv2.CALIB_HAND_EYE_PARK: "PARK",
    cv2.CALIB_HAND_EYE_HORAUD: "HORAUD",
    cv2.CALIB_HAND_EYE_ANDREFF: "ANDREFF",
    cv2.CALIB_HAND_EYE_DANIILIDIS: "DANIILIDIS",
}


def hand_eye_residual(R_g2b, t_g2b, R_t2c, t_t2c, R_X, t_X):
    """Residual on AX - XB across all pose pairs (from the calibration plan)."""
    X = np.eye(4)
    X[:3, :3] = R_X
    X[:3, 3] = np.asarray(t_X).flatten()
    rot_errs, trans_errs = [], []
    pairs = list(combinations(range(len(R_g2b)), 2))
    for i, j in pairs:
        Bi = np.eye(4); Bi[:3, :3] = R_g2b[i]; Bi[:3, 3] = np.asarray(t_g2b[i]).flatten()
        Bj = np.eye(4); Bj[:3, :3] = R_g2b[j]; Bj[:3, 3] = np.asarray(t_g2b[j]).flatten()
        A = np.linalg.inv(Bi) @ Bj

        Ci = np.eye(4); Ci[:3, :3] = R_t2c[i]; Ci[:3, 3] = np.asarray(t_t2c[i]).flatten()
        Cj = np.eye(4); Cj[:3, :3] = R_t2c[j]; Cj[:3, 3] = np.asarray(t_t2c[j]).flatten()
        B = Ci @ np.linalg.inv(Cj)

        residual = A @ X - X @ B
        rot_errs.append(np.linalg.norm(residual[:3, :3]))
        trans_errs.append(np.linalg.norm(residual[:3, 3]))
    return np.asarray(rot_errs), np.asarray(trans_errs), pairs


def _summarize_residuals(label: str, rot_errs: np.ndarray, trans_errs: np.ndarray) -> dict:
    def stats(a):
        return dict(
            mean=float(np.mean(a)),
            median=float(np.median(a)),
            max=float(np.max(a)),
            p95=float(np.percentile(a, 95)),
        )

    t = stats(trans_errs)
    r = stats(rot_errs)
    print(
        f"[{label:>10}]  trans (mm)  mean={t['mean']*1e3:7.3f}  "
        f"median={t['median']*1e3:7.3f}  p95={t['p95']*1e3:7.3f}  max={t['max']*1e3:7.3f}"
    )
    print(
        f"[{label:>10}]  rot (Frob)  mean={r['mean']:7.4f}  "
        f"median={r['median']:7.4f}  p95={r['p95']:7.4f}  max={r['max']:7.4f}"
    )
    return dict(
        trans_mean=t["mean"], trans_median=t["median"], trans_max=t["max"], trans_p95=t["p95"],
        rot_mean=r["mean"], rot_median=r["median"], rot_max=r["max"], rot_p95=r["p95"],
    )


def _solve_all_methods(R_g2b, t_g2b, R_t2c, t_t2c) -> dict:
    """Run all cv2 hand-eye methods and return dict method_name -> result."""
    results = {}
    for method_id, method_name in _METHOD_NAMES.items():
        try:
            R_X, t_X = cv2.calibrateHandEye(
                R_g2b, t_g2b, R_t2c, t_t2c, method=method_id
            )
        except cv2.error as e:
            print(f"[{method_name:>10}]  failed: {e}")
            continue
        rot_errs, trans_errs, pairs = hand_eye_residual(
            R_g2b, t_g2b, R_t2c, t_t2c, R_X, t_X
        )
        stats = _summarize_residuals(method_name, rot_errs, trans_errs)
        results[method_name] = dict(
            R=R_X,
            t=t_X,
            stats=stats,
            rot_errs=rot_errs,
            trans_errs=trans_errs,
            pairs=pairs,
        )
    return results


def _verify_rpy_convention():
    """URDF rpy is ROS REP-103 extrinsic XYZ, matching scipy uppercase 'XYZ'.
    Round-trip a non-degenerate angle as a sanity check."""
    test = np.array([0.1, 0.2, 0.3])
    R_test = Rotation.from_euler("XYZ", test).as_matrix()
    back = Rotation.from_matrix(R_test).as_euler("XYZ")
    if not np.allclose(back, test, atol=1e-9):
        raise RuntimeError(f"rpy round-trip failed: {back} vs {test}")


def _emit_urdf_block(R_X: np.ndarray, t_X: np.ndarray) -> str:
    """Return the drop-in URDF XML block for wrist_camera_origin_joint.

    The calibration solves gripper_base_T_cam in OpenCV convention (X right,
    Y down, Z forward).  But the wrist_camera_origin link frame expected by
    CAMERA_POSE_Q in panda_wristcam.py uses (X left, Y up, Z forward).
    The two conventions differ by Rz(pi), so we post-multiply by that
    rotation before converting to URDF rpy.
    """
    _verify_rpy_convention()
    x, y, z = np.asarray(t_X).flatten()
    R_corrected = Rotation.from_matrix(R_X) * Rotation.from_euler("Z", np.pi)
    rx, ry, rz = R_corrected.as_euler("XYZ", degrees=False)
    return (
        '<link name="wrist_camera_origin"/>\n'
        '<joint name="wrist_camera_origin_joint" type="fixed">\n'
        f'  <origin rpy="{rx:.6f} {ry:.6f} {rz:.6f}" xyz="{x:.6f} {y:.6f} {z:.6f}"/>\n'
        '  <parent link="fr3_gripper_base"/>\n'
        '  <child link="wrist_camera_origin"/>\n'
        '</joint>\n'
    )


def _euler_diversity(R_list) -> np.ndarray:
    """Per-axis std-dev (deg) of base-frame Euler angles across recorded samples."""
    if len(R_list) < 2:
        return np.zeros(3)
    eulers = np.stack(
        [Rotation.from_matrix(R).as_euler("XYZ", degrees=True) for R in R_list]
    )
    return eulers.std(axis=0)


def _load_pinocchio_model():
    """Load FR3 URDF and return (model, data, gripper_base_frame_id)."""
    model = pin.buildModelFromUrdf(URDF_PATH)
    data = model.createData()

    # Verify the first 7 pinocchio joints (after the universe joint at index 0)
    # are the arm joints, so q_full[:7] = qpos is a safe assignment.
    expected = [f"fr3_joint{i}" for i in range(1, 8)]
    for i, name in enumerate(expected):
        actual = model.names[i + 1]  # +1 to skip "universe"
        if actual != name:
            raise RuntimeError(
                f"Pinocchio joint layout mismatch: expected {name} at index {i+1}, "
                f"got {actual}. Full joint list: "
                f"{[model.names[j] for j in range(model.njoints)]}"
            )

    if not model.existFrame("fr3_gripper_base"):
        raise RuntimeError("fr3_gripper_base frame not in URDF.")
    gripper_base_fid = model.getFrameId("fr3_gripper_base")
    print(
        f"[Calibration] Loaded pinocchio model from {URDF_PATH}  "
        f"(nq={model.nq}, nv={model.nv})"
    )
    return model, data, gripper_base_fid


def main(cfg: DictConfig):
    """Hand-eye calibration with spacemouse teleoperation via ``FR3Controller``."""
    print("[Calibration] Starting FR3 controller...")
    robot = FR3Controller(cfg)
    robot.start()

    agilex = AgileXPositionController()
    agilex.move(np.asarray(cfg.agilex_qpos))

    print("[Calibration] Initializing spacemouse...")
    sm = SpaceMouseController(freq=100)

    print("[Calibration] Resetting robot to home...")
    robot.reset()

    current_ee_pose = robot.get_ee_pose()
    print(f"[Calibration] Initial EE pose:\n{current_ee_pose}")

    model, data, gripper_base_fid = _load_pinocchio_model()
    q_full = pin.neutral(model)

    R_gripper2base: list = []
    t_gripper2base: list = []
    R_target2cam: list = []
    t_target2cam: list = []
    current_candidate = None  # (base_T_gripper, cam_T_tag) for the latest frame

    with MultiRSDevice(OmegaConf.create({"wrist": cfg.wrist})) as cam_ctx:
        dev = cam_ctx._rget_device("wrist")
        K = dev.K
        D = dev.distortion_coefficients
        print(f"[Calibration] K =\n{K}")
        print(f"[Calibration] D = {D}")
        detector = apriltags.Detector(families="tag36h11")

        try:
            while True:
                frame = cam_ctx.get_frames()["wrist"]
                if frame is not None:
                    color = frame.color
                    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
                    qpos = robot.get_qpos()

                    detections = detector.detect(gray)
                    matching = [d for d in detections if d.tag_id == int(cfg.tag_id)]

                    current_candidate = None
                    if matching and qpos is not None:
                        det = matching[0]
                        for i, corner in enumerate(det.corners):
                            cv2.circle(
                                color,
                                (int(corner[0] + 5), int(corner[1] + 5)),
                                5, (0, 0, 255), -1,
                            )
                            cv2.putText(
                                color, str(i),
                                (int(corner[0]), int(corner[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                            )

                        success, rvec, tvec = cv2.solvePnP(
                            CORNER_Xi_TAG_FRAME,
                            det.corners[[3, 2, 1, 0]],
                            K,
                            D,
                            flags=cv2.SOLVEPNP_IPPE_SQUARE,
                        )

                        if success:
                            cv2.drawFrameAxes(
                                color, K, D, rvec, tvec, SQUARE_HALF_LENGTH
                            )

                            q_full[:7] = np.asarray(qpos)
                            pin.forwardKinematics(model, data, q_full)
                            pin.updateFramePlacements(model, data)
                            base_T_gripper = np.array(
                                data.oMf[gripper_base_fid].homogeneous
                            )

                            cam_T_tag = np.eye(4)
                            cam_T_tag[:3, :3] = Rotation.from_rotvec(
                                rvec.flatten()
                            ).as_matrix()
                            cam_T_tag[:3, 3] = tvec.flatten()
                            current_candidate = (base_T_gripper, cam_T_tag)

                    # overlay
                    status_color = (0, 255, 0) if current_candidate else (0, 0, 255)
                    hud_lines = [
                        f"N={len(R_gripper2base)}  target>={int(cfg.min_samples)}"
                    ]
                    if len(R_gripper2base) >= 2:
                        std = _euler_diversity(R_gripper2base)
                        hud_lines.append(
                            f"eul std(deg) rx={std[0]:.1f} ry={std[1]:.1f} rz={std[2]:.1f}"
                        )
                    hud_lines.append("r=record  u=undo  q=solve+quit")
                    for i, line in enumerate(hud_lines):
                        cv2.putText(
                            color, line, (10, 25 + 25 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2,
                        )

                    cv2.imshow(
                        "Hand-Eye Calibration",
                        cv2.cvtColor(color, cv2.COLOR_RGB2BGR),
                    )
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        break
                    elif k == ord("r"):
                        if current_candidate is None:
                            print("[Calibration] No tag detection; sample skipped.")
                        else:
                            base_T_gripper, cam_T_tag = current_candidate
                            R_gripper2base.append(base_T_gripper[:3, :3].copy())
                            t_gripper2base.append(base_T_gripper[:3, 3].copy())
                            R_target2cam.append(cam_T_tag[:3, :3].copy())
                            t_target2cam.append(cam_T_tag[:3, 3].copy())
                            std = _euler_diversity(R_gripper2base)
                            print(
                                f"[Recorded N={len(R_gripper2base)}]  "
                                f"eul std(deg)  rx={std[0]:.1f}  "
                                f"ry={std[1]:.1f}  rz={std[2]:.1f}"
                            )
                    elif k == ord("u"):
                        if len(R_gripper2base) > 0:
                            R_gripper2base.pop()
                            t_gripper2base.pop()
                            R_target2cam.pop()
                            t_target2cam.pop()
                            print(f"[Undo] N={len(R_gripper2base)}")

                # Apply spacemouse delta to current EE pose. set_ee_pose is
                # non-blocking (sub-ms ZMQ write-through); wait_for_next_tick
                # enforces the control rate.
                delta = sm.homogeneous_delta
                current_ee_pose[:3, 3] += delta[:3, 3]
                current_ee_pose[:3, :3] = delta[:3, :3] @ current_ee_pose[:3, :3]
                robot.set_ee_pose(current_ee_pose)
                robot.wait_for_next_tick()

        except KeyboardInterrupt:
            print("\n[Calibration] Interrupted by user")
        finally:
            print("[Calibration] Cleaning up...")
            sm.close()
            robot.close()
            try:
                agilex.move(np.zeros(6))
            except Exception as e:
                print(f"[Calibration] agilex reset failed: {e}")
            cv2.destroyAllWindows()

            if len(R_gripper2base) < 3:
                print(
                    f"[Calibration] Only {len(R_gripper2base)} sample(s). "
                    f"Need >=3 to solve. Exiting without saving."
                )
                return

            print(
                f"\n[Calibration] Solving AX=XB on {len(R_gripper2base)} samples...\n"
            )
            results = _solve_all_methods(
                R_gripper2base, t_gripper2base, R_target2cam, t_target2cam
            )
            if not results:
                print("[Calibration] All hand-eye methods failed.")
                return

            best_name = min(
                results, key=lambda m: results[m]["stats"]["trans_mean"]
            )
            best = results[best_name]
            R_X = best["R"]
            t_X = np.asarray(best["t"]).flatten()
            print(
                f"\n[Calibration] Best method (lowest trans_mean): {best_name}\n"
                f"[Calibration] gripper_base_T_cam:\n"
                f"R =\n{R_X}\n"
                f"t = {t_X}"
            )

            # Flag worst outlier pairs to help the user prune samples.
            if len(best["pairs"]) > 0:
                top = np.argsort(best["trans_errs"])[-3:][::-1]
                print("\n[Calibration] Top-3 worst pair residuals:")
                for rank, idx in enumerate(top):
                    i, j = best["pairs"][idx]
                    print(
                        f"   #{rank+1}  pair ({i},{j})  "
                        f"trans={best['trans_errs'][idx]*1e3:7.3f} mm  "
                        f"rot={best['rot_errs'][idx]:7.4f}"
                    )

            xml_block = _emit_urdf_block(R_X, t_X)
            print(
                "\n=== Paste into fr3_agilex_gripper_wristcam.urdf "
                "(replace the wrist_camera_origin_joint block) ===\n"
            )
            print(xml_block)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            assets = Path("assets")
            assets.mkdir(parents=True, exist_ok=True)
            npz_path = assets / f"wrist_calibration_data_{timestamp}.npz"
            xml_path = assets / f"wrist_camera_origin_joint_{timestamp}.xml"

            gripper_base_T_cam = np.eye(4)
            gripper_base_T_cam[:3, :3] = R_X
            gripper_base_T_cam[:3, 3] = t_X

            np.savez(
                npz_path,
                gripper_base_T_cam=gripper_base_T_cam,
                R_gripper2base=np.stack(R_gripper2base),
                t_gripper2base=np.stack(t_gripper2base),
                R_target2cam=np.stack(R_target2cam),
                t_target2cam=np.stack(t_target2cam),
                K=K,
                D=D,
                N=len(R_gripper2base),
                method=best_name,
                residual_trans_mean=best["stats"]["trans_mean"],
                residual_trans_median=best["stats"]["trans_median"],
                residual_trans_max=best["stats"]["trans_max"],
                residual_trans_p95=best["stats"]["trans_p95"],
                residual_rot_mean=best["stats"]["rot_mean"],
                residual_rot_median=best["stats"]["rot_median"],
                residual_rot_max=best["stats"]["rot_max"],
                residual_rot_p95=best["stats"]["rot_p95"],
                urdf_path=URDF_PATH,
                tag_size=2 * SQUARE_HALF_LENGTH,
                tag_id=int(cfg.tag_id),
                agilex_qpos=np.asarray(cfg.agilex_qpos),
            )
            xml_path.write_text(xml_block)
            print(f"\n[Calibration] Saved NPZ -> {npz_path}")
            print(f"[Calibration] Saved XML -> {xml_path}")
            print("[Calibration] Done.")


@hydra.main(version_base=None, config_path="config", config_name="hand_eye_calibration")
def _outer_main(cfg: DictConfig):
    main(cfg)


if __name__ == "__main__":
    _outer_main()
