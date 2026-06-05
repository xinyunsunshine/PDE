"""Rotation and 10-dim pose helpers for the FR3 VLA pipeline.

The 10-dim action/state layout used end-to-end in teleop, SFT, and deploy is:
    [x, y, z,  r6d_0, r6d_1, r6d_2, r6d_3, r6d_4, r6d_5,  gripper_width]
"""

from __future__ import annotations

import numpy as np


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """Flatten the first two columns of a 3x3 rotation matrix into a (6,) vec.

    Args:
        R: (3, 3) rotation matrix.

    Returns:
        (6,) float array = ``[R[:, 0], R[:, 1]]`` concatenated.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"Expected (3, 3) rotation matrix, got {R.shape}")
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float64)


def rot6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
    """Reconstruct a 3x3 rotation matrix from a 6-vector via Gram-Schmidt.

    Args:
        r6d: (6,) array holding two stacked column vectors of length 3.

    Returns:
        (3, 3) proper rotation matrix (``det == +1``).
    """
    r6d = np.asarray(r6d, dtype=np.float64).reshape(-1)
    if r6d.shape != (6,):
        raise ValueError(f"Expected (6,) rot6d vector, got {r6d.shape}")

    a1 = r6d[:3]
    a2 = r6d[3:]

    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    b2_unnorm = a2 - np.dot(b1, a2) * b1
    b2 = b2_unnorm / (np.linalg.norm(b2_unnorm) + 1e-12)
    b3 = np.cross(b1, b2)

    return np.stack([b1, b2, b3], axis=1)


def homogeneous_to_pose_10d(T: np.ndarray, gripper_width: float) -> np.ndarray:
    """Pack a 4x4 homogeneous transform + gripper width into the 10-dim layout.

    Args:
        T: (4, 4) homogeneous transform in the robot base frame.
        gripper_width: scalar width in meters (AgileX gripper range [0, 0.069]).

    Returns:
        (10,) float32 array ``[xyz(3), rot6d(6), gripper(1)]``.
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Expected (4, 4) homogeneous matrix, got {T.shape}")

    xyz = T[:3, 3]
    r6d = matrix_to_rot6d(T[:3, :3])
    return np.concatenate([xyz, r6d, [float(gripper_width)]]).astype(np.float32)


def pose_10d_to_homogeneous(action: np.ndarray) -> tuple[np.ndarray, float]:
    """Unpack the 10-dim layout into a homogeneous transform + gripper width.

    Args:
        action: (10,) array ``[xyz(3), rot6d(6), gripper(1)]``.

    Returns:
        Tuple ``(T, gripper_width)`` where ``T`` is a (4, 4) homogeneous
        transform and ``gripper_width`` is a scalar float.
    """
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape != (10,):
        raise ValueError(f"Expected (10,) action, got {action.shape}")

    T = np.eye(4)
    T[:3, :3] = rot6d_to_matrix(action[3:9])
    T[:3, 3] = action[:3]
    return T, float(action[9])
