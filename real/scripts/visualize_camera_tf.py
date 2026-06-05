#!/usr/bin/env python3
"""
Generates a ROS2 static_transform_publisher command for camera pose visualization.

Usage:
    # Step 1: Run this script in your conda env to get the command
    python -m real.scripts.visualize_camera_tf --npz_path /path/to/calibration_data.npz
    
    # Step 2: Copy the output command and run it in a ROS2-sourced terminal
"""

import argparse
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description='Generate ROS2 static TF command for camera pose')
    parser.add_argument('--npz_path', type=str, required=True,
                        help='Path to the calibration NPZ file')
    parser.add_argument('--parent_frame', type=str, default='fr3/panda_link0',
                        help='Parent TF frame (robot base frame)')
    parser.add_argument('--child_frame', type=str, default='calibrated_camera',
                        help='Child TF frame name for the camera')
    args = parser.parse_args()
    
    # Load calibration data
    data = np.load(args.npz_path)
    
    print(f"NPZ keys: {list(data.keys())}")
    
    # Try different key names (old format vs new format)
    if 'translations' in data:
        translation = data['translations']
        rotation_matrix = data['rotations']
    elif 'cam_translation_inbase_frame' in data:
        translation = data['cam_translation_inbase_frame']
        rotation_matrix = data['cam_rotation_inbase_frame']
    else:
        raise ValueError(f"Unknown NPZ format. Keys: {list(data.keys())}")
    
    print(f"\nTranslation (xyz): {translation}")
    print(f"Rotation matrix:\n{rotation_matrix}")
    
    # Convert rotation matrix to quaternion
    rot = Rotation.from_matrix(rotation_matrix)
    quat = rot.as_quat()  # Returns [x, y, z, w]
    euler = rot.as_euler('XYZ', degrees=True)
    
    print(f"Euler angles (XYZ, degrees): {euler}")
    print(f"Quaternion (xyzw): {quat}")
    
    # Generate the ROS2 command
    cmd = (
        f"ros2 run tf2_ros static_transform_publisher "
        f"--x {translation[0]:.6f} "
        f"--y {translation[1]:.6f} "
        f"--z {translation[2]:.6f} "
        f"--qx {quat[0]:.6f} "
        f"--qy {quat[1]:.6f} "
        f"--qz {quat[2]:.6f} "
        f"--qw {quat[3]:.6f} "
        f"--frame-id {args.parent_frame} "
        f"--child-frame-id {args.child_frame}"
    )
    
    print("\n" + "="*80)
    print("Run this command in a ROS2-sourced terminal to publish the camera TF:")
    print("="*80)
    print(f"\n{cmd}\n")
    print("="*80)
    print("\nIn RViz2:")
    print("  1. Add a TF display (Add -> By display type -> TF)")
    print(f"  2. Set Fixed Frame to '{args.parent_frame}' (or 'world')")
    print(f"  3. You should see '{args.child_frame}' frame appear")
    print("="*80)


if __name__ == '__main__':
    main()
