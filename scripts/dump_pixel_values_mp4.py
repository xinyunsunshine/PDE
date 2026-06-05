"""Dump pixel_values_*.npy files to MP4 videos.

The npy has shape [T, B, C, H, W].

The T dimension may be stride-interleaved: with --stride S, T contains S
interleaved trajectories per b, so the actual number of frames per episode
is T // S.  E.g. with stride=2, even frames (0,2,4,...) are one episode and
odd frames (1,3,5,...) are another.

Use --stride auto to auto-detect the stride by comparing frame similarities.

Channels: C=6 = two camera views stacked (front=0:3, wrist=3:6).
"""

import argparse
import glob
import os

import imageio
import numpy as np


def extract_frame(img: np.ndarray, channels: slice) -> np.ndarray:
    """img: [C, H, W] or [N, C, H, W]. Returns [H, W, 3] uint8."""
    if img.ndim == 4:
        img = img[0]
    img = img[channels]  # [3, H, W]
    img = (img * 0.5 + 0.5).clip(0, 1)
    return (img.transpose(1, 2, 0) * 255).astype("uint8")


def detect_stride(arr: np.ndarray, max_stride: int = 16) -> int:
    """Auto-detect interleave stride by finding the stride that minimizes
    mean frame-to-frame difference for b=0."""
    T = arr.shape[0]
    ref = arr[:, 0].astype(np.float32)
    best_stride, best_diff = 1, float("inf")
    for s in range(1, min(max_stride + 1, T // 2)):
        seq = ref[0::s]
        diffs = [np.abs(seq[t + 1] - seq[t]).mean() for t in range(min(len(seq) - 1, 30))]
        mean_diff = np.mean(diffs)
        if mean_diff < best_diff:
            best_diff = mean_diff
            best_stride = s
    return best_stride


def dump_npy(npy_path: str, fps: int, channels: slice, out_dir: str, stride: int) -> None:
    arr = np.load(npy_path)  # [T, B, C, H, W]
    T, B = arr.shape[:2]
    rank = os.path.splitext(os.path.basename(npy_path))[0].split("_")[-1]
    real_T = T // stride
    print(f"{npy_path}: shape={arr.shape}, stride={stride}, real_T={real_T}")

    for b in range(B):
        for s in range(stride):
            frames = [extract_frame(arr[t, b], channels) for t in range(s, T, stride)]
            out_path = os.path.join(out_dir, f"rank{rank}_b{b:02d}_s{s}.mp4")
            imageio.mimwrite(out_path, frames, fps=fps)
            print(f"  wrote {out_path} ({len(frames)} frames)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "npy_files",
        nargs="*",
        default=None,
        help="Paths to pixel_values_*.npy files (default: auto-detect in cwd)",
    )
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument(
        "--channels",
        choices=["wrist", "front"],
        default="wrist",
        help="wrist=3:6, front=0:3",
    )
    parser.add_argument("--out-dir", default=".", help="Output directory for MP4s")
    parser.add_argument(
        "--stride",
        default="auto",
        help="Deinterleave stride (int or 'auto'). 2 = even/odd frames are separate episodes.",
    )
    args = parser.parse_args()

    ch_map = {"wrist": slice(3, 6), "front": slice(0, 3)}
    channels = ch_map[args.channels]

    npy_files = args.npy_files or sorted(glob.glob("pixel_values_*.npy"))
    if not npy_files:
        print("No pixel_values_*.npy files found.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    for path in npy_files:
        stride = args.stride
        if stride == "auto":
            arr = np.load(path)
            stride = detect_stride(arr)
            print(f"Auto-detected stride={stride} for {path}")
        else:
            stride = int(stride)
        dump_npy(path, fps=args.fps, channels=channels, out_dir=args.out_dir, stride=stride)


if __name__ == "__main__":
    main()
