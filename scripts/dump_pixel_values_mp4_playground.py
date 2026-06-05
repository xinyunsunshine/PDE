"""Dump pixel_values_*.npy files to MP4 videos.

The npy is saved POST-reshape (after process_nested_dict_for_adv), so its shape is:
  [n_chunk_steps, rollout_epoch * bsz, C, H, W]

The B dimension layout (C-order flatten of [rollout_epoch, bsz]):
  b = epoch * bsz + env_idx
  b=0..bsz-1          → epoch 0, envs 0..bsz-1
  b=bsz..2*bsz-1      → epoch 1, envs 0..bsz-1
  ...

Each b is one continuous episode (all T frames). Different epochs are separate
episodes with different init states.

Pass --rollout-epoch so output filenames show epoch/env correctly (default=1).

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


def dump_npy(npy_path: str, fps: int, channels: slice, out_dir: str, rollout_epoch: int) -> None:
    arr = np.load(npy_path)  # [n_chunk_steps, rollout_epoch * bsz, C, H, W]
    T, B = arr.shape[:2]
    rank = os.path.splitext(os.path.basename(npy_path))[0].split("_")[-1]
    bsz = B // rollout_epoch
    print(f"{npy_path}: shape={arr.shape}, rollout_epoch={rollout_epoch}, bsz={bsz}")

    arr = arr.reshape(T, rollout_epoch, -1, *arr.shape[2:])
    arr = arr.transpose(0, 2, 1, *range(3, arr.ndim))
    for epoch in range(arr.shape[1]):
        for b in range(arr.shape[2]):
            env = b
            epoch = epoch
            frames = [extract_frame(arr[t, epoch, b], channels) for t in range(T)]
            out_path = os.path.join(out_dir, f"rank{rank}_ep{epoch}_env{env:02d}.mp4")
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
    parser.add_argument("--out-dir", default="./tmp/manual/mp4", help="Output directory for MP4s")
    parser.add_argument(
        "--rollout-epoch",
        type=int,
        default=2,
        help="rollout_epoch from config — used for epoch/env labeling in filenames.",
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
        dump_npy(path, fps=args.fps, channels=channels, out_dir=args.out_dir, rollout_epoch=args.rollout_epoch)


if __name__ == "__main__":
    main()
