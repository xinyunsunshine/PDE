#!/usr/bin/env python3
"""Smoke-test for GroupRankingProcessor with a synthetic rollout batch.

Creates a mock all-fail group with color-coded trajectories (each trajectory
has a different background color to give the VLM something to distinguish),
calls the VLM ranking endpoint, and prints the resulting reward assignments.

Usage:
    python scripts/group_ranking/test_group_ranking.py
    python scripts/group_ranking/test_group_ranking.py \\
        --endpoint http://VLM_ENDPOINT_HOST:PORT/v1 \\
        --model Qwen/Qwen3-VL-235B-A22B-Thinking-FP8 \\
        --group-size 4 \\
        --timesteps 20 \\
        --img-size 224 \\
        --task "pick up the ketchup and place it in the basket"
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rlinf.workers.actor.group_ranking_processor import GroupRankingProcessor

# ── Mock helpers ──────────────────────────────────────────────────────────────

_TRAJ_COLORS = [
    (220, 80, 80),  # red-ish   — trajectory 1
    (80, 200, 80),  # green-ish — trajectory 2
    (80, 80, 220),  # blue-ish  — trajectory 3
    (200, 200, 60),  # yellow-ish — trajectory 4
]


def _make_cfg(endpoint: str, model: str, group_size: int) -> OmegaConf:
    return OmegaConf.create(
        {
            "algorithm": {
                "group_size": group_size,
                # group selection — keep default threshold so all-zero groups are picked
                "her_selection": "threshold",
                "her_low_threshold": 0.5,
                "her_high_threshold": 5.0,
                "her_anchor_from_success": True,
                "her_anchor_reward_one": False,
                # VLM backend
                "her_backend": "vllm",
                "her_endpoint": endpoint,
                "her_model": model,
                "her_video_mode": "frames",
                "her_prompt_version": "v7",
                "her_reward_eval_prompt_version": 2,
                "her_prompt_debug_log_n": 0,
                "log_max_groups": 2,
                # ranking-specific
                "ranking_max_frames_per_traj": 8,
            },
            "env": {"train": {"reward_coef": 5.0}},
            "runner": {
                "logger": {
                    "log_path": "/tmp/group_ranking_test",
                    "experiment_name": "test",
                }
            },
        }
    )


def _make_mock_rollout_batch(
    group_size: int,
    T: int,
    H: int,
    W: int,
    task: str,
) -> dict:
    """Synthetic all-fail batch with one group.

    Each trajectory has a distinct background color.  A bright moving patch in
    the top-left corner varies with time to simulate motion.  All rewards are 0
    so the single group is selected by the threshold gate.

    pixel_values shape: [T, B, H, W, 3] uint8 (HWC — passes _to_vlm_frame_uint8
    unchanged).
    """
    B = group_size
    pixel_values = torch.zeros(T, B, H, W, 3, dtype=torch.uint8)

    for b in range(B):
        r, g, bl = _TRAJ_COLORS[b % len(_TRAJ_COLORS)]
        pixel_values[:, b, :, :, 0] = r
        pixel_values[:, b, :, :, 1] = g
        pixel_values[:, b, :, :, 2] = bl
        # Animate a small patch to simulate arm movement
        patch = H // 6
        for t in range(T):
            intensity = int(255 * t / max(T - 1, 1))
            pixel_values[t, b, :patch, :patch, 0] = intensity
            pixel_values[t, b, :patch, :patch, 1] = intensity
            pixel_values[t, b, :patch, :patch, 2] = intensity

    return {
        "rewards": torch.zeros(T, B, 1),
        "forward_inputs": {
            "pixel_values": pixel_values,
            # Dummy token tensors — not used by GroupRankingProcessor but
            # required by _setup_her_batch for tokenizer extraction.
            "input_ids": torch.zeros(T, B, 32, dtype=torch.long),
            "attention_mask": torch.ones(T, B, 32, dtype=torch.long),
        },
        "task_descriptions": [task] * B,
    }


def _load_video_rollout_batch(video_path: str, group_size: int, task: str) -> dict:
    """Extract group_size trajectories from a 2×2 grid video.

    The video is assumed to be a 2×2 mosaic of group_size=4 robot trajectories
    (top-left, top-right, bottom-left, bottom-right). Each quadrant is cropped
    out and stored as a separate trajectory in pixel_values [T, B, H, W, 3].
    """
    if group_size != 4:
        raise ValueError(
            f"_load_video_rollout_batch expects group_size=4 for 2×2 grid, got {group_size}"
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    T = len(frames)
    full_H, full_W = frames[0].shape[:2]
    half_H, half_W = full_H // 2, full_W // 2
    print(
        f"[video] {T} frames, {full_W}×{full_H} grid → "
        f"4 trajs × {half_W}×{half_H} each"
    )

    # Quadrant order: top-left, top-right, bottom-left, bottom-right
    quadrants = [
        (0, 0),           # top-left
        (0, half_W),      # top-right
        (half_H, 0),      # bottom-left
        (half_H, half_W), # bottom-right
    ]

    pixel_values = torch.zeros(T, 4, half_H, half_W, 3, dtype=torch.uint8)
    for b, (r0, c0) in enumerate(quadrants):
        crops = np.stack(
            [f[r0 : r0 + half_H, c0 : c0 + half_W] for f in frames], axis=0
        )  # [T, half_H, half_W, 3]
        pixel_values[:, b] = torch.from_numpy(crops)

    return {
        "rewards": torch.zeros(T, 4, 1),
        "forward_inputs": {
            "pixel_values": pixel_values,
            "input_ids": torch.zeros(T, 4, 32, dtype=torch.long),
            "attention_mask": torch.ones(T, 4, 32, dtype=torch.long),
        },
        "task_descriptions": [task] * 4,
    }


def _save_ranking_videos(
    rollout_batch: dict,
    result: dict,
    save_dir: str,
    max_frames: int = 8,
    fps: int = 4,
) -> None:
    """Save the frames sent to the VLM for each trajectory as H.264 mp4 files.

    Files are named traj_{b}_rank{rank}_reward{reward:.2f}.mp4 so the
    ranking result is visible in the filename.
    """
    from rlinf.workers.actor.her_utils import select_frames_for_vlm

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    pixel_values = rollout_batch["forward_inputs"]["pixel_values"]  # [T, B, H, W, 3]
    T, B = pixel_values.shape[:2]
    vlm_rewards = result.get("vlm_rewards", {})

    sorted_by_reward = sorted(vlm_rewards.items(), key=lambda x: -x[1])
    rank_map = {b: i + 1 for i, (b, _) in enumerate(sorted_by_reward)}

    for b in range(B):
        frames_b = [pixel_values[t, b].numpy() for t in range(T)]
        selected = select_frames_for_vlm(frames_b, max_frames)

        reward = vlm_rewards.get(b, float("nan"))
        rank = rank_map.get(b, "?")
        fname = save_path / f"traj_{b}_rank{rank}_reward{reward:.2f}.mp4"

        frames_rgb = [np.array(f, dtype=np.uint8) for f in selected]
        H_f, W_f = frames_rgb[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(fname), fourcc, fps, (W_f, H_f))
        for frame in frames_rgb:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"  saved: {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default="http://VLM_ENDPOINT_HOST:PORT/v1",
        help="vLLM-compatible VLM endpoint URL",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-235B-A22B-Thinking-FP8",
        help="Model name served at the endpoint",
    )
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument(
        "--timesteps", type=int, default=20, help="Number of frames per trajectory"
    )
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument(
        "--task",
        default="pick up the ketchup and place it in the basket",
        help="Task description shown to the VLM",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Path to an mp4; split into group-size equal segments instead of synthetic frames",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory to save per-trajectory videos sent to the VLM (for review)",
    )
    args = parser.parse_args()

    cfg = _make_cfg(args.endpoint, args.model, args.group_size)
    if args.video:
        rollout_batch = _load_video_rollout_batch(args.video, args.group_size, args.task)
    else:
        rollout_batch = _make_mock_rollout_batch(
            group_size=args.group_size,
            T=args.timesteps,
            H=args.img_size,
            W=args.img_size,
            task=args.task,
        )

    processor = GroupRankingProcessor(
        cfg=cfg,
        rank=0,
        get_model_fn=lambda: None,
        log_warning_fn=lambda msg: print(f"[WARNING] {msg}", flush=True),
    )

    print(
        f"\nRunning GroupRankingProcessor"
        f" | group_size={args.group_size}"
        f" | T={args.timesteps}"
        f" | img={args.img_size}x{args.img_size}"
        f" | task='{args.task}'\n"
    )

    result = processor(rollout_batch, apply_to_batch=True)

    print("\n=== Metrics ===")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")

    print("\n=== VLM rewards (normalized, before reward_coef) ===")
    for b, r in sorted(result["vlm_rewards"].items()):
        print(f"  traj {b}: {r:.3f}")

    print("\n=== Patched rewards in batch (after reward_coef) ===")
    rewards = rollout_batch["rewards"]  # [T, B, 1]
    reward_coef = cfg.env.train.reward_coef
    B_actual = rewards.shape[1]
    for b in range(B_actual):
        total = rewards[:, b, :].sum().item()
        print(f"  traj {b}: {total:.3f}  (reward_coef={reward_coef})")

    if args.save_dir:
        print(f"\n=== Saving trajectory videos to {args.save_dir} ===")
        _save_ranking_videos(rollout_batch, result, args.save_dir)


if __name__ == "__main__":
    main()
