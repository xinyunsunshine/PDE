"""Fix obs-action off-by-one in a collected LeRobot dataset.

Hack for sanity checking original sft data.

Usage:
    python -m real.scripts.fix_obs_action_shift \
        --src-root /path/to/old_dataset \
        --src-repo-id jmarangola/meta-green-slim-up \
        --dst-root /path/to/new_dataset \
        --dst-repo-id jmarangola/meta-green-slim-up-fixed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm


def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) float [0, 1] tensor to (H, W, C) uint8 numpy."""
    return (t.permute(1, 2, 0) * 255).byte().numpy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src-root", type=Path, required=True)
    p.add_argument("--src-repo-id", type=str, required=True)
    p.add_argument("--dst-root", type=Path, required=True)
    p.add_argument("--dst-repo-id", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading source dataset: {args.src_repo_id} from {args.src_root}")
    src = LeRobotDataset(repo_id=args.src_repo_id, root=args.src_root)
    hf = src.hf_dataset

    num_episodes = src.meta.total_episodes
    fps = src.meta.fps
    features = src.meta.features
    robot_type = src.meta.robot_type
    video_keys = src.meta.video_keys
    use_videos = len(video_keys) > 0

    print(
        f"Source: {num_episodes} episodes, {src.meta.total_frames} frames, "
        f"{fps} fps, videos={use_videos}"
    )

    print(f"Creating destination dataset: {args.dst_repo_id} at {args.dst_root}")
    dst = LeRobotDataset.create(
        repo_id=args.dst_repo_id,
        fps=fps,
        features=features,
        root=args.dst_root,
        robot_type=robot_type,
        use_videos=use_videos,
        image_writer_threads=10,
    )

    auto_keys = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    non_video_feature_keys = [
        k for k, v in features.items()
        if v["dtype"] not in ("video", "image") and k not in auto_keys
    ]

    total_src_frames = 0
    total_dst_frames = 0

    for ep_idx in tqdm(range(num_episodes), desc="Episodes"):
        ep_mask = np.array(hf["episode_index"]) == ep_idx
        ep_indices = np.where(ep_mask)[0]
        n = len(ep_indices)

        if n < 2:
            print(f"  Episode {ep_idx}: only {n} frame(s), skipping.")
            continue

        total_src_frames += n

        task = src[int(ep_indices[0])]["task"]

        actions = []
        for idx in ep_indices:
            row = hf[int(idx)]
            actions.append(np.asarray(row["action"], dtype=np.float32))

        for i in range(n - 1):
            src_idx = int(ep_indices[i])
            item = src[src_idx]

            frame: dict = {"task": task}

            for key in non_video_feature_keys:
                if key == "action":
                    frame[key] = actions[i + 1]
                else:
                    val = item[key]
                    if isinstance(val, torch.Tensor):
                        val = val.numpy()
                    frame[key] = val

            for vk in video_keys:
                frame[vk] = _tensor_to_uint8(item[vk])

            dst.add_frame(frame)

        dst.save_episode()
        ep_dst_frames = n - 1
        total_dst_frames += ep_dst_frames

    dst.finalize()

    print(
        f"\nDone. {num_episodes} episodes processed.\n"
        f"  Source frames: {total_src_frames}\n"
        f"  Output frames: {total_dst_frames}\n"
        f"  Dropped:       {total_src_frames - total_dst_frames} "
        f"(1 per episode)\n"
        f"  Output:        {args.dst_root}"
    )


if __name__ == "__main__":
    main()
