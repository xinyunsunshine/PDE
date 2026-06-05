#!/usr/bin/env python3
"""Sample a random subset of episodes from a LeRobot v3.0 dataset.

Usage:
    python real/scripts/sample_lerobot_subset.py \
        /path/to/source_dataset /path/to/output_dataset \
        --num-episodes 10 --seed 42
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description="Sample a subset of episodes from a LeRobot v3.0 dataset")
    parser.add_argument("source", type=Path, help="Path to source LeRobot v3.0 dataset")
    parser.add_argument("output", type=Path, help="Path to output subset dataset")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()

    # Load episode metadata
    episodes_path = source / "meta" / "episodes.jsonl"
    episodes = []
    with open(episodes_path) as f:
        for line in f:
            episodes.append(json.loads(line))

    total = len(episodes)
    n = min(args.num_episodes, total)
    print(f"Source dataset: {total} episodes")
    print(f"Sampling {n} episodes with seed={args.seed}")

    random.seed(args.seed)
    selected = sorted(random.sample(episodes, n), key=lambda e: e["episode_index"])
    selected_indices = {e["episode_index"] for e in selected}
    print(f"Selected episode indices: {sorted(selected_indices)}")

    # Create output directory
    output.mkdir(parents=True, exist_ok=True)

    # --- Parquet data ---
    data_dir = source / "data" / "chunk-000"
    out_data_dir = output / "data" / "chunk-000"
    out_data_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(data_dir.glob("file-*.parquet"))
    all_tables = []
    for pf in parquet_files:
        all_tables.append(pq.read_table(pf))
    full_table = pa.concat_tables(all_tables)

    ep_col = full_table.column("episode_index").to_pylist()
    mask = [idx in selected_indices for idx in ep_col]
    filtered = full_table.filter(mask)

    # Re-index episodes 0..n-1 and recompute frame_index and global index
    old_to_new = {old: new for new, old in enumerate(sorted(selected_indices))}
    old_ep = filtered.column("episode_index").to_pylist()
    old_frame = filtered.column("frame_index").to_pylist()

    new_ep = [old_to_new[e] for e in old_ep]
    # Recompute frame_index per new episode
    new_frame = []
    ep_frame_counter = {}
    for e in new_ep:
        c = ep_frame_counter.get(e, 0)
        new_frame.append(c)
        ep_frame_counter[e] = c + 1
    new_index = list(range(len(new_ep)))

    filtered = filtered.drop("episode_index").append_column("episode_index", pa.array(new_ep, type=pa.int64()))
    filtered = filtered.drop("frame_index").append_column("frame_index", pa.array(new_frame, type=pa.int64()))
    filtered = filtered.drop("index").append_column("index", pa.array(new_index, type=pa.int64()))

    pq.write_table(filtered, out_data_dir / "file-000.parquet")
    print(f"Wrote {len(filtered)} frames to {out_data_dir / 'file-000.parquet'}")

    # --- Videos: symlink entire videos directory ---
    src_videos = source / "videos"
    out_videos = output / "videos"
    if src_videos.exists():
        if out_videos.exists() or out_videos.is_symlink():
            out_videos.unlink() if out_videos.is_symlink() else shutil.rmtree(out_videos)
        os.symlink(str(src_videos), str(out_videos))
        print(f"Symlinked videos: {out_videos} -> {src_videos}")

    # --- Metadata ---
    meta_dir = output / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # episodes.jsonl — re-indexed
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for new_idx, ep in enumerate(selected):
            entry = {
                "episode_index": new_idx,
                "tasks": ep["tasks"],
                "length": ep["length"],
            }
            f.write(json.dumps(entry) + "\n")

    # tasks.jsonl — copy as-is
    shutil.copy2(source / "meta" / "tasks.jsonl", meta_dir / "tasks.jsonl")

    # info.json — update counts
    with open(source / "meta" / "info.json") as f:
        info = json.load(f)
    total_frames = sum(e["length"] for e in selected)
    info["total_episodes"] = n
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{n}"}
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # stats.json — copy from source (approximate, good enough for training)
    stats_src = source / "meta" / "stats.json"
    if stats_src.exists():
        shutil.copy2(stats_src, meta_dir / "stats.json")

    # episodes_stats.jsonl — filter to selected episodes (re-indexed)
    ep_stats_src = source / "meta" / "episodes_stats.jsonl"
    if ep_stats_src.exists():
        all_ep_stats = []
        with open(ep_stats_src) as f:
            for line in f:
                all_ep_stats.append(json.loads(line))
        with open(meta_dir / "episodes_stats.jsonl", "w") as f:
            for old_idx in sorted(selected_indices):
                new_idx = old_to_new[old_idx]
                if old_idx < len(all_ep_stats):
                    stat = all_ep_stats[old_idx]
                    stat["episode_index"] = new_idx
                    f.write(json.dumps(stat) + "\n")

    print(f"\nSubset dataset created at: {output}")
    print(f"  Episodes: {n}")
    print(f"  Frames: {total_frames}")
    print(f"  Original indices: {sorted(selected_indices)}")


if __name__ == "__main__":
    main()
