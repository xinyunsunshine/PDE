"""Delete all episodes for a given task from a LeRobot v2 dataset.

Only safe to use when the episodes to delete are the LAST episodes in the
dataset (highest indices). If they are interleaved with other tasks, the
parquet/video files for later episodes would need renaming and reindexing,
which this script does not handle.

Usage:
    python -m real.scripts.delete_task_episodes \
        --dataset /home/user/lrds/pde_real_sft \
        --task "open the top drawer and put the bowl inside"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w") as f:
        for item in items:
            f.write(json.dumps(item, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without doing it")
    args = parser.parse_args()

    ds = args.dataset.expanduser().resolve()
    meta = ds / "meta"
    task_to_delete = args.task

    # --- Load metadata ---
    episodes = load_jsonl(meta / "episodes.jsonl")
    episodes_stats = load_jsonl(meta / "episodes_stats.jsonl")
    tasks = load_jsonl(meta / "tasks.jsonl")
    with (meta / "info.json").open() as f:
        info = json.load(f)

    # --- Find episodes to delete ---
    delete_indices = {
        ep["episode_index"]
        for ep in episodes
        if task_to_delete in ep.get("tasks", [])
    }
    if not delete_indices:
        print(f"No episodes found for task: '{task_to_delete}'")
        sys.exit(1)

    keep_episodes = [ep for ep in episodes if ep["episode_index"] not in delete_indices]
    max_kept = max((ep["episode_index"] for ep in keep_episodes), default=-1)
    if min(delete_indices) <= max_kept:
        print(
            "ERROR: Episodes to delete are not all at the tail of the dataset.\n"
            f"  Deleting indices: {sorted(delete_indices)}\n"
            f"  Max kept index:   {max_kept}\n"
            "This script only supports deleting trailing episodes. Aborting."
        )
        sys.exit(1)

    # --- Summarize ---
    deleted_frames = sum(ep["length"] for ep in episodes if ep["episode_index"] in delete_indices)
    print(f"Task:              '{task_to_delete}'")
    print(f"Episodes to delete: {sorted(delete_indices)}")
    print(f"Frames to delete:   {deleted_frames}")

    # Collect files to delete
    chunks_size = info.get("chunks_size", 1000)
    video_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
    files_to_delete: list[Path] = []
    for idx in sorted(delete_indices):
        chunk = idx // chunks_size
        # Parquet
        pq = ds / f"data/chunk-{chunk:03d}/episode_{idx:06d}.parquet"
        if pq.exists():
            files_to_delete.append(pq)
        # Videos
        for vk in video_keys:
            vf = ds / f"videos/chunk-{chunk:03d}/{vk}/episode_{idx:06d}.mp4"
            if vf.exists():
                files_to_delete.append(vf)

    print(f"\nFiles to delete ({len(files_to_delete)}):")
    for f in files_to_delete:
        print(f"  {f}")

    if args.dry_run:
        print("\n[dry-run] No changes made.")
        return

    # --- Confirm ---
    answer = input("\nProceed? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        return

    # --- Delete files ---
    for f in files_to_delete:
        f.unlink()
        print(f"  deleted {f.name}")

    # --- Rewrite episodes.jsonl ---
    write_jsonl(meta / "episodes.jsonl", keep_episodes)

    # --- Rewrite episodes_stats.jsonl ---
    keep_stats = [s for s in episodes_stats if s["episode_index"] not in delete_indices]
    write_jsonl(meta / "episodes_stats.jsonl", keep_stats)

    # --- Rewrite tasks.jsonl (remove task if no episodes remain) ---
    remaining_tasks = set()
    for ep in keep_episodes:
        remaining_tasks.update(ep.get("tasks", []))
    keep_tasks = [t for t in tasks if t["task"] in remaining_tasks]
    # Re-index task indices to be contiguous
    for new_idx, t in enumerate(keep_tasks):
        t["task_index"] = new_idx
    write_jsonl(meta / "tasks.jsonl", keep_tasks)

    # --- Update info.json ---
    info["total_episodes"] = len(keep_episodes)
    info["total_frames"] -= deleted_frames
    info["total_tasks"] = len(keep_tasks)
    info["total_videos"] -= len(delete_indices) * len(video_keys)
    info["splits"] = {"train": f"0:{len(keep_episodes)}"}
    with (meta / "info.json").open("w") as f:
        json.dump(info, f, indent=4)
        f.write("\n")

    print(f"\nDone. Deleted {len(delete_indices)} episodes ({deleted_frames} frames).")
    print(f"Dataset now has {info['total_episodes']} episodes, {info['total_frames']} frames, {info['total_tasks']} tasks.")


if __name__ == "__main__":
    main()
