#!/usr/bin/env python3
"""Convert a LeRobot v3.0 dataset to v2.1 format.

v3.0 stores multiple episodes per parquet file (data/chunk-NNN/file-NNN.parquet)
and per video file (videos/{key}/chunk-NNN/file-NNN.mp4), with a rich episode
metadata directory (meta/episodes/chunk-NNN/file-NNN.parquet).

v2.1 expects one parquet per episode (data/chunk-NNN/episode_NNNNNN.parquet)
and one video per episode (videos/{key}/episode_NNNNNN.mp4).

This script:
  1. Reads the v3 episode metadata to learn the episode→file mapping.
  2. Splits multi-episode parquets into per-episode parquets.
  3. Symlinks per-episode video files to the v3 originals (LeRobot uses
     frame_index to seek, so shared video files work fine).
  4. Rewrites info.json with v2.1 paths and version.
  5. Rewrites episodes.jsonl with task indices (v2.1) instead of task text (v3).
  6. Copies episodes_stats.jsonl, tasks.jsonl, stats.json as-is.

Usage:
    python convert_lerobot_v3_to_v21.py /path/to/v3_dataset /path/to/v21_output

The original v3 dataset is not modified. Video files are symlinked (not copied)
to save disk space.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq


def load_v3_episode_metadata(v3_root: Path) -> list[dict]:
    """Read all episode rows from meta/episodes/chunk-NNN/file-NNN.parquet."""
    episodes_dir = v3_root / "meta" / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(
            f"v3 episode metadata not found at {episodes_dir}. "
            "Is this a v3.0 dataset?"
        )

    all_episodes = []
    for chunk_dir in sorted(episodes_dir.iterdir()):
        if not chunk_dir.is_dir():
            continue
        for pf in sorted(chunk_dir.glob("file-*.parquet")):
            table = pq.read_table(pf)
            for i in range(len(table)):
                row = {col: table[col][i].as_py() for col in table.column_names}
                all_episodes.append(row)

    all_episodes.sort(key=lambda r: r["episode_index"])
    return all_episodes


def build_task_map(v3_root: Path) -> dict[str, int]:
    """Build task text -> task_index map from meta/tasks.jsonl."""
    task_map = {}
    tasks_path = v3_root / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        with open(tasks_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                task_map[entry["task"]] = entry["task_index"]
    return task_map


def split_parquets(
    v3_root: Path,
    v21_root: Path,
    episodes: list[dict],
    chunks_size: int,
) -> None:
    """Split multi-episode v3 parquets into per-episode v2.1 parquets.

    An episode can span multiple v3 file-NNN.parquet files (e.g. episode 1
    may have rows at the end of file-000 and the start of file-001).  We
    load ALL parquet files, concatenate, then filter per episode.
    """
    import pyarrow as pa

    data_dir = v3_root / "data"
    all_tables = []
    for chunk_dir in sorted(data_dir.iterdir()):
        if not chunk_dir.is_dir():
            continue
        for pf in sorted(chunk_dir.glob("file-*.parquet")):
            all_tables.append(pq.read_table(pf))

    full_table = pa.concat_tables(all_tables)
    ep_col = full_table["episode_index"].to_pylist()

    for ep in episodes:
        ep_idx = ep["episode_index"]
        mask = [e == ep_idx for e in ep_col]
        ep_table = full_table.filter(mask)

        ep_chunk = ep_idx // chunks_size
        dst_dir = v21_root / "data" / f"chunk-{ep_chunk:03d}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"episode_{ep_idx:06d}.parquet"
        pq.write_table(ep_table, dst)

    print(f"  Split {len(episodes)} episodes into per-episode parquets.")


def symlink_videos(
    v3_root: Path,
    v21_root: Path,
    episodes: list[dict],
    video_keys: list[str],
) -> None:
    """Create per-episode video symlinks pointing to v3 originals."""
    for vkey in video_keys:
        chunk_col = f"videos/{vkey}/chunk_index"
        file_col = f"videos/{vkey}/file_index"

        dst_base = v21_root / "videos" / vkey
        dst_base.mkdir(parents=True, exist_ok=True)

        for ep in episodes:
            if chunk_col not in ep or file_col not in ep:
                print(f"  Warning: episode {ep['episode_index']} missing video metadata for {vkey}, skipping.")
                continue

            v_chunk = ep[chunk_col]
            v_file = ep[file_col]
            src = (v3_root / "videos" / vkey / f"chunk-{v_chunk:03d}" / f"file-{v_file:03d}.mp4").resolve()

            if not src.exists():
                print(f"  Warning: video {src} not found, skipping.")
                continue

            dst = dst_base / f"episode_{ep['episode_index']:06d}.mp4"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)

    print(f"  Created video symlinks for {len(video_keys)} camera(s).")


def write_info_json(v3_root: Path, v21_root: Path) -> dict:
    """Rewrite info.json with v2.1 version and path templates."""
    with open(v3_root / "meta" / "info.json") as f:
        info = json.load(f)

    info["codebase_version"] = "v2.1"
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info["video_path"] = "videos/{video_key}/episode_{episode_index:06d}.mp4"

    dst = v21_root / "meta" / "info.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(info, f, indent=4)
        f.write("\n")

    print(f"  Wrote info.json (v2.1).")
    return info


def write_episodes_jsonl(
    v21_root: Path,
    episodes: list[dict],
    task_map: dict[str, int],
) -> None:
    """Write episodes.jsonl with task indices instead of task text."""
    dst = v21_root / "meta" / "episodes.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)

    with open(dst, "w") as f:
        for ep in episodes:
            # v3 stores task text in "tasks" list; v2.1 stores task index as string
            tasks_v3 = ep.get("tasks", [])
            tasks_v21 = []
            for t in tasks_v3:
                if t in task_map:
                    tasks_v21.append(str(task_map[t]))
                else:
                    tasks_v21.append(t)

            entry = {
                "episode_index": ep["episode_index"],
                "tasks": tasks_v21,
                "length": ep["length"],
            }
            f.write(json.dumps(entry) + "\n")

    print(f"  Wrote episodes.jsonl ({len(episodes)} episodes).")


def write_episodes_stats(
    v3_root: Path,
    v21_root: Path,
    episodes: list[dict],
) -> None:
    """Build episodes_stats.jsonl from v3 episode metadata.

    v3 stores per-episode stats as columns in meta/episodes/chunk-NNN/file-NNN.parquet
    (e.g. stats/observation.state/mean, stats/action/min, etc.).
    v2.1 expects a flat episodes_stats.jsonl with nested dicts.
    """
    dst = v21_root / "meta" / "episodes_stats.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Collect all stats/ column names
    stats_cols = [c for c in episodes[0].keys() if c.startswith("stats/")]

    # Group by feature: stats/observation.state/mean -> observation.state -> mean
    from collections import defaultdict

    with open(dst, "w") as f:
        for ep in episodes:
            feature_stats: dict = defaultdict(dict)
            for col in stats_cols:
                parts = col.split("/", 2)  # stats / feature.name / stat_name
                if len(parts) == 3:
                    _, feature, stat = parts
                    val = ep[col]
                    # Ensure values are lists (v2.1 expects arrays, not scalars)
                    if not isinstance(val, list):
                        val = [val]
                    feature_stats[feature][stat] = val

            entry = {
                "episode_index": ep["episode_index"],
                "stats": dict(feature_stats),
            }
            f.write(json.dumps(entry) + "\n")

    print(f"  Wrote episodes_stats.jsonl ({len(episodes)} episodes).")


def copy_meta_files(v3_root: Path, v21_root: Path) -> None:
    """Copy tasks.jsonl and stats.json verbatim."""
    for name in ("tasks.jsonl", "stats.json"):
        src = v3_root / "meta" / name
        if src.exists():
            dst = v21_root / "meta" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  Copied {name}.")


def convert(v3_root: Path, v21_root: Path) -> None:
    print(f"Converting LeRobot v3.0 -> v2.1")
    print(f"  Source: {v3_root}")
    print(f"  Output: {v21_root}")
    print()

    # Load v3 metadata
    episodes = load_v3_episode_metadata(v3_root)
    task_map = build_task_map(v3_root)
    print(f"Found {len(episodes)} episodes, {len(task_map)} tasks.")

    with open(v3_root / "meta" / "info.json") as f:
        v3_info = json.load(f)
    chunks_size = v3_info.get("chunks_size", 1000)

    # Detect video keys from features
    video_keys = [
        k for k, v in v3_info.get("features", {}).items()
        if v.get("dtype") == "video"
    ]
    print(f"Video keys: {video_keys}")
    print()

    # Create output directory
    v21_root.mkdir(parents=True, exist_ok=True)

    # 1. Write info.json
    write_info_json(v3_root, v21_root)

    # 2. Split parquets
    split_parquets(v3_root, v21_root, episodes, chunks_size)

    # 3. Symlink videos
    symlink_videos(v3_root, v21_root, episodes, video_keys)

    # 4. Write episodes.jsonl
    write_episodes_jsonl(v21_root, episodes, task_map)

    # 5. Write episodes_stats.jsonl
    write_episodes_stats(v3_root, v21_root, episodes)

    # 6. Copy tasks.jsonl, stats.json
    copy_meta_files(v3_root, v21_root)

    print()
    print(f"Done. v2.1 dataset at: {v21_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot v3.0 dataset to v2.1 format"
    )
    parser.add_argument("v3_root", type=Path, help="Path to v3.0 dataset")
    parser.add_argument("v21_output", type=Path, help="Path for v2.1 output")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Remove existing output directory before converting"
    )
    args = parser.parse_args()

    if args.v21_output.exists():
        if args.overwrite:
            shutil.rmtree(args.v21_output)
        else:
            raise FileExistsError(
                f"Output directory {args.v21_output} already exists. "
                "Use --overwrite to replace it."
            )

    convert(args.v3_root, args.v21_output)


if __name__ == "__main__":
    main()
