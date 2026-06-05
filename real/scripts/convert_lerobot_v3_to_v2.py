#!/usr/bin/env python3
"""Convert a LeRobot v3.0 dataset to v2.1 format for use with LeRobot 0.1.0.

v3.0 stores all episodes in a single parquet file and multi-episode video files.
v2.1 stores one parquet and one mp4 per episode.

Usage:
    python real/scripts/convert_lerobot_v3_to_v2.py \
        /path/to/v3_dataset /path/to/v2_output
"""

import argparse
import json
from pathlib import Path

from fractions import Fraction

import av
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot v3.0 dataset to v2.1 format")
    parser.add_argument("source", type=Path, help="Path to v3.0 dataset root")
    parser.add_argument("output", type=Path, help="Path to write v2.1 dataset")
    parser.add_argument("--video-keys", nargs="+",
                        default=["observation.images.global", "observation.images.wrist"],
                        help="Video keys to extract")
    args = parser.parse_args()

    src = args.source.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Source (v3.0): {src}")
    print(f"Output (v2.1): {out}")

    # --- Load source info ---
    with open(src / "meta" / "info.json") as f:
        info = json.load(f)
    if info["codebase_version"] not in ("v3.0", "v2.1"):
        raise ValueError(f"Expected v3.0 or v2.1, got {info['codebase_version']}")

    # --- Load episodes metadata from parquet ---
    ep_files = sorted((src / "meta" / "episodes").glob("chunk-*"))
    ep_tables = [pq.read_table(str(f)) for f in ep_files]
    episodes_meta = pa.concat_tables(ep_tables).to_pydict()
    num_episodes = len(episodes_meta["episode_index"])
    print(f"Total episodes: {num_episodes}")

    # --- Load full data parquet (v3.0 uses file-NNN.parquet, skip any episode_*.parquet from prior runs) ---
    data_files = sorted(f for f in (src / "data").glob("chunk-*/file-*.parquet"))
    data_table = pa.concat_tables([pq.read_table(str(f)) for f in data_files])
    ep_index_col = data_table.column("episode_index").to_pylist()
    print(f"Total frames: {len(data_table)}")

    # --- Create output directory structure ---
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)

    # --- Convert schema: fixed_size_list -> list, strip huggingface metadata ---
    new_fields = []
    for field in data_table.schema:
        if pa.types.is_fixed_size_list(field.type):
            new_fields.append(pa.field(field.name, pa.list_(field.type.value_type)))
        else:
            new_fields.append(field)
    new_schema = pa.schema(new_fields)
    # Rebuild table with compatible schema (drops huggingface metadata)
    data_table = pa.table(
        {f.name: data_table.column(f.name).combine_chunks() for f in new_fields},
        schema=new_schema,
    )

    # --- Split parquet into per-episode files ---
    print("Splitting parquet into per-episode files...")
    # Build index: episode -> row indices
    ep_to_rows: dict[int, list[int]] = {}
    for i, ep in enumerate(ep_index_col):
        ep_to_rows.setdefault(ep, []).append(i)

    for ep_idx in sorted(ep_to_rows.keys()):
        rows = ep_to_rows[ep_idx]
        ep_table = data_table.take(rows)
        out_parquet = out / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        pq.write_table(ep_table, str(out_parquet))

    print(f"Wrote {len(ep_to_rows)} per-episode parquets.")

    # --- Extract per-episode videos using PyAV ---
    for video_key in args.video_keys:
        (out / "videos" / video_key).mkdir(parents=True, exist_ok=True)
        print(f"Extracting {video_key} videos...")

        # Group episodes by source video file to avoid re-opening
        file_to_eps: dict[str, list[tuple[int, float, float]]] = {}
        for i in range(num_episodes):
            ep_idx = episodes_meta["episode_index"][i]
            chunk_idx = episodes_meta[f"videos/{video_key}/chunk_index"][i]
            file_idx = episodes_meta[f"videos/{video_key}/file_index"][i]
            from_ts = episodes_meta[f"videos/{video_key}/from_timestamp"][i]
            to_ts = episodes_meta[f"videos/{video_key}/to_timestamp"][i]
            src_video = str(src / "videos" / video_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4")
            file_to_eps.setdefault(src_video, []).append((ep_idx, from_ts, to_ts))

        done = 0
        for src_video_path, ep_list in file_to_eps.items():
            for ep_idx, from_ts, to_ts in ep_list:
                out_video = out / "videos" / video_key / f"episode_{ep_idx:06d}.mp4"
                if out_video.exists():
                    done += 1
                    continue

                container_in = av.open(src_video_path)
                stream_in = container_in.streams.video[0]
                time_base = float(stream_in.time_base)
                fps = stream_in.average_rate or Fraction(50)

                out_tb = Fraction(1, int(fps))

                container_out = av.open(str(out_video), mode="w")
                stream_out = container_out.add_stream("libx264", rate=fps)
                stream_out.width = stream_in.codec_context.width
                stream_out.height = stream_in.codec_context.height
                stream_out.pix_fmt = "yuv420p"
                stream_out.options = {"crf": "23", "preset": "ultrafast"}
                stream_out.time_base = out_tb

                start_pts = int(from_ts / time_base)
                end_pts = int(to_ts / time_base)
                container_in.seek(start_pts, stream=stream_in)

                frame_idx = 0
                for frame in container_in.decode(stream_in):
                    if frame.pts is not None and frame.pts >= end_pts:
                        break
                    if frame.pts is not None and frame.pts < start_pts:
                        continue
                    frame.pts = frame_idx
                    frame.dts = frame_idx
                    frame.time_base = out_tb
                    for pkt in stream_out.encode(frame):
                        container_out.mux(pkt)
                    frame_idx += 1

                for pkt in stream_out.encode():
                    container_out.mux(pkt)

                container_out.close()
                container_in.close()

                done += 1
                if done % 20 == 0:
                    print(f"  Extracted {done}/{num_episodes} episodes")

        print(f"  Done: {video_key} ({done} episodes)")

    # --- Write meta/tasks.jsonl ---
    tasks_file = src / "meta" / "tasks.parquet"
    tasks_table = pq.read_table(str(tasks_file)).to_pydict()
    with open(out / "meta" / "tasks.jsonl", "w") as f:
        for idx, task in zip(tasks_table["task_index"], tasks_table["__index_level_0__"]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    # --- Write meta/episodes.jsonl ---
    with open(out / "meta" / "episodes.jsonl", "w") as f:
        for i in range(num_episodes):
            ep_idx = episodes_meta["episode_index"][i]
            tasks = episodes_meta["tasks"][i]
            length = episodes_meta["length"][i]
            # tasks in v2.1 are stored as string task indices ("0", "1", ...)
            task_strs = [str(t) for t in tasks] if isinstance(tasks[0], int) else tasks
            f.write(json.dumps({"episode_index": ep_idx, "tasks": task_strs, "length": length}) + "\n")

    # --- Write meta/info.json ---
    total_frames = sum(episodes_meta["length"])
    out_info = {
        "codebase_version": "v2.1",
        "robot_type": info.get("robot_type", ""),
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "total_tasks": info.get("total_tasks", 1),
        "chunks_size": info.get("chunks_size", 1000),
        "data_files_size_in_mb": info.get("data_files_size_in_mb", 100),
        "video_files_size_in_mb": info.get("video_files_size_in_mb", 200),
        "fps": info.get("fps", 50),
        "splits": {"train": f"0:{num_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4",
        "features": info.get("features", {}),
    }
    with open(out / "meta" / "info.json", "w") as f:
        json.dump(out_info, f, indent=4)

    # --- Copy stats.json if present and different path ---
    src_stats = src / "meta" / "stats.json"
    out_stats = out / "meta" / "stats.json"
    if src_stats.exists() and src_stats.resolve() != out_stats.resolve():
        import shutil
        shutil.copy2(src_stats, out_stats)

    print(f"\nConversion complete!")
    print(f"  Episodes: {num_episodes}")
    print(f"  Frames:   {total_frames}")
    print(f"  Output:   {out}")


if __name__ == "__main__":
    main()
