"""Frame extraction + manifest computation for the MCP server.

Layout: each prompt directory under `task<X>/round<Y>/prompt<Z>/` holds one or
more `*_uncut.{mp4,json}` pairs (one per deploy.py invocation). Frame extraction
writes per-episode subdirs into `prompt<Z>/processed/`. The aggregate manifest
in `processed/manifest.json` only carries success_rate when the strict
completion criterion (successes + failures >= expected_trials, resets ignored)
is met across all uncut.json files in the prompt dir.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from real.mcp_server.config import FRAME_SIZE, NUM_FRAMES
from real.round_task import find_uncut_jsons, processed_dir, prompt_dir


def load_metadata(video_path: Path) -> dict | None:
    json_path = video_path.with_suffix(".json")
    if not json_path.exists():
        json_path = video_path.parent / f"{video_path.stem}.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        return json.load(f)


def find_uncut_videos(prompt_path: Path) -> list[Path]:
    if not prompt_path.exists():
        return []
    return sorted(p for p in prompt_path.iterdir() if p.suffix == ".mp4" and "_uncut" in p.stem)


def extract_episode_frames(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    num_frames: int = NUM_FRAMES,
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    total = end_frame - start_frame
    if total <= 0:
        cap.release()
        return []

    indices = [start_frame + int(i * total / num_frames) for i in range(num_frames)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            if FRAME_SIZE is not None:
                frame = cv2.resize(frame, FRAME_SIZE)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def save_frames(frames: list[np.ndarray], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        path = output_dir / f"frame_{i:02d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        paths.append(path)
    return paths


def process_single_video(
    video_path: Path,
    output_base: Path,
    prompt_text: str,
    episode_offset: int = 0,
) -> list[dict]:
    """Extract frames for each non-reset episode in `video_path`.

    Reset-labeled episodes are recorded in the source uncut.json but get no
    frames extracted and no per-episode metadata written.
    """
    metadata = load_metadata(video_path)
    if metadata is None:
        return []

    episodes = metadata.get("episodes", [])
    if not episodes:
        return []

    timestamp = video_path.stem.replace("_uncut", "")
    results = []

    for ep in episodes:
        ep_id = ep["episode_id"]
        start_frame = ep["start_frame"]
        end_frame = ep["end_frame"]
        label = ep.get("label", "unknown")

        if label == "reset":
            results.append({
                "episode_id": episode_offset + ep_id,
                "label": "reset",
                "skipped": True,
                "reason": "reset",
            })
            continue

        ep_dir_name = f"ep{episode_offset + ep_id:03d}_{timestamp}"
        ep_dir = output_base / ep_dir_name

        if ep_dir.exists() and (ep_dir / "prompt.txt").exists():
            existing_frames = list(ep_dir.glob("frame_*.png"))
            if len(existing_frames) == NUM_FRAMES:
                results.append({
                    "episode_id": episode_offset + ep_id,
                    "dir": str(ep_dir),
                    "frames": NUM_FRAMES,
                    "label": label,
                    "skipped": True,
                    "reason": "already_processed",
                })
                continue

        frames = extract_episode_frames(video_path, start_frame, end_frame)
        if not frames:
            continue

        save_frames(frames, ep_dir)

        prompt_file = ep_dir / "prompt.txt"
        prompt_file.write_text(prompt_text)

        ep_meta = {
            "episode_id": episode_offset + ep_id,
            "label": label,
            "steps": ep.get("steps", 0),
            "prompt": prompt_text,
            "source_video": video_path.name,
            "start_frame": start_frame,
            "end_frame": end_frame,
        }
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump(ep_meta, f, indent=2)

        results.append({
            "episode_id": episode_offset + ep_id,
            "dir": str(ep_dir),
            "frames": len(frames),
            "label": label,
            "skipped": False,
        })

    return results


def _aggregate_uncut_metadata(prompt_path: Path) -> dict[str, Any]:
    """Pull self-describing fields from any uncut.json in the dir.

    Returns the union of {task_id, task_name, round_num, prompt_idx, prompt_text,
    expected_trials}. Newer files override older ones if values disagree.
    """
    out: dict[str, Any] = {
        "task_id": None,
        "task_name": None,
        "round_num": None,
        "prompt_idx": None,
        "prompt_text": None,
        "expected_trials": None,
    }
    for uj in find_uncut_jsons(prompt_path):
        try:
            with open(uj) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for key in ("task_id", "task_name", "round_num", "prompt_idx", "prompt_text"):
            if data.get(key) is not None:
                out[key] = data[key]
        rec_info = data.get("recording_info", {}) or {}
        if rec_info.get("expected_trials") is not None:
            cur = out["expected_trials"] or 0
            out["expected_trials"] = max(int(cur), int(rec_info["expected_trials"]))
    return out


def _read_episode_metadata(processed_path: Path) -> list[dict[str, Any]]:
    eps: list[dict[str, Any]] = []
    if not processed_path.exists():
        return eps
    for ep_dir in sorted(processed_path.iterdir()):
        if not ep_dir.is_dir() or not ep_dir.name.startswith("ep"):
            continue
        meta_file = ep_dir / "metadata.json"
        if not meta_file.exists():
            continue
        with open(meta_file) as f:
            data = json.load(f)
        eps.append({
            "name": ep_dir.name,
            "label": data.get("label", "unknown"),
            "steps": data.get("steps", 0),
            "episode_id": data.get("episode_id"),
        })
    return eps


def compute_manifest(
    task_id: int,
    round_num: int,
    prompt_idx: int,
    base: Path | None = None,
) -> dict[str, Any]:
    """Aggregate per-episode metadata + uncut headers into a manifest dict.

    Writes the manifest to `<prompt_dir>/processed/manifest.json` and returns it.
    `success_rate` is included only when the strict completion criterion holds
    (successes + failures >= expected_trials, resets excluded).
    """
    pd = prompt_dir(task_id, round_num, prompt_idx, base)
    proc = processed_dir(pd)
    proc.mkdir(parents=True, exist_ok=True)

    header = _aggregate_uncut_metadata(pd)
    episodes = _read_episode_metadata(proc)

    successes = sum(1 for e in episodes if e["label"] == "success")
    failures = sum(1 for e in episodes if e["label"] == "failure")
    # Resets are not extracted into per-episode metadata, but pull the count
    # straight from the uncut.json summaries so the manifest tells the full story.
    resets = 0
    for uj in find_uncut_jsons(pd):
        try:
            with open(uj) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        resets += int((data.get("summary") or {}).get("resets", 0))

    expected = int(header["expected_trials"]) if header["expected_trials"] else None
    valid = successes + failures
    complete = expected is not None and valid >= expected

    existing: dict[str, Any] = {}
    manifest_file = proc / "manifest.json"
    if manifest_file.exists():
        try:
            with open(manifest_file) as f:
                existing = json.load(f)
        except (OSError, json.JSONDecodeError):
            existing = {}

    manifest: dict[str, Any] = {
        "task_id": header["task_id"] if header["task_id"] is not None else task_id,
        "task_name": header["task_name"],
        "round_num": header["round_num"] if header["round_num"] is not None else round_num,
        "prompt_idx": header["prompt_idx"] if header["prompt_idx"] is not None else prompt_idx,
        "prompt_text": header["prompt_text"],
        "expected_trials": expected,
        "complete": complete,
        "episodes": episodes,
        "processed_at": datetime.now().isoformat(),
        "synced": existing.get("synced", False),
        "last_sync": existing.get("last_sync"),
        "client_path": existing.get("client_path"),
    }

    if complete:
        manifest["successes"] = successes
        manifest["failures"] = failures
        manifest["resets"] = resets
        manifest["success_rate"] = successes / valid if valid > 0 else 0.0
    else:
        manifest["progress"] = {
            "successes": successes,
            "failures": failures,
            "resets": resets,
            "valid": valid,
            "expected_trials": expected,
        }

    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def is_recorded_complete(prompt_path: Path) -> tuple[bool, dict[str, int]]:
    """Inspect raw uncut.json files only — true if successes+failures >= expected_trials."""
    from real.round_task import count_recorded_trials

    progress = count_recorded_trials(prompt_path)
    expected = progress["expected_trials"]
    return (expected > 0 and progress["valid"] >= expected), progress
