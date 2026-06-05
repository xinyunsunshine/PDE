"""Test sub-trajectory HER relabeling on a single video.

Mimics the HERProcessor pipeline for a single trajectory:
  1. request_hindsight_instruction  (prompt_version="subtraj_v2")
     → returns (instruction, cutoff_fraction)
  2. request_reward_eval on the prefix frames
     → returns (reward, _)

Uses HERVLMClient and her_utils from rlinf throughout.

Usage
-----
    python scripts/subtraj/test_subtraj_relabeling.py \\
        --video path/to/traj.mp4 \\
        --instruction "put both moka pots on the stove" \\
        --endpoint http://VLM_ENDPOINT_HOST:PORT/v1 \\
        --output scripts/subtraj/results
"""

import argparse
import os
import sys
from pathlib import Path

import torch

from rlinf.workers.actor.her_utils import (
    HER_VLM_MAX_FRAMES,
    _save_trajectory_video,
    _to_vlm_frame_uint8,
    select_frame_indices_for_vlm,
    select_frames_for_vlm,
)
from rlinf.workers.actor.her_vlm_client import HERVLMClient

DEFAULT_MODEL = "Qwen/Qwen3-VL-235B-A22B-Thinking-FP8"

# ---------------------------------------------------------------------------
# Video loading  (exception: input-video-to-frame-list utilities)
# ---------------------------------------------------------------------------


def _load_video_frames(video_path: str) -> tuple[list[torch.Tensor], float]:
    """Load all frames as HWC uint8 RGB torch.Tensor. Returns (frames, fps)."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frames = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frames.append(torch.from_numpy(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames, fps


# ---------------------------------------------------------------------------
# Frame folder saving
# ---------------------------------------------------------------------------


def _save_frame_folder(
    frames: list[torch.Tensor], output_dir: Path, name: str
) -> Path:
    """Save a list of frames as individual JPEGs into output_dir/name/.

    Files are named frame_{i:06d}.jpg where i is the index within the list.
    Clears the folder first so stale frames from prior runs don't linger.
    Uses _to_vlm_frame_uint8 from her_utils for tensor→uint8 conversion.
    Returns the folder path.
    """
    import shutil
    import cv2

    folder = output_dir / name
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for i, frame in enumerate(frames):
        uint8 = _to_vlm_frame_uint8(frame)
        bgr = cv2.cvtColor(uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(folder / f"frame_{i:06d}.jpg"), bgr)
    print(f"[SAVE] {name}: {len(frames)} frames → {folder}", flush=True)
    return folder


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _write_markdown_report(output_dir: Path, result: dict) -> Path:
    video_name = Path(result["video"]).name
    instruction = result.get("instruction")
    cutoff_fraction = result.get("cutoff_fraction")
    reward = result.get("reward")

    if instruction and instruction.lower() == "nothing":
        tag = "**[nothing]**"
    elif cutoff_fraction is not None:
        n_vlm = result.get("n_vlm_frames", 1) - 1
        total = result.get("total_frames", 1) - 1
        tag = (
            f"**[subtraj]** vlm_frame={result.get('vlm_cutoff_frame','?')}/{n_vlm}"
            f" → orig_frame={result.get('orig_cutoff_frame','?')}/{total}"
            f" → fraction={cutoff_fraction:.3f}"
        )
    else:
        tag = "**[whole-video]**"

    lines = [
        "# Sub-trajectory relabeling result\n",
        f"## {video_name}\n",
        f"{tag}  ",
        f"- **orig:** {result.get('original_instruction', '')}",
        f"- **new :** {instruction or '(parse failed)'}",
        f"- **reward:** {reward if reward is not None else 'N/A (nothing)'}",
        "",
    ]
    for key, label in [("vlm_video", "VLM frames video"), ("prefix_video", "Prefix video")]:
        path = result.get(key)
        if path and os.path.exists(path):
            lines.append(f"**{label}:** [{Path(path).name}]({Path(path).name})  ")

    if result.get("cutoff_frame_jpg"):
        path = result["cutoff_frame_jpg"]
        fname = Path(path).name
        lines.append(f"\n**Cutoff frame:**  \n![{fname}]({fname})  ")

    out = output_dir / "report.md"
    out.write_text("\n".join(lines))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument(
        "--video-mode", default="frames", choices=["mp4", "frames", "frames_no_thinking"],
    )
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--scene-objects", nargs="+", default=[], metavar="OBJ")
    parser.add_argument("--no-trim", action="store_true")
    parser.add_argument("--max-frames", type=int, default=HER_VLM_MAX_FRAMES)
    parser.add_argument("--output", default="scripts/subtraj/results")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    # Load all frames (video-to-frame-list exception)
    print(f"[VIDEO] loading {video_path.name} ...", flush=True)
    all_frames, src_fps = _load_video_frames(str(video_path))
    total_frames = len(all_frames)

    vlm_indices = select_frame_indices_for_vlm(total_frames, args.max_frames)
    n_vlm_frames = len(vlm_indices)
    print(
        f"[VIDEO] total_frames={total_frames}, fps={src_fps:.1f}, "
        f"duration={total_frames/max(src_fps,1):.1f}s, "
        f"vlm_frames={n_vlm_frames} (indices [{vlm_indices[0]}..{vlm_indices[-1]}])",
        flush=True,
    )

    raw_out: dict = {}

    def _warn(msg: str) -> None:
        print(f"  [WARN] {msg}", flush=True)
        if raw_out.get("response"):
            print(f"\n{'='*60}\n[VLM RAW RESPONSE]\n{'='*60}", flush=True)
            print(raw_out["response"], flush=True)
            print(f"{'='*60}\n", flush=True)

    client = HERVLMClient(
        endpoint=args.endpoint,
        model=args.model,
        video_mode=args.video_mode,
        api_key=api_key,
        max_frames=args.max_frames,
        log_warning_fn=_warn,
    )

    # ── Step 1: instruction generation (mirrors HERProcessor._generate_group_instructions)
    print(f"[VLM] step 1/2 — hindsight instruction ({args.video_mode}) ...", flush=True)
    instruction, cutoff_fraction = client.request_hindsight_instruction(
        all_frames,
        original_instruction=args.instruction,
        scene_objects=args.scene_objects or None,
        prompt_version="subtraj_v2",
        _raw_out=raw_out,
    )
    print(f"\n{'='*60}\n[VLM RAW RESPONSE]\n{'='*60}", flush=True)
    print(raw_out.get("response", "(no response captured)"), flush=True)
    print(f"{'='*60}\n", flush=True)
    print(f"  instruction={instruction!r}  cutoff_fraction={cutoff_fraction}", flush=True)

    # Save VLM-seen frames as a video and as individual JPEGs
    vlm_frames_subset = select_frames_for_vlm(all_frames, args.max_frames)
    vlm_video_path = str(output_dir / f"{stem}_vlm_frames.mp4")
    _save_trajectory_video(vlm_frames_subset, out_path=vlm_video_path)
    print(f"[SAVE] VLM frames video → {Path(vlm_video_path).name}", flush=True)

    stem_dir = output_dir / stem
    _save_frame_folder(all_frames, stem_dir, "all_frames")
    _save_frame_folder(vlm_frames_subset, stem_dir, "vlm_frames")

    result: dict = {
        "video": str(video_path),
        "original_instruction": args.instruction,
        "total_frames": total_frames,
        "n_vlm_frames": n_vlm_frames,
        "instruction": instruction,
        "cutoff_fraction": cutoff_fraction,
        "vlm_video": vlm_video_path,
        "cutoff_frame_jpg": None,
        "prefix_video": None,
        "reward": None,
        "vlm_cutoff_frame": None,
        "orig_cutoff_frame": None,
    }

    if instruction is None or instruction.lower() == "nothing":
        form = "[nothing]" if instruction else "[parse failed]"
        print(f"\n{form}")
        print(f"  orig : {args.instruction}")
        print(f"  new  : {instruction or '(parse failed)'}")
        _write_markdown_report(output_dir, result)
        return

    # ── Determine prefix frames (mirrors HERProcessor._get_traj_frames_with_cutoff)
    if cutoff_fraction is not None:
        n_keep = max(1, int(round(total_frames * cutoff_fraction)))
        prefix_frames = all_frames[:n_keep]

        # Compute orig/vlm frame indices for logging
        orig_cutoff = n_keep - 1
        result["orig_cutoff_frame"] = orig_cutoff
        # Recover the matching VLM frame index
        vlm_clamped = min(
            range(len(vlm_indices)), key=lambda i: abs(vlm_indices[i] - orig_cutoff)
        )
        result["vlm_cutoff_frame"] = vlm_clamped
        print(
            f"  [CONVERT] cutoff_fraction={cutoff_fraction:.4f}"
            f" → keep {n_keep}/{total_frames} frames"
            f" → orig_frame={orig_cutoff}/{total_frames-1}"
            f" → approx vlm_frame={vlm_clamped}/{n_vlm_frames-1}",
            flush=True,
        )

        # Save cutoff frame as JPEG
        import cv2
        cutoff_jpg = str(output_dir / f"{stem}_cutoff_orig{orig_cutoff:06d}.jpg")
        frame_uint8 = _to_vlm_frame_uint8(all_frames[orig_cutoff])
        cv2.imwrite(cutoff_jpg, cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR))
        result["cutoff_frame_jpg"] = cutoff_jpg
        print(f"[SAVE] cutoff frame → {Path(cutoff_jpg).name}", flush=True)

        _save_frame_folder(prefix_frames, stem_dir, "prefix_frames")

        if not args.no_trim:
            prefix_path = str(output_dir / f"{stem}_prefix.mp4")
            _save_trajectory_video(prefix_frames, out_path=prefix_path)
            result["prefix_video"] = prefix_path
            print(f"[SAVE] prefix video ({n_keep} frames) → {Path(prefix_path).name}", flush=True)
    else:
        prefix_frames = all_frames
        _save_frame_folder(all_frames, stem_dir, "prefix_frames")

    # ── Step 2: reward eval on the prefix (mirrors HERProcessor._run_parallel_reward_eval)
    print(f"[VLM] step 2/2 — reward eval ...", flush=True)
    reward, _ = client.request_reward_eval(prefix_frames, instruction)
    result["reward"] = reward
    print(f"  reward={reward}", flush=True)

    # Print summary
    if cutoff_fraction is not None:
        form = (
            f"[subtraj  orig_frame={result['orig_cutoff_frame']}/{total_frames-1}"
            f"  fraction={cutoff_fraction:.3f}]"
        )
    else:
        form = "[whole-video]"

    print(f"\n{form}")
    print(f"  orig   : {args.instruction}")
    print(f"  new    : {instruction}")
    print(f"  reward : {reward}")

    out_md = _write_markdown_report(output_dir, result)
    print(f"\nMarkdown report: {out_md}")


if __name__ == "__main__":
    main()
