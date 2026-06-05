"""CLI for the prompt-optimization pipeline.

Exposes the same operations as the former MCP server as simple subcommands.
All output is JSON to stdout.

Usage:
    python -m real.promptopt_cli list-tasks
    python -m real.promptopt_cli list-rounds 0
    python -m real.promptopt_cli ready [--task-id 0]
    python -m real.promptopt_cli process 0 1 0 [--force]
    python -m real.promptopt_cli episode-frames 0 1 0
    python -m real.promptopt_cli tag 0 1 0 ep000_2026-05-04_12-34-56 success
    python -m real.promptopt_cli sync 0 1
    python -m real.promptopt_cli mark-synced 0 1
    python -m real.promptopt_cli submit 0 1 --prompts-json '...' --model claude-sonnet-4-6
    python -m real.promptopt_cli get-optimized 0 [--round 1]
    python -m real.promptopt_cli progress
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from real.mcp_server.config import BASE_DIR, CLIENT_SYNC_PATH
from real.mcp_server.transfer import (
    build_round_sync_command,
    get_local_episode_paths,
    is_same_machine,
    mark_round_synced,
)
from real.mcp_server.video_processor import (
    compute_manifest,
    find_uncut_videos,
    is_recorded_complete,
    process_single_video,
)
from real.round_task import (
    iter_prompt_dirs,
    iter_round_dirs,
    iter_task_ids,
    load_task_yaml,
    optimized_prompts_path,
    processed_dir,
    prompt_dir,
    prompts_json_path,
    round_dir,
    task_dir,
    write_prompts,
)


def _load_manifest(pd: Path) -> dict | None:
    mp = processed_dir(pd) / "manifest.json"
    if not mp.exists():
        return None
    with open(mp) as f:
        return json.load(f)


def _prompt_idx_from_dir(pd: Path) -> int | None:
    if not pd.name.startswith("prompt"):
        return None
    try:
        return int(pd.name[len("prompt"):])
    except ValueError:
        return None


def _out(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_list_tasks(_args: argparse.Namespace) -> None:
    if not BASE_DIR.exists():
        _out({"error": f"Base directory not found: {BASE_DIR}", "tasks": []})
        return
    tasks = []
    for tid in iter_task_ids():
        try:
            meta = load_task_yaml(tid)
        except (FileNotFoundError, ValueError) as e:
            tasks.append({"task_id": tid, "error": str(e)})
            continue
        rounds = sum(1 for _ in iter_round_dirs(tid))
        tasks.append({
            "task_id": tid,
            "task_name": meta.get("task_name"),
            "description": meta.get("description"),
            "expected_trials": meta.get("expected_trials"),
            "num_rounds": rounds,
        })
    _out({"base_dir": str(BASE_DIR), "tasks": tasks})


def cmd_list_rounds(args: argparse.Namespace) -> None:
    task_id = args.task_id
    if not task_dir(task_id).exists():
        _out({"error": f"task{task_id} not found under {BASE_DIR}"})
        return
    rounds = []
    for rd in iter_round_dirs(task_id):
        try:
            round_num = int(rd.name[len("round"):])
        except ValueError:
            continue
        prompts_in: dict[str, Any] | None = None
        pj = prompts_json_path(task_id, round_num)
        if pj.exists():
            with open(pj) as f:
                prompts_in = json.load(f)
        prompts_status = []
        for pd in iter_prompt_dirs(task_id, round_num):
            idx = _prompt_idx_from_dir(pd)
            if idx is None:
                continue
            recorded_complete, progress = is_recorded_complete(pd)
            manifest = _load_manifest(pd)
            prompts_status.append({
                "prompt_idx": idx,
                "recorded_complete": recorded_complete,
                "progress": progress,
                "processed": manifest is not None,
                "complete": (manifest or {}).get("complete", False),
                "success_rate": (manifest or {}).get("success_rate"),
                "synced": (manifest or {}).get("synced", False),
            })
        opt = optimized_prompts_path(task_id, round_num)
        rounds.append({
            "round_num": round_num,
            "input_prompts": [p.get("text") for p in (prompts_in or {}).get("prompts", [])],
            "prompts_status": prompts_status,
            "optimized_prompts_present": opt.exists(),
        })
    _out({"task_id": task_id, "rounds": rounds})


def cmd_ready(args: argparse.Namespace) -> None:
    targets = [args.task_id] if args.task_id is not None else list(iter_task_ids())
    ready: list[dict[str, Any]] = []
    for tid in targets:
        for rd in iter_round_dirs(tid):
            try:
                round_num = int(rd.name[len("round"):])
            except ValueError:
                continue
            for pd in iter_prompt_dirs(tid, round_num):
                idx = _prompt_idx_from_dir(pd)
                if idx is None:
                    continue
                recorded_complete, progress = is_recorded_complete(pd)
                if not recorded_complete:
                    continue
                manifest = _load_manifest(pd)
                if manifest is not None and manifest.get("complete"):
                    continue
                ready.append({
                    "task_id": tid,
                    "round_num": round_num,
                    "prompt_idx": idx,
                    "prompt_dir": str(pd),
                    "progress": progress,
                    "manifest_status": "missing" if manifest is None else "incomplete",
                })
    _out({"ready": ready, "count": len(ready)})


def cmd_process(args: argparse.Namespace) -> None:
    pd = prompt_dir(args.task_id, args.round_num, args.prompt_idx)
    if not pd.exists():
        _out({"error": f"Prompt dir not found: {pd}"})
        return
    videos = find_uncut_videos(pd)
    if not videos:
        _out({"error": f"No *_uncut.mp4 found in {pd}"})
        return
    recorded_complete, progress = is_recorded_complete(pd)
    if not recorded_complete and not args.force:
        _out({
            "error": "Recording is not complete; refusing to process. Pass --force to override.",
            "progress": progress,
        })
        return
    out = processed_dir(pd)
    out.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []
    episode_offset = 0
    for video in videos:
        meta = json.loads(video.with_suffix(".json").read_text())
        prompt_text = meta.get("prompt_text") or ""
        results = process_single_video(video, out, prompt_text, episode_offset)
        all_results.extend(results)
        episode_offset += len(meta.get("episodes", []))
    manifest = compute_manifest(args.task_id, args.round_num, args.prompt_idx)
    _out({
        "prompt_dir": str(pd),
        "videos_processed": len(videos),
        "episode_results": all_results,
        "manifest": manifest,
    })


def cmd_episode_frames(args: argparse.Namespace) -> None:
    pd = prompt_dir(args.task_id, args.round_num, args.prompt_idx)
    if not pd.exists():
        _out({"error": f"Prompt dir not found: {pd}"})
        return
    manifest = _load_manifest(pd)
    if manifest is None:
        _out({"error": "Not yet processed; run 'process' first."})
        return
    episodes = get_local_episode_paths(args.task_id, args.round_num, args.prompt_idx)
    _out({
        "task_id": manifest.get("task_id"),
        "task_name": manifest.get("task_name"),
        "round_num": manifest.get("round_num"),
        "prompt_idx": manifest.get("prompt_idx"),
        "prompt_text": manifest.get("prompt_text"),
        "expected_trials": manifest.get("expected_trials"),
        "complete": manifest.get("complete", False),
        "successes": manifest.get("successes"),
        "failures": manifest.get("failures"),
        "success_rate": manifest.get("success_rate"),
        "episodes": episodes,
    })


def cmd_tag(args: argparse.Namespace) -> None:
    label = args.label
    if label not in ("success", "failure", "reset"):
        _out({"error": f"Invalid label {label!r}. Must be success/failure/reset."})
        return
    pd = prompt_dir(args.task_id, args.round_num, args.prompt_idx)
    proc = processed_dir(pd)
    ep_dir = proc / args.episode_name
    if not ep_dir.exists():
        _out({"error": f"Episode dir not found: {ep_dir}"})
        return
    meta_file = ep_dir / "metadata.json"
    if not meta_file.exists():
        _out({"error": f"metadata.json missing at {meta_file}"})
        return
    with open(meta_file) as f:
        ep_meta = json.load(f)
    old_label = ep_meta.get("label")
    if label == "reset":
        for child in ep_dir.iterdir():
            child.unlink()
        ep_dir.rmdir()
    else:
        ep_meta["label"] = label
        ep_meta["relabeled_from"] = old_label
        ep_meta["relabeled_at"] = datetime.now().isoformat()
        with open(meta_file, "w") as f:
            json.dump(ep_meta, f, indent=2)
    manifest = compute_manifest(args.task_id, args.round_num, args.prompt_idx)
    _out({
        "status": "ok",
        "episode": args.episode_name,
        "old_label": old_label,
        "new_label": label,
        "manifest": manifest,
    })


def cmd_sync(args: argparse.Namespace) -> None:
    rd = round_dir(args.task_id, args.round_num)
    if not rd.exists():
        _out({"error": f"Nothing to sync. Round dir missing: {rd}"})
        return
    if is_same_machine():
        updated = mark_round_synced(args.task_id, args.round_num)
        _out({
            "status": "noop",
            "reason": "same-machine",
            "robot_path": str(rd),
            "client_path": str(rd),
            "manifests_marked_synced": [str(p) for p in updated],
        })
        return
    cmd = build_round_sync_command(args.task_id, args.round_num)
    _out({
        "status": "ok",
        "rsync_command": cmd,
        "robot_path": str(rd),
        "client_path": f"{CLIENT_SYNC_PATH}/task{args.task_id}/round{args.round_num}",
    })


def cmd_mark_synced(args: argparse.Namespace) -> None:
    rd = round_dir(args.task_id, args.round_num)
    if not rd.exists():
        _out({"error": f"Round dir missing: {rd}"})
        return
    updated = mark_round_synced(args.task_id, args.round_num)
    _out({
        "status": "ok",
        "manifests_updated": [str(p) for p in updated],
    })


def cmd_submit(args: argparse.Namespace) -> None:
    try:
        prompts = json.loads(args.prompts_json)
    except json.JSONDecodeError as e:
        _out({"error": f"Invalid JSON for --prompts-json: {e}"})
        return

    if not isinstance(prompts, list) or not prompts:
        _out({"error": "`prompts` must be a non-empty list of dicts."})
        return

    metadata = None
    if args.metadata_json:
        try:
            metadata = json.loads(args.metadata_json)
        except json.JSONDecodeError as e:
            _out({"error": f"Invalid JSON for --metadata-json: {e}"})
            return

    cleaned: list[dict[str, Any]] = []
    for i, p in enumerate(prompts):
        if not isinstance(p, dict) or "text" not in p:
            _out({"error": f"prompts[{i}] must be a dict with at least a 'text' field."})
            return
        text = str(p["text"]).strip()
        if not text:
            _out({"error": f"prompts[{i}].text is empty."})
            return
        entry: dict[str, Any] = {"text": text}
        if "rationale" in p:
            entry["rationale"] = str(p["rationale"])
        if "score" in p:
            try:
                entry["score"] = float(p["score"])
            except (TypeError, ValueError):
                _out({"error": f"prompts[{i}].score must be a number."})
                return
        cleaned.append(entry)

    task_id = args.task_id
    source_round = args.source_round
    target_round = source_round + 1
    try:
        task_meta = load_task_yaml(task_id)
    except (FileNotFoundError, ValueError) as e:
        _out({"error": str(e)})
        return
    task_name = str(task_meta.get("task_name", ""))
    expected_trials = int(task_meta.get("expected_trials", 50))

    payload = {
        "task_id": task_id,
        "task_name": task_name,
        "source_round": source_round,
        "target_round": target_round,
        "generated_at": datetime.now().isoformat(),
        "source_model": args.model,
        "metadata": metadata or {},
        "prompts": cleaned,
    }

    src_path = optimized_prompts_path(task_id, source_round)
    src_path.parent.mkdir(parents=True, exist_ok=True)
    with open(src_path, "w") as f:
        json.dump(payload, f, indent=2)

    next_payload = {
        "task_id": task_id,
        "task_name": task_name,
        "round_num": target_round,
        "expected_trials": expected_trials,
        "source": f"round{source_round}_optimization",
        "source_model": args.model,
        "generated_at": payload["generated_at"],
        "prompts": cleaned,
    }
    next_path = write_prompts(task_id, target_round, next_payload)

    _out({
        "status": "ok",
        "optimized_prompts_path": str(src_path),
        "next_round_prompts_path": str(next_path),
        "num_prompts": len(cleaned),
    })


def cmd_get_optimized(args: argparse.Namespace) -> None:
    task_id = args.task_id
    round_num = args.round

    if round_num is not None:
        path = optimized_prompts_path(task_id, round_num)
        if not path.exists():
            _out({"error": f"Not found: {path}"})
            return
        with open(path) as f:
            print(f.read())
        return

    latest = None
    for rd in iter_round_dirs(task_id):
        try:
            n = int(rd.name[len("round"):])
        except ValueError:
            continue
        if (rd / "optimized_prompts.json").exists():
            latest = n
    if latest is None:
        _out({"error": f"No optimized_prompts.json found under task{task_id}"})
        return
    with open(optimized_prompts_path(task_id, latest)) as f:
        print(f.read())


def cmd_progress(_args: argparse.Namespace) -> None:
    if not BASE_DIR.exists():
        _out({"error": f"Base directory not found: {BASE_DIR}"})
        return
    out: dict[str, Any] = {"base_dir": str(BASE_DIR), "tasks": {}}
    for tid in iter_task_ids():
        rounds: dict[str, Any] = {}
        for rd in iter_round_dirs(tid):
            try:
                round_num = int(rd.name[len("round"):])
            except ValueError:
                continue
            prompts: dict[str, Any] = {}
            for pd in iter_prompt_dirs(tid, round_num):
                idx = _prompt_idx_from_dir(pd)
                if idx is None:
                    continue
                manifest = _load_manifest(pd)
                recorded_complete, progress = is_recorded_complete(pd)
                prompts[f"prompt{idx}"] = {
                    "recorded_complete": recorded_complete,
                    "progress": progress,
                    "processed": manifest is not None,
                    "complete": (manifest or {}).get("complete", False),
                    "success_rate": (manifest or {}).get("success_rate"),
                    "synced": (manifest or {}).get("synced", False),
                }
            rounds[f"round{round_num}"] = {
                "prompts": prompts,
                "optimized_prompts_present": optimized_prompts_path(tid, round_num).exists(),
            }
        out["tasks"][f"task{tid}"] = rounds
    _out(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m real.promptopt_cli",
        description="Prompt-optimization pipeline CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-tasks", help="List all tasks")

    p = sub.add_parser("list-rounds", help="List rounds for a task")
    p.add_argument("task_id", type=int)

    p = sub.add_parser("ready", help="Find prompts ready to process")
    p.add_argument("--task-id", type=int, default=None)

    p = sub.add_parser("process", help="Extract frames and write manifest")
    p.add_argument("task_id", type=int)
    p.add_argument("round_num", type=int)
    p.add_argument("prompt_idx", type=int)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("episode-frames", help="Get episode frame paths and labels")
    p.add_argument("task_id", type=int)
    p.add_argument("round_num", type=int)
    p.add_argument("prompt_idx", type=int)

    p = sub.add_parser("tag", help="Override episode label")
    p.add_argument("task_id", type=int)
    p.add_argument("round_num", type=int)
    p.add_argument("prompt_idx", type=int)
    p.add_argument("episode_name", type=str)
    p.add_argument("label", choices=["success", "failure", "reset"])

    p = sub.add_parser("sync", help="Sync round data to client")
    p.add_argument("task_id", type=int)
    p.add_argument("round_num", type=int)

    p = sub.add_parser("mark-synced", help="Mark round manifests as synced")
    p.add_argument("task_id", type=int)
    p.add_argument("round_num", type=int)

    p = sub.add_parser("submit", help="Submit optimized prompts")
    p.add_argument("task_id", type=int)
    p.add_argument("source_round", type=int)
    p.add_argument("--prompts-json", required=True, help="JSON array of prompt dicts")
    p.add_argument("--model", required=True, help="Source model identifier")
    p.add_argument("--metadata-json", default=None, help="Optional JSON metadata")

    p = sub.add_parser("get-optimized", help="Read optimized prompts")
    p.add_argument("task_id", type=int)
    p.add_argument("--round", type=int, default=None)

    sub.add_parser("progress", help="Full progress rollup")

    args = parser.parse_args()
    dispatch = {
        "list-tasks": cmd_list_tasks,
        "list-rounds": cmd_list_rounds,
        "ready": cmd_ready,
        "process": cmd_process,
        "episode-frames": cmd_episode_frames,
        "tag": cmd_tag,
        "sync": cmd_sync,
        "mark-synced": cmd_mark_synced,
        "submit": cmd_submit,
        "get-optimized": cmd_get_optimized,
        "progress": cmd_progress,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
