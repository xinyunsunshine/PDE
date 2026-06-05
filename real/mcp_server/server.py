"""
MCP Server for the prompt-optimization data pipeline.

Run on the robot workstation. The cluster LLM / the cluster session connects via
SSH tunnel.

Usage: python -m real.mcp_server.server [--port PORT]

Tool surface (in workflow order):

  list_tasks()                                 -> all task<X> dirs with metadata
  list_rounds(task_id)                         -> rounds + prompts under a task
  list_ready_to_process(task_id?)              -> prompt dirs whose recording is
                                                  complete but not processed yet
                                                  (this is the polling tool for
                                                  the cluster session)
  process_videos(task_id, round_num, prompt_idx, force=False)
                                               -> extract frames + per-ep meta
                                                  + manifest with success_rate
  get_episode_frames(task_id, round_num, prompt_idx)
                                               -> client-side frame paths +
                                                  binary success labels for the
                                                  Qwen prompt assembly
  tag_episode(task_id, round_num, prompt_idx, episode_name, label)
                                               -> override label, recompute SR
  sync_to_client(task_id, round_num)           -> rsync command + manifest
  mark_sync_complete(task_id, round_num)       -> flip synced=true on manifests
  submit_optimized_prompts(task_id, source_round, prompts, source_model, metadata?)
                                               -> remote client → robot handoff
  get_optimized_prompts(task_id, round_num?)   -> read latest optimized prompts
  get_progress()                               -> per-task / per-round / per-prompt
                                                  rollup
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from real.mcp_server.config import BASE_DIR, CLIENT_SYNC_PATH, MCP_PORT, ROBOT_SSH_HOST
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
    find_uncut_jsons,
    iter_prompt_dirs,
    iter_round_dirs,
    iter_task_ids,
    load_task_yaml,
    optimized_prompts_path,
    processed_dir,
    prompt_dir,
    prompt_dir as _prompt_dir,
    prompts_json_path,
    round_dir,
    task_dir,
    write_prompts,
)

mcp = FastMCP(
    "promptopt",
    instructions="Tools for the real-robot prompt-optimization pipeline.",
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


@mcp.tool()
def list_tasks() -> str:
    """List all task<X> directories under PROMPTOPT_BASE_DIR with their task.yaml metadata."""
    if not BASE_DIR.exists():
        return json.dumps({"error": f"Base directory not found: {BASE_DIR}", "tasks": []})

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
    return json.dumps({"base_dir": str(BASE_DIR), "tasks": tasks}, indent=2)


@mcp.tool()
def list_rounds(task_id: int) -> str:
    """List all rounds under a task, with per-prompt completion + processing status.

    Args:
        task_id: The task index (e.g. 0 for task0).
    """
    if not task_dir(task_id).exists():
        return json.dumps({"error": f"task{task_id} not found under {BASE_DIR}"})

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

    return json.dumps({"task_id": task_id, "rounds": rounds}, indent=2)


@mcp.tool()
def list_ready_to_process(task_id: int | None = None) -> str:
    """Find prompt dirs whose recording is complete but processed manifest is missing/incomplete.

    The the cluster session polls this. A prompt is "ready" iff its uncut.json files
    show successes+failures >= expected_trials (resets ignored) AND no
    manifest.json exists yet OR the existing manifest has complete=false.

    Args:
        task_id: Restrict to a single task (None = scan all tasks).
    """
    targets = [task_id] if task_id is not None else iter_task_ids()
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

    return json.dumps({"ready": ready, "count": len(ready)}, indent=2)


@mcp.tool()
def process_videos(
    task_id: int,
    round_num: int,
    prompt_idx: int,
    force: bool = False,
) -> str:
    """Extract frames for one prompt's recordings and write its manifest.

    Refuses if the prompt dir is not "recording-complete" (successes+failures >=
    expected_trials) unless force=True. Reset-labeled episodes are skipped.

    Args:
        task_id: Task index.
        round_num: Round number.
        prompt_idx: Prompt index within the round.
        force: Bypass the strict completion check.
    """
    pd = _prompt_dir(task_id, round_num, prompt_idx)
    if not pd.exists():
        return json.dumps({"error": f"Prompt dir not found: {pd}"})

    videos = find_uncut_videos(pd)
    if not videos:
        return json.dumps({"error": f"No *_uncut.mp4 found in {pd}"})

    recorded_complete, progress = is_recorded_complete(pd)
    if not recorded_complete and not force:
        return json.dumps({
            "error": "Recording is not complete; refusing to process. Pass force=True to override.",
            "progress": progress,
        }, indent=2)

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

    manifest = compute_manifest(task_id, round_num, prompt_idx)

    return json.dumps({
        "prompt_dir": str(pd),
        "videos_processed": len(videos),
        "episode_results": all_results,
        "manifest": manifest,
    }, indent=2, default=str)


@mcp.tool()
def get_episode_frames(task_id: int, round_num: int, prompt_idx: int) -> str:
    """Return client-side frame paths + per-episode binary success labels + per-prompt SR.

    This is the canonical payload to feed the cluster LLM (Qwen). Frame paths are
    rewritten to the client side so the caller can read them directly after
    rsync.

    Args:
        task_id: Task index.
        round_num: Round number.
        prompt_idx: Prompt index within the round.
    """
    pd = _prompt_dir(task_id, round_num, prompt_idx)
    if not pd.exists():
        return json.dumps({"error": f"Prompt dir not found: {pd}"})

    manifest = _load_manifest(pd)
    if manifest is None:
        return json.dumps({"error": "Not yet processed; call process_videos first."})

    episodes = get_local_episode_paths(task_id, round_num, prompt_idx)

    payload: dict[str, Any] = {
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
    }
    return json.dumps(payload, indent=2)


@mcp.tool()
def tag_episode(
    task_id: int,
    round_num: int,
    prompt_idx: int,
    episode_name: str,
    label: str,
) -> str:
    """Override the label on a processed episode and recompute the manifest.

    `label` must be one of "success", "failure", or "reset". Tagging an existing
    episode as "reset" deletes its frame extraction (a reset doesn't count
    toward N and doesn't get sent to the LLM).

    Args:
        task_id: Task index.
        round_num: Round number.
        prompt_idx: Prompt index.
        episode_name: Per-episode subdir name (e.g. "ep000_2026-05-04_12-34-56").
        label: New label: "success" | "failure" | "reset".
    """
    if label not in ("success", "failure", "reset"):
        return json.dumps({"error": f"Invalid label {label!r}. Must be success/failure/reset."})

    pd = _prompt_dir(task_id, round_num, prompt_idx)
    proc = processed_dir(pd)
    ep_dir = proc / episode_name
    if not ep_dir.exists():
        return json.dumps({"error": f"Episode dir not found: {ep_dir}"})

    meta_file = ep_dir / "metadata.json"
    if not meta_file.exists():
        return json.dumps({"error": f"metadata.json missing at {meta_file}"})

    with open(meta_file) as f:
        ep_meta = json.load(f)
    old_label = ep_meta.get("label")

    if label == "reset":
        # Resets aren't part of the LLM payload; remove the directory entirely.
        for child in ep_dir.iterdir():
            child.unlink()
        ep_dir.rmdir()
    else:
        ep_meta["label"] = label
        ep_meta["relabeled_from"] = old_label
        ep_meta["relabeled_at"] = datetime.now().isoformat()
        with open(meta_file, "w") as f:
            json.dump(ep_meta, f, indent=2)

    manifest = compute_manifest(task_id, round_num, prompt_idx)
    return json.dumps({
        "status": "ok",
        "episode": episode_name,
        "old_label": old_label,
        "new_label": label,
        "manifest": manifest,
    }, indent=2)


@mcp.tool()
def sync_to_client(task_id: int, round_num: int) -> str:
    """Build the rsync command for one whole round (all prompts).

    On a single-machine setup (where BASE_DIR == CLIENT_SYNC_PATH), returns
    `status: noop` instead — the data is already where the client expects it.
    Otherwise, run the returned command on the client; afterward call
    mark_sync_complete.

    Args:
        task_id: Task index.
        round_num: Round number.
    """
    rd = round_dir(task_id, round_num)
    if not rd.exists():
        return json.dumps({"error": f"Nothing to sync. Round dir missing: {rd}"})

    if is_same_machine():
        # No rsync needed; flip the synced flags so the manifests are tidy.
        updated = mark_round_synced(task_id, round_num)
        return json.dumps({
            "status": "noop",
            "reason": "same-machine",
            "robot_path": str(rd),
            "client_path": str(rd),
            "manifests_marked_synced": [str(p) for p in updated],
        }, indent=2)

    cmd = build_round_sync_command(task_id, round_num)
    return json.dumps({
        "status": "ok",
        "rsync_command": cmd,
        "robot_path": str(rd),
        "client_path": f"{CLIENT_SYNC_PATH}/task{task_id}/round{round_num}",
    }, indent=2)


@mcp.tool()
def mark_sync_complete(task_id: int, round_num: int) -> str:
    """Mark every per-prompt manifest under a round as synced.

    On single-machine setups, sync_to_client already does this; calling this
    afterward is harmless (idempotent).

    Args:
        task_id: Task index.
        round_num: Round number.
    """
    rd = round_dir(task_id, round_num)
    if not rd.exists():
        return json.dumps({"error": f"Round dir missing: {rd}"})
    updated = mark_round_synced(task_id, round_num)
    return json.dumps({
        "status": "ok",
        "manifests_updated": [str(p) for p in updated],
    }, indent=2)


@mcp.tool()
def submit_optimized_prompts(
    task_id: int,
    source_round: int,
    prompts: list[dict],
    source_model: str,
    metadata: dict | None = None,
) -> str:
    """Coauthor → robot. Persist optimized prompts and seed the next round.

    Writes `task<X>/round<source_round>/optimized_prompts.json` AND seeds
    `task<X>/round<source_round + 1>/prompts.json` with the same prompts list,
    so deploy.py can pick prompt_idx directly.

    Args:
        task_id: Task index.
        source_round: The round whose data was used to generate these prompts
            (the prompts are FOR round source_round + 1).
        prompts: List of dicts, each MUST have a "text" field. Optional:
            "rationale" (str), "score" (float).
        source_model: Identifier for the LLM that produced these (e.g.
            "qwen3-vl-235b-a22b-thinking").
        metadata: Free-form metadata (temperature, frames seen, input SRs, ...).
    """
    if not isinstance(prompts, list) or not prompts:
        return json.dumps({"error": "`prompts` must be a non-empty list of dicts."})

    cleaned: list[dict[str, Any]] = []
    for i, p in enumerate(prompts):
        if not isinstance(p, dict) or "text" not in p:
            return json.dumps({"error": f"prompts[{i}] must be a dict with at least a 'text' field."})
        text = str(p["text"]).strip()
        if not text:
            return json.dumps({"error": f"prompts[{i}].text is empty."})
        entry: dict[str, Any] = {"text": text}
        if "rationale" in p:
            entry["rationale"] = str(p["rationale"])
        if "score" in p:
            try:
                entry["score"] = float(p["score"])
            except (TypeError, ValueError):
                return json.dumps({"error": f"prompts[{i}].score must be a number."})
        cleaned.append(entry)

    target_round = source_round + 1
    try:
        task_meta = load_task_yaml(task_id)
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})
    task_name = str(task_meta.get("task_name", ""))
    expected_trials = int(task_meta.get("expected_trials", 50))

    payload = {
        "task_id": task_id,
        "task_name": task_name,
        "source_round": source_round,
        "target_round": target_round,
        "generated_at": datetime.now().isoformat(),
        "source_model": source_model,
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
        "source_model": source_model,
        "generated_at": payload["generated_at"],
        "prompts": cleaned,
    }
    next_path = write_prompts(task_id, target_round, next_payload)

    return json.dumps({
        "status": "ok",
        "optimized_prompts_path": str(src_path),
        "next_round_prompts_path": str(next_path),
        "num_prompts": len(cleaned),
    }, indent=2)


@mcp.tool()
def get_optimized_prompts(task_id: int, round_num: int | None = None) -> str:
    """Read optimized prompts. If round_num is None, return the latest round's.

    Args:
        task_id: Task index.
        round_num: Specific round to fetch (None = latest available).
    """
    if round_num is not None:
        path = optimized_prompts_path(task_id, round_num)
        if not path.exists():
            return json.dumps({"error": f"Not found: {path}"})
        with open(path) as f:
            return f.read()

    latest = None
    for rd in iter_round_dirs(task_id):
        try:
            n = int(rd.name[len("round"):])
        except ValueError:
            continue
        if (rd / "optimized_prompts.json").exists():
            latest = n
    if latest is None:
        return json.dumps({"error": f"No optimized_prompts.json found under task{task_id}"})
    with open(optimized_prompts_path(task_id, latest)) as f:
        return f.read()


@mcp.tool()
def get_progress() -> str:
    """Per-task / per-round / per-prompt rollup of recording, processing, and sync status."""
    if not BASE_DIR.exists():
        return json.dumps({"error": f"Base directory not found: {BASE_DIR}"})

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
    return json.dumps(out, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="Prompt Optimization MCP Server")
    parser.add_argument("--port", type=int, default=MCP_PORT)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"[MCP Server] Starting on {args.host}:{args.port}")
    print(f"[MCP Server] Base dir: {BASE_DIR}")
    print(f"[MCP Server] Robot SSH host: {ROBOT_SSH_HOST}")
    print(f"[MCP Server] Client sync path: {CLIENT_SYNC_PATH}")

    mcp.run(transport="sse", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
