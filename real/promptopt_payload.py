"""Local-side helpers for the Sonnet-driven prompt-opt loop.

Builds the round-level payload the analyzer agent needs from the synced data
on the client (`~/real_prompt_opt/task<X>/round<Y>/prompt<Z>/processed/...`).
Persists each round's analysis JSON. Pulls accumulated `prompt_effects` from
prior rounds for memory continuity.

All path resolution flows through `real.round_task` so this stays in lock-step
with the recording side and MCP server.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from real.round_task import (
    base_dir,
    iter_prompt_dirs,
    iter_round_dirs,
    load_task_yaml,
    optimized_prompts_path,
    processed_dir,
    prompt_dir,
    round_dir,
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _episode_frames(ep_dir: Path) -> list[str]:
    return [str(p) for p in sorted(ep_dir.glob("frame_*.png"))]


def _collect_episodes(proc_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Walk processed/ep*/metadata.json; return (successes, failures) episode dicts."""
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if not proc_dir.exists():
        return successes, failures
    for ep_dir in sorted(proc_dir.iterdir()):
        if not ep_dir.is_dir() or not ep_dir.name.startswith("ep"):
            continue
        meta = _read_json(ep_dir / "metadata.json") or {}
        entry = {
            "name": ep_dir.name,
            "label": meta.get("label", "unknown"),
            "steps": meta.get("steps", 0),
            "frame_paths": _episode_frames(ep_dir),
        }
        if entry["label"] == "success":
            successes.append(entry)
        elif entry["label"] == "failure":
            failures.append(entry)
    return successes, failures


def _representative_episodes(proc_dir: Path, max_videos: int) -> list[dict[str, Any]]:
    """Pick a representative rollout (or two): prefer one success + one failure."""
    if max_videos <= 0:
        return []
    successes, failures = _collect_episodes(proc_dir)
    chosen: list[dict[str, Any]] = []
    if max_videos == 1:
        chosen = (successes or failures or [])[:1]
    else:
        if successes:
            chosen.append(successes[0])
        if failures and len(chosen) < max_videos:
            chosen.append(failures[0])
        pool = successes[1:] + failures[1:]
        while len(chosen) < max_videos and pool:
            chosen.append(pool.pop(0))
    return chosen


def _pick_success_episode(proc_dir: Path) -> dict[str, Any] | None:
    """One episode for a prior prompt: prefer success; fall back to failure if no successes."""
    successes, failures = _collect_episodes(proc_dir)
    if successes:
        return successes[0]
    if failures:
        return failures[0]
    return None


def load_top_prior_prompts(
    task_id: int,
    round_num: int,
    k: int = 3,
    base: Path | None = None,
) -> list[dict[str, Any]]:
    """Top-k prompts by SR across **all** prior rounds, each with one success rollout.

    Returns a list of dicts shaped:
        {round_num, prompt_idx, prompt_text, success_rate, successes, failures,
         video: {name, label, steps, frame_paths} | None}
    Ranked by SR descending, with newer rounds breaking ties (so we lean on more
    recent evidence). If a top prompt has no success rollouts, the video falls
    back to a failure rollout. Skipped entirely if neither success nor failure
    rollouts exist (i.e. resets only).
    """
    b = base or base_dir()
    candidates: list[dict[str, Any]] = []
    for rd in iter_round_dirs(task_id, b):
        try:
            r = int(rd.name[len("round"):])
        except ValueError:
            continue
        if r >= round_num:
            continue
        for pd in iter_prompt_dirs(task_id, r, b):
            manifest = _read_json(processed_dir(pd) / "manifest.json")
            if manifest is None or manifest.get("success_rate") is None:
                continue
            try:
                pidx = int(pd.name[len("prompt"):])
            except ValueError:
                continue
            candidates.append({
                "round_num": r,
                "prompt_idx": pidx,
                "prompt_text": manifest.get("prompt_text") or "",
                "success_rate": manifest.get("success_rate"),
                "successes": manifest.get("successes"),
                "failures": manifest.get("failures"),
                "_proc_dir": processed_dir(pd),
            })
    # Sort: SR desc, then more recent rounds first, then prompt_idx asc.
    candidates.sort(
        key=lambda c: (-(c["success_rate"] or 0.0), -c["round_num"], c["prompt_idx"])
    )
    top: list[dict[str, Any]] = []
    for cand in candidates:
        video = _pick_success_episode(cand["_proc_dir"])
        del cand["_proc_dir"]
        if video is None:
            continue
        cand["video"] = video
        top.append(cand)
        if len(top) >= k:
            break
    return top


def load_prior_round_history(
    task_id: int,
    round_num: int,
    base: Path | None = None,
) -> list[dict[str, Any]]:
    """Compact per-round text history of every prompt tested in every prior round.

    Returns a list (oldest round first) of:
        {round_num,
         kept_best: [<text>, ...],
         prompts: [
            {prompt_idx, prompt_text, success_rate, successes, failures,
             behavior, was_kept_best}
         ]}
    `behavior` is the prompt's entry from that round's analysis.json[prompt_effects]
    (Sonnet's own one-liner observation), or "" if not present.
    """
    b = base or base_dir()
    history: list[dict[str, Any]] = []
    for rd in iter_round_dirs(task_id, b):
        try:
            r = int(rd.name[len("round"):])
        except ValueError:
            continue
        if r >= round_num:
            continue
        analysis = _read_json(rd / "analysis.json") or {}
        opt = _read_json(rd / "optimized_prompts.json") or {}
        kept_best = (
            (opt.get("metadata") or {}).get("kept_best")
            or analysis.get("kept_best")
            or []
        )
        prompt_effects = analysis.get("prompt_effects") or {}
        prompts_in_round: list[dict[str, Any]] = []
        for pd in iter_prompt_dirs(task_id, r, b):
            try:
                pidx = int(pd.name[len("prompt"):])
            except ValueError:
                continue
            manifest = _read_json(processed_dir(pd) / "manifest.json")
            if manifest is None:
                continue
            text = manifest.get("prompt_text") or ""
            prompts_in_round.append({
                "prompt_idx": pidx,
                "prompt_text": text,
                "success_rate": manifest.get("success_rate"),
                "successes": manifest.get("successes"),
                "failures": manifest.get("failures"),
                "behavior": str(prompt_effects.get(text, "")),
                "was_kept_best": text in kept_best,
            })
        prompts_in_round.sort(key=lambda p: p["prompt_idx"])
        history.append({
            "round_num": r,
            "kept_best": kept_best,
            "prompts": prompts_in_round,
        })
    return history


def build_round_payload(
    task_id: int,
    round_num: int,
    base: Path | None = None,
    videos_per_top_prompt: int = 1,
    max_prompts_with_video: int = 3,
    top_prior_prompts: int = 3,
) -> dict[str, Any]:
    """Assemble the analyzer's input for one round.

    Reads each prompt's manifest + per-episode metadata. Picks top-N prompts by
    success_rate to receive video frames; the rest contribute text + SR only.
    Pulls prior-round best prompt's frames if the round's optimized_prompts.json
    or this round's predecessor is around.
    """
    b = base or base_dir()
    task_meta = load_task_yaml(task_id, b)

    prompts_payload: list[dict[str, Any]] = []
    for pd in iter_prompt_dirs(task_id, round_num, b):
        try:
            prompt_idx = int(pd.name[len("prompt"):])
        except ValueError:
            continue
        manifest = _read_json(processed_dir(pd) / "manifest.json")
        if manifest is None or manifest.get("successes") is None:
            # Prompt hasn't been processed yet (or wasn't complete enough to
            # write SR). Skip — the LLM payload only contains real data.
            continue
        prompts_payload.append({
            "prompt_idx": prompt_idx,
            "prompt_text": manifest.get("prompt_text") or "",
            "successes": manifest.get("successes"),
            "failures": manifest.get("failures"),
            "success_rate": manifest.get("success_rate"),
            "expected_trials": manifest.get("expected_trials"),
            "complete": manifest.get("complete", False),
            "include_video": False,  # filled in below
            "video": None,
            "_proc_dir": processed_dir(pd),
        })
    prompts_payload.sort(key=lambda p: p["prompt_idx"])

    # Top-K by SR get the video; ties broken by prompt_idx (stable).
    rankable = [p for p in prompts_payload if p["success_rate"] is not None]
    rankable.sort(key=lambda p: (-(p["success_rate"] or 0.0), p["prompt_idx"]))
    top_ids = {p["prompt_idx"] for p in rankable[:max_prompts_with_video]}

    for p in prompts_payload:
        if p["prompt_idx"] in top_ids:
            videos = _representative_episodes(p["_proc_dir"], videos_per_top_prompt)
            if videos:
                p["include_video"] = True
                p["video"] = videos[0] if videos_per_top_prompt == 1 else videos
        del p["_proc_dir"]

    return {
        "task_id": task_id,
        "task_name": task_meta.get("task_name"),
        "task_description": task_meta.get("description"),
        "round_num": round_num,
        "expected_trials": task_meta.get("expected_trials"),
        "prompts": prompts_payload,
        # Top-k prompts by SR across ALL prior rounds, each with one (preferably
        # success) rollout. Replaces the old `previous_best` field.
        "top_prior_prompts": load_top_prior_prompts(task_id, round_num, k=top_prior_prompts, base=b),
        # Full per-round text history of every prompt ever tried (no videos).
        "prior_round_history": load_prior_round_history(task_id, round_num, b),
    }


def save_round_analysis(
    task_id: int,
    round_num: int,
    analysis: dict[str, Any],
    base: Path | None = None,
) -> Path:
    """Write `round<N>/analysis.json` with the LLM's full output."""
    b = base or base_dir()
    rd = round_dir(task_id, round_num, b)
    rd.mkdir(parents=True, exist_ok=True)
    payload = dict(analysis)
    payload.setdefault("task_id", task_id)
    payload.setdefault("round_num", round_num)
    payload.setdefault("analyzed_at", datetime.now().isoformat())
    payload.setdefault("analyzed_by", "claude-sonnet-4-6")
    path = rd / "analysis.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_prior_prompt_effects(
    task_id: int,
    round_num: int,
    base: Path | None = None,
    max_entries: int = 10,
) -> dict[str, str]:
    """Walk `round<0..round_num-1>/analysis.json` and merge `prompt_effects`.

    Most recent observation wins per prompt. Returns at most `max_entries`
    entries (most recent first).
    """
    b = base or base_dir()
    merged: dict[str, str] = {}
    for rd in iter_round_dirs(task_id, b):
        try:
            n = int(rd.name[len("round"):])
        except ValueError:
            continue
        if n >= round_num:
            continue
        analysis = _read_json(rd / "analysis.json") or {}
        effects = analysis.get("prompt_effects") or {}
        for k, v in effects.items():
            merged[str(k)] = str(v)
    if len(merged) <= max_entries:
        return merged
    # Keep the last `max_entries` insertion-order entries.
    items = list(merged.items())[-max_entries:]
    return dict(items)


def load_round_analysis(
    task_id: int,
    round_num: int,
    base: Path | None = None,
) -> dict[str, Any] | None:
    """Read a previously-saved analysis.json for inspection / resume."""
    b = base or base_dir()
    return _read_json(round_dir(task_id, round_num, b) / "analysis.json")
