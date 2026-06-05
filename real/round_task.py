"""Shared filesystem layout for the prompt-optimization pipeline.

Single source of truth for the `task<X>/round<Y>/prompt<Z>/` convention used by
both `real/deploy.py` (recording) and `real/mcp_server/` (processing). Any code
that needs to locate a task, round, or prompt directory should go through this
module rather than rebuilding the path itself.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def base_dir() -> Path:
    """Root of the prompt-opt data tree (`PROMPTOPT_BASE_DIR`)."""
    return Path(os.environ.get("PROMPTOPT_BASE_DIR", "~/real_prompt_opt")).expanduser()


def task_dir(task_id: int, base: Path | None = None) -> Path:
    return (base or base_dir()) / f"task{task_id}"


def round_dir(task_id: int, round_num: int, base: Path | None = None) -> Path:
    return task_dir(task_id, base) / f"round{round_num}"


def prompt_dir(task_id: int, round_num: int, prompt_idx: int, base: Path | None = None) -> Path:
    return round_dir(task_id, round_num, base) / f"prompt{prompt_idx}"


def processed_dir(prompt_path: Path) -> Path:
    return prompt_path / "processed"


def task_yaml_path(task_id: int, base: Path | None = None) -> Path:
    return task_dir(task_id, base) / "task.yaml"


def prompts_json_path(task_id: int, round_num: int, base: Path | None = None) -> Path:
    return round_dir(task_id, round_num, base) / "prompts.json"


def optimized_prompts_path(task_id: int, round_num: int, base: Path | None = None) -> Path:
    return round_dir(task_id, round_num, base) / "optimized_prompts.json"


def manifest_path(task_id: int, round_num: int, prompt_idx: int, base: Path | None = None) -> Path:
    return processed_dir(prompt_dir(task_id, round_num, prompt_idx, base)) / "manifest.json"


def load_task_yaml(task_id: int, base: Path | None = None) -> dict[str, Any]:
    path = task_yaml_path(task_id, base)
    if not path.exists():
        raise FileNotFoundError(
            f"task.yaml not found at {path}. Create it with: task_id, task_name, "
            "expected_trials, seed_prompts."
        )
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if int(data.get("task_id", -1)) != task_id:
        raise ValueError(
            f"task.yaml at {path} declares task_id={data.get('task_id')!r} but was "
            f"loaded as task_id={task_id}."
        )
    return data


def load_prompts(task_id: int, round_num: int, base: Path | None = None) -> dict[str, Any]:
    """Load `round<Y>/prompts.json`. Auto-seeds round 0 from `task.yaml.seed_prompts`."""
    path = prompts_json_path(task_id, round_num, base)
    if path.exists():
        with open(path) as f:
            return json.load(f)

    if round_num != 0:
        raise FileNotFoundError(
            f"prompts.json missing at {path}. For round>0, this file is normally "
            "seeded by submit_optimized_prompts() from the previous round."
        )

    task_meta = load_task_yaml(task_id, base)
    seeds = task_meta.get("seed_prompts") or []
    if not seeds:
        raise ValueError(
            f"task.yaml at {task_yaml_path(task_id, base)} has no seed_prompts; "
            "cannot auto-seed round0/prompts.json."
        )
    payload = {
        "task_id": task_id,
        "task_name": task_meta.get("task_name", ""),
        "round_num": 0,
        "expected_trials": int(task_meta.get("expected_trials", 50)),
        "source": "seed",
        "generated_at": datetime.now().isoformat(),
        "prompts": [{"text": str(p)} for p in seeds],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def write_prompts(task_id: int, round_num: int, payload: dict[str, Any], base: Path | None = None) -> Path:
    path = prompts_json_path(task_id, round_num, base)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def find_uncut_jsons(prompt_path: Path) -> list[Path]:
    """All `*_uncut.json` sidecars in a prompt directory, sorted by name (chronological)."""
    if not prompt_path.exists():
        return []
    return sorted(p for p in prompt_path.iterdir() if p.suffix == ".json" and "_uncut" in p.stem)


def count_recorded_trials(prompt_path: Path) -> dict[str, int]:
    """Aggregate trial counts from existing `*_uncut.json` files in a prompt dir.

    Returns:
        ``{successes, failures, resets, valid, expected_trials}`` where
        ``valid = successes + failures`` (resets do not count toward N) and
        ``expected_trials`` is the max value seen across all files.

    Used by deploy.py for resume support and by the MCP server for completion
    gating. Pure JSON walking — no cv2 or other heavy deps.
    """
    successes = failures = resets = 0
    expected = 0
    for uj in find_uncut_jsons(prompt_path):
        try:
            with open(uj) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        summary = data.get("summary") or {}
        successes += int(summary.get("successes", 0))
        failures += int(summary.get("failures", 0))
        resets += int(summary.get("resets", 0))
        rec_info = data.get("recording_info") or {}
        if rec_info.get("expected_trials") is not None:
            expected = max(expected, int(rec_info["expected_trials"]))
    return {
        "successes": successes,
        "failures": failures,
        "resets": resets,
        "valid": successes + failures,
        "expected_trials": expected,
    }


def iter_prompt_dirs(task_id: int, round_num: int, base: Path | None = None):
    """Yield existing `prompt<Z>/` paths under a round, sorted by prompt_idx."""
    rd = round_dir(task_id, round_num, base)
    if not rd.exists():
        return
    pairs = []
    for d in rd.iterdir():
        if not d.is_dir() or not d.name.startswith("prompt"):
            continue
        try:
            idx = int(d.name[len("prompt"):])
        except ValueError:
            continue
        pairs.append((idx, d))
    for _, d in sorted(pairs):
        yield d


def iter_round_dirs(task_id: int, base: Path | None = None):
    """Yield existing `round<Y>/` paths under a task, sorted by round_num."""
    td = task_dir(task_id, base)
    if not td.exists():
        return
    pairs = []
    for d in td.iterdir():
        if not d.is_dir() or not d.name.startswith("round"):
            continue
        try:
            n = int(d.name[len("round"):])
        except ValueError:
            continue
        pairs.append((n, d))
    for _, d in sorted(pairs):
        yield d


def iter_task_ids(base: Path | None = None) -> list[int]:
    b = base or base_dir()
    if not b.exists():
        return []
    ids = []
    for d in b.iterdir():
        if not d.is_dir() or not d.name.startswith("task"):
            continue
        try:
            ids.append(int(d.name[len("task"):]))
        except ValueError:
            continue
    return sorted(ids)
