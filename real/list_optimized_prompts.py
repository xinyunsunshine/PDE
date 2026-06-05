"""Print the optimized prompts produced for one task across all rounds.

Usage:
    python -m real.list_optimized_prompts --task-id 0
    python -m real.list_optimized_prompts --task-id 0 --round 2
"""

import argparse
import json
from typing import Any

from real.round_task import (
    iter_round_dirs,
    optimized_prompts_path,
    processed_dir,
    iter_prompt_dirs,
    task_dir,
)


def _format_round_block(payload: dict[str, Any], src_round: int, prior_summary: dict[str, Any] | None) -> str:
    target_round = payload.get("target_round", src_round + 1)
    model = payload.get("source_model", "?")
    when = payload.get("generated_at", "?")
    lines = [f"Round {src_round} -> Round {target_round}  [{model}, {when}]"]
    if prior_summary:
        lines.append("  Round " + str(src_round) + " observed:")
        for key in sorted(prior_summary):
            v = prior_summary[key]
            sr = v.get("success_rate")
            sr_s = f"{sr:.2f}" if isinstance(sr, (int, float)) else "?"
            text = v.get("prompt_text") or v.get("text") or ""
            lines.append(f"    {key}: SR={sr_s}  {text!r}")
    for i, p in enumerate(payload.get("prompts", [])):
        score = p.get("score")
        score_s = f"score {score:.2f}" if isinstance(score, (int, float)) else "score ?"
        text = p.get("text", "")
        lines.append(f"  [{i}] ({score_s}) {text!r}")
        if p.get("rationale"):
            lines.append(f"      why: {p['rationale']}")
    return "\n".join(lines)


def _prior_round_summary(task_id: int, src_round: int) -> dict[str, Any]:
    """Aggregate per-prompt SR + text from `round<src_round>/prompt<idx>/processed/manifest.json`."""
    summary: dict[str, Any] = {}
    for pd in iter_prompt_dirs(task_id, src_round):
        if not pd.name.startswith("prompt"):
            continue
        idx = pd.name
        mf = processed_dir(pd) / "manifest.json"
        if not mf.exists():
            continue
        with open(mf) as f:
            manifest = json.load(f)
        summary[idx] = {
            "prompt_text": manifest.get("prompt_text"),
            "success_rate": manifest.get("success_rate"),
            "successes": manifest.get("successes"),
            "failures": manifest.get("failures"),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="Specific round whose optimized_prompts.json to print (omit = all rounds).",
    )
    args = parser.parse_args()

    td = task_dir(args.task_id)
    if not td.exists():
        raise SystemExit(f"task{args.task_id} not found at {td}")

    rounds_to_show: list[int]
    if args.round is not None:
        rounds_to_show = [args.round]
    else:
        rounds_to_show = []
        for rd in iter_round_dirs(args.task_id):
            try:
                rounds_to_show.append(int(rd.name[len("round"):]))
            except ValueError:
                continue

    blocks = []
    for src_round in rounds_to_show:
        path = optimized_prompts_path(args.task_id, src_round)
        if not path.exists():
            continue
        with open(path) as f:
            payload = json.load(f)
        prior = _prior_round_summary(args.task_id, src_round)
        blocks.append(_format_round_block(payload, src_round, prior))

    if not blocks:
        print(f"No optimized_prompts.json found under {td}.")
        return
    print("\n\n".join(blocks))


if __name__ == "__main__":
    main()
