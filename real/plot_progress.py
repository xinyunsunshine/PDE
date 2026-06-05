"""Plot prompt-opt progress (per-round success rates) for one task.

Walks `task<X>/round<Y>/prompt<Z>/processed/manifest.json` for every round,
plus `task<X>/round<Y>/optimized_prompts.json` to flag kept_best winners.
Always emits CSV; optionally renders a PNG if matplotlib is available.

Usage:
    python -m real.plot_progress --task-id 1
    python -m real.plot_progress --task-id 1 --out /tmp/task1.csv
    python -m real.plot_progress --task-id 1 --out /tmp/task1.csv --plot /tmp/task1.png
    python -m real.plot_progress --task-id 1 --plot -                    # plot to stdout (PNG bytes)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from real.round_task import (
    iter_prompt_dirs,
    iter_round_dirs,
    processed_dir,
    task_dir,
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def collect_rows(task_id: int) -> list[dict[str, Any]]:
    """One row per (round, prompt) combination with completed manifests."""
    rows: list[dict[str, Any]] = []
    for rd in iter_round_dirs(task_id):
        try:
            round_num = int(rd.name[len("round"):])
        except ValueError:
            continue

        opt = _read_json(rd / "optimized_prompts.json") or {}
        kept_best = (opt.get("metadata") or {}).get("kept_best") or []

        for pd in iter_prompt_dirs(task_id, round_num):
            try:
                prompt_idx = int(pd.name[len("prompt"):])
            except ValueError:
                continue
            manifest = _read_json(processed_dir(pd) / "manifest.json")
            if manifest is None:
                continue
            text = manifest.get("prompt_text") or ""
            rows.append({
                "task_id": task_id,
                "round_num": round_num,
                "prompt_idx": prompt_idx,
                "prompt_text": text,
                "successes": manifest.get("successes"),
                "failures": manifest.get("failures"),
                "success_rate": manifest.get("success_rate"),
                "was_kept_best": text in kept_best,
                "complete": manifest.get("complete", False),
                "expected_trials": manifest.get("expected_trials"),
            })
    rows.sort(key=lambda r: (r["round_num"], r["prompt_idx"]))
    return rows


_FIELDS = [
    "task_id", "round_num", "prompt_idx", "prompt_text",
    "successes", "failures", "success_rate", "was_kept_best",
    "complete", "expected_trials",
]


def write_csv(rows: list[dict[str, Any]], out: Path | None) -> None:
    fp = sys.stdout if out is None else open(out, "w", newline="")
    try:
        w = csv.DictWriter(fp, fieldnames=_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    finally:
        if out is not None:
            fp.close()


def render_plot(rows: list[dict[str, Any]], task_id: int, out: str | Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping --plot output.", file=sys.stderr)
        return

    plottable = [r for r in rows if r["success_rate"] is not None]
    if not plottable:
        print(f"No completed manifests with success_rate found for task{task_id}.",
              file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Per-prompt scatter; kept_best highlighted.
    for r in plottable:
        kept = r["was_kept_best"]
        ax.scatter(
            r["round_num"], r["success_rate"],
            color="tab:orange" if kept else "tab:blue",
            edgecolors="black" if kept else "none",
            s=110 if kept else 55,
            zorder=3 if kept else 2,
            alpha=0.95 if kept else 0.7,
            label=None,
        )

    # Best-per-round line (max SR observed in each round).
    by_round: dict[int, list[float]] = {}
    for r in plottable:
        by_round.setdefault(r["round_num"], []).append(r["success_rate"])
    rs = sorted(by_round)
    best_per_round = [max(by_round[r]) for r in rs]
    if rs:
        ax.plot(rs, best_per_round, "-", color="tab:orange", alpha=0.5,
                label="max SR per round")

    # Manual legend handles for the scatter classes.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="tab:orange", markeredgecolor="black",
               markersize=10, label="kept_best"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="tab:blue", markersize=7, label="other prompts"),
        Line2D([0], [0], color="tab:orange", alpha=0.5, label="max SR per round"),
    ]
    ax.legend(handles=handles, loc="best")

    ax.set_xlabel("round_num")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(rs)
    ax.set_title(f"task{task_id} prompt-opt progress")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if str(out) == "-":
        fig.savefig(sys.stdout.buffer, format="png", dpi=120)
    else:
        fig.savefig(out, dpi=120)
        print(f"Plot saved to {out}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--out", type=str, default=None,
                   help="CSV output path (default: stdout)")
    p.add_argument("--plot", type=str, default=None,
                   help="PNG output path (use '-' for stdout). Needs matplotlib.")
    args = p.parse_args()

    td = task_dir(args.task_id)
    if not td.exists():
        sys.exit(f"task{args.task_id} not found at {td}")

    rows = collect_rows(args.task_id)
    if not rows:
        sys.exit(f"No manifests found under {td}. Have you run /promptopt-process yet?")

    write_csv(rows, Path(args.out) if args.out else None)
    if args.plot:
        render_plot(rows, args.task_id, args.plot)


if __name__ == "__main__":
    main()
