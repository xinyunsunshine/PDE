"""Compare VLM relabeling results across prompt variants.

Reads one or more results JSON files (from test_relabel_prompts.py) and
produces a side-by-side comparison table grouped by (step, traj).

Usage::

    python scripts/vlm_prompt_tuning/compare_prompts.py \
        scripts/vlm_prompt_tuning/results_v5_v7.json \
        scripts/vlm_prompt_tuning/results_v7_v8.json

    # Only compare specific prompts:
    python scripts/vlm_prompt_tuning/compare_prompts.py \
        results_v5_v7.json results_v7_v8.json \
        --prompts v5 v7 v8
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_results(*paths: str) -> list[dict]:
    all_results = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        all_results.extend(data["results"])
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "result_files", nargs="+", help="Result JSON files from test_relabel_prompts.py"
    )
    parser.add_argument(
        "--prompts", nargs="*", default=None,
        help="Only show these prompt variants (default: all found in results)"
    )
    parser.add_argument(
        "--output", default=None, help="Write comparison to file (default: stdout)"
    )
    args = parser.parse_args()

    results = load_results(*args.result_files)

    # Deduplicate: keep last entry per (step, traj, prompt)
    seen = {}
    for r in results:
        key = (r["step"], r["traj"], r["prompt"])
        seen[key] = r
    results = list(seen.values())

    # Determine prompt variants
    all_prompts = sorted({r["prompt"] for r in results})
    if args.prompts:
        prompts = [p for p in args.prompts if p in all_prompts]
    else:
        prompts = all_prompts

    # Group by (step, traj)
    grouped = defaultdict(dict)
    for r in results:
        if r["prompt"] in prompts:
            grouped[(r["step"], r["traj"])][r["prompt"]] = r

    keys = sorted(grouped.keys())

    # Build output
    lines = []
    lines.append(f"# Prompt Comparison: {', '.join(prompts)}")
    lines.append(f"# Trajectories: {len(keys)}")
    lines.append("")

    # --- Summary stats ---
    lines.append("## Summary Statistics")
    lines.append("")
    header = f"{'Prompt':<8} {'Parsed':>8} {'Nothing':>8} {'Unchanged':>10} {'Attempts':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for p in prompts:
        subset = [grouped[k][p] for k in keys if p in grouped[k]]
        n_total = len(subset)
        n_parsed = sum(1 for r in subset if r["parse_ok"])
        n_nothing = sum(
            1 for r in subset
            if r["parse_ok"] and r.get("parsed_instruction", "").lower() == "nothing"
        )
        n_unchanged = sum(
            1 for r in subset
            if r["parse_ok"] and r.get("parsed_instruction")
            and r["parsed_instruction"].lower() == r.get("original_instruction", "").lower()
        )
        n_attempts = sum(
            1 for r in subset
            if r["parse_ok"] and r.get("parsed_instruction")
            and "attempt" in r["parsed_instruction"].lower()
        )
        lines.append(
            f"{p:<8} {n_parsed:>4}/{n_total:<3} {n_nothing:>4}/{n_total:<3} "
            f"{n_unchanged:>6}/{n_total:<3} {n_attempts:>6}/{n_total:<3}"
        )
    lines.append("")

    # --- Agreement matrix ---
    if len(prompts) >= 2:
        lines.append("## Pairwise Agreement (same parsed instruction, case-insensitive)")
        lines.append("")
        header = f"{'':>8}" + "".join(f"{p:>10}" for p in prompts)
        lines.append(header)
        for p1 in prompts:
            row = f"{p1:>8}"
            for p2 in prompts:
                agree = 0
                total = 0
                for k in keys:
                    if p1 in grouped[k] and p2 in grouped[k]:
                        r1 = grouped[k][p1]
                        r2 = grouped[k][p2]
                        if r1["parse_ok"] and r2["parse_ok"]:
                            total += 1
                            i1 = (r1.get("parsed_instruction") or "").lower().strip()
                            i2 = (r2.get("parsed_instruction") or "").lower().strip()
                            if i1 == i2:
                                agree += 1
                if total > 0:
                    row += f"{agree}/{total:>3}".rjust(10)
                else:
                    row += f"{'—':>10}"
            lines.append(row)
        lines.append("")

    # --- Per-trajectory comparison ---
    lines.append("## Per-Trajectory Comparison")
    lines.append("")

    for step, traj in keys:
        data = grouped[(step, traj)]
        if not data:
            continue
        sample = next(iter(data.values()))
        orig = sample.get("original_instruction", "?")
        reward = sample.get("total_reward", "?")
        scene = sample.get("scene_objects", "")

        lines.append(f"### step={step} traj={traj}  reward={reward}")
        lines.append(f"  original: {orig}")
        if scene and scene != "not specified":
            lines.append(f"  objects:  {scene}")

        for p in prompts:
            if p in data:
                r = data[p]
                instr = r.get("parsed_instruction") or "(parse failed)"
                marker = ""
                if instr.lower() == "nothing":
                    marker = " [NOTHING]"
                elif instr.lower() == orig.lower():
                    marker = " [UNCHANGED]"
                elif "attempt" in instr.lower():
                    marker = " [ATTEMPT]"
                lines.append(f"  {p:<6}: {instr}{marker}")
            else:
                lines.append(f"  {p:<6}: (not run)")
        lines.append("")

    # --- "Nothing" disagreement highlights ---
    lines.append("## Disagreement Highlights (one says Nothing, other doesn't)")
    lines.append("")
    for step, traj in keys:
        data = grouped[(step, traj)]
        instructions = {}
        for p in prompts:
            if p in data and data[p]["parse_ok"]:
                instructions[p] = (data[p].get("parsed_instruction") or "").strip()
        nothings = {p for p, i in instructions.items() if i.lower() == "nothing"}
        non_nothings = {p for p, i in instructions.items() if i.lower() != "nothing"}
        if nothings and non_nothings:
            sample = next(iter(data.values()))
            orig = sample.get("original_instruction", "?")
            lines.append(f"  step={step} traj={traj} | original: {orig}")
            for p in prompts:
                if p in instructions:
                    lines.append(f"    {p:<6}: {instructions[p]}")
            lines.append("")

    output = "\n".join(lines)
    if args.output:
        Path(args.output).write_text(output)
        print(f"Comparison written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
