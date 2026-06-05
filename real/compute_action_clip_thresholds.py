"""Compute principled action-clip thresholds from the SFT training distribution.

Loads all episode parquets under a LeRobot-format dataset, computes the
per-control-step requested action delta (action[t] - state[t]) for each
frame, and reports distribution statistics so you can pick clip thresholds
that don't fire on legitimate training actions.

The recommendation logic:
  - Clip should NOT fire on training-distribution actions (clipping a real
    SFT-style action would actively hurt the policy).
  - Clip SHOULD fire on rare excursions (sensor glitches, OOD obs, model
    edge cases producing malformed chunks).
  - Threshold = p99.5 of training delta × (1 + safety_margin). p99.5 is
    the highest quantile we trust as legitimate; p99.9+ is contaminated by
    recording artifacts (frame drops, async timestamps, residual reach-
    to-start transitions even after --trim-boundary-frames).

Action layout: [xyz(3), rot6d(6), grip(1)]. The xyz clip used in
``real_rl.py`` is direction-preserving (L2 norm cap), so the recommended
threshold is the L2 norm of the per-step xyz delta. L∞ percentiles are
also reported as diagnostic context.

Usage:
    python -m real.compute_action_clip_thresholds \\
        --dataset-root /home/user/lrds/pde_real_sft

Output: prints a table to stdout. No file artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


def load_dataset_actions_and_states(
    dataset_root: Path,
    trim_boundary_frames: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate all (state, action) pairs across every episode parquet.

    ``trim_boundary_frames`` drops the first and last N frames of each
    episode. Frame 0 in LeRobot recordings is a "reach-to-start" transition:
    state is wherever the robot ended the previous take, action is the
    target start pose — often 50+ cm apart. These are NOT per-control-step
    actions and contaminate the upper percentiles. Default 1 drops only
    frame 0; bump to 2-3 if your recording also has a settle frame at the
    end. Set to 0 to keep everything.

    Returns (state, action) of shape (total_frames_after_trim, 10) each.
    """
    parquet_paths = sorted(dataset_root.glob("data/chunk-*/episode_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(
            f"No episode parquets found under {dataset_root}/data/chunk-*/. "
            "Pass --dataset-root pointing at a LeRobot-format dataset root."
        )
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    n_trimmed = 0
    n_total = 0
    for p in tqdm(parquet_paths, desc="Loading episodes"):
        t = pq.read_table(str(p), columns=["observation.state", "action"])
        s = np.stack(t["observation.state"].to_numpy())
        a = np.stack(t["action"].to_numpy())
        n_total += len(s)
        if trim_boundary_frames > 0 and len(s) > 2 * trim_boundary_frames:
            s = s[trim_boundary_frames : len(s) - trim_boundary_frames]
            a = a[trim_boundary_frames : len(a) - trim_boundary_frames]
            n_trimmed += 2 * trim_boundary_frames
        states.append(s)
        actions.append(a)
    state = np.concatenate(states, axis=0).astype(np.float32)
    action = np.concatenate(actions, axis=0).astype(np.float32)
    if trim_boundary_frames > 0:
        print(f"Trimmed {n_trimmed:,}/{n_total:,} boundary frames "
              f"({trim_boundary_frames} from each end of each episode).")
    return state, action


def report_block(name: str, deltas: np.ndarray, also_l2: bool = False) -> dict[str, dict[str, float]]:
    """Print percentile breakdown for one action-dim block.

    deltas: (N, D) per-frame per-dim deltas in robot space.
    Returns {"linf": pct_dict} and (if also_l2) {"l2": pct_dict}.
    L∞ (max-abs-over-dims) matches per-dim clip behavior.
    L2 (Euclidean norm)    matches direction-preserving norm clip behavior.
    """
    abs_deltas = np.abs(deltas)
    max_per_frame = abs_deltas.max(axis=-1)
    quantiles = [50, 90, 99, 99.5, 99.9, 99.99]

    def percentiles(arr: np.ndarray) -> dict[str, float]:
        out = {f"p{q}": float(np.percentile(arr, q)) for q in quantiles}
        out["max"] = float(arr.max())
        return out

    pct_linf = percentiles(max_per_frame)
    print(f"\n  [{name}]  per-frame L∞ (max-abs over {deltas.shape[1]} dims):")
    for label, val in pct_linf.items():
        print(f"    {label:>7s}: {val:.5f}")

    out: dict[str, dict[str, float]] = {"linf": pct_linf}
    if also_l2:
        l2_per_frame = np.linalg.norm(deltas, axis=-1)   # (N,) Euclidean norm
        pct_l2 = percentiles(l2_per_frame)
        print(f"  [{name}]  per-frame L2 (Euclidean norm — for direction-preserving clip):")
        for label, val in pct_l2.items():
            print(f"    {label:>7s}: {val:.5f}")
        out["l2"] = pct_l2
    return out


def recommend(name: str, pct: dict[str, float], safety_margin: float) -> tuple[float, float]:
    """Two recommendations:
        'p99'  = p99 × (1+margin) — fires on 1%% of training actions, tight.
        'p99.5' = p99.5 × (1+margin) — fires on 0.5%%, the default I'd ship.

    We deliberately do NOT recommend max() here. Even after trimming
    episode boundary frames, lerobot recordings can have transient
    discontinuities (state jumps, async timestamps) that appear as
    multi-cm "deltas" but aren't real per-control-step actions. p99.5 is
    the highest quantile we'd trust as legitimate; anything past that is
    contaminated by recording artifacts.
    """
    return pct["p99"] * (1.0 + safety_margin), pct["p99.5"] * (1.0 + safety_margin)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset-root", type=str, required=True,
                    help="Root of LeRobot-format dataset (contains data/, meta/).")
    ap.add_argument("--safety-margin", type=float, default=0.50,
                    help="Headroom above p99/p99.5 percentile (default 0.50 = 50%%, "
                         "since we want a comfortable buffer above legitimate actions).")
    ap.add_argument("--trim-boundary-frames", type=int, default=1,
                    help="Drop first/last N frames of each episode to remove "
                         "reach-to-start transitions (default 1).")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    state, action = load_dataset_actions_and_states(
        dataset_root, trim_boundary_frames=args.trim_boundary_frames,
    )
    delta = action - state                                # (N, 10)
    N = delta.shape[0]
    print(f"\nLoaded {N:,} frames across {N // 500} ~episodes (avg ~500 frames/ep).")
    print(f"Action layout: [xyz(3), rot6d(6), grip(1)]. All values in robot space.")

    pct_xyz = report_block("xyz",    delta[:, 0:3], also_l2=True)
    pct_rot = report_block("rot6d",  delta[:, 3:9], also_l2=True)
    pct_grip = report_block("grip",  delta[:, 9:10])

    print("\n" + "=" * 72)
    print("RECOMMENDED CLIP THRESHOLDS")
    print("=" * 72)
    print(f"Logic: clip should not fire on legitimate SFT-distribution actions.")
    print(f"  - 'p99 × {1 + args.safety_margin:.2f}': tight floor, fires on ~1%% of training actions.")
    print(f"  - 'p99.5 × {1 + args.safety_margin:.2f}': recommended default, fires on ~0.5%%.")
    print(f"max() and p99.9 are NOT used: even with --trim-boundary-frames "
          f"{args.trim_boundary_frames}, the upper tail is contaminated by "
          "recording artifacts.")
    print()

    margin = 1.0 + args.safety_margin
    xyz_l2_p995    = pct_xyz["l2"]["p99.5"] * margin
    xyz_l2_max     = pct_xyz["l2"]["max"] * margin
    rot_linf_p995  = pct_rot["linf"]["p99.5"] * margin
    grip_linf_p995 = pct_grip["linf"]["p99.5"] * margin

    print(f"\n  Direction-preserving xyz (L2) — for action_clip.xyz_norm_m:")
    print(f"    xyz_norm_m (p99.5 × {margin:.2f}): {xyz_l2_p995:.4f} m  ← recommended")
    print(f"    xyz_norm_m (max   × {margin:.2f}): {xyz_l2_max:.4f} m  (covers worst training frame)")
    print(f"\n  Optional per-dim L∞ caps (usually left null):")
    print(f"    rot6d_per_dim (p99.5 × {margin:.2f}): {rot_linf_p995:.4f}")
    print(f"    grip          (p99.5 × {margin:.2f}): {grip_linf_p995:.4f} m")

    print()
    print("Drop-in real_rl.yaml block:")
    print(f"  action_clip:")
    print(f"    xyz_norm_m: {xyz_l2_p995:.4f}   # L2 cap on per-step xyz delta, direction preserved")
    print(f"    rot6d_per_dim: null      # rot6d has no clean per-component physical bound")
    print(f"    grip: null               # gripper hardware-bounded in real/server/robot.py:23")


if __name__ == "__main__":
    main()
