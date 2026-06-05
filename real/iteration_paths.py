"""Iteration-aware path resolution for the bifranka inference side.

The trainer-side `RealworldEmbodiedRunner` delivers files into a `<run_root>/`
tree on bifranka. Per iteration K, the layout is:

    <run_root>/
        iteration{K:03d}/
            itr{K:02d}_prompt.json         ← prompts FOR this iteration
            train_after_itr{K-1}_model.pt  ← ckpt produced by trainer AFTER iter K-1
            rollout000.h5 ... rollout031.h5
            iteration_meta.json
            iteration_complete.json

Naming notes:
- Ckpt that was *trained on iteration K's data* lands at
  ``iteration{K:03d}/train_after_itr{K}_model.pt`` (same dir as that iter's
  rollouts). The trainer writes this via scp after PPO.
- For iteration K+1, the inference server loads
  ``iteration{K:03d}/train_after_itr{K}_model.pt`` — i.e. the file from the
  PREVIOUS iteration's directory. For K=0 there is no previous iteration, so
  the bootstrap SFT init is used instead.
- Prompts for iteration K live INSIDE ``iteration{K:03d}/`` (operator drops
  iter-0 prompts manually before bootstrap; for K≥1 the trainer scp's them).

The helpers in this module are imported by:
- ``real/real_rl.py``     → auto-resolve ``rl.prompts_path`` from ``rl.run_root``
- ``real/vla_server.py``  → auto-resolve ``inference.weights`` from
                            ``inference.run_root`` + ``inference.sft_weights``

Both files share the same "next iteration to run" computation so the two
processes never disagree about which iteration K is current.
"""

from __future__ import annotations

import re
from pathlib import Path

_ITER_DIR_RE = re.compile(r"^iteration(\d+)$")


def scan_iterations(run_root: Path) -> list[int]:
    """Return the sorted list of iteration numbers present under run_root."""
    run_root = Path(run_root)
    if not run_root.is_dir():
        return []
    nums: list[int] = []
    for entry in run_root.iterdir():
        if not entry.is_dir():
            continue
        m = _ITER_DIR_RE.match(entry.name)
        if m:
            nums.append(int(m.group(1)))
    return sorted(nums)


def next_iteration_to_run(run_root: Path) -> int:
    """Determine the next iteration K to run under run_root.

    Rule (mirrors ``real_rl.py::resolve_iteration_and_resume``):
      - If no iterationNNN/ dirs exist → K = 0.
      - Otherwise let K_max = highest existing N. If
        iteration{K_max:03d}/iteration_complete.json exists, K_max is done and
        the next iteration is K_max + 1. Else we're resuming K_max (partial).
    """
    run_root = Path(run_root)
    existing = scan_iterations(run_root)
    if not existing:
        return 0
    last = existing[-1]
    if (run_root / f"iteration{last:03d}" / "iteration_complete.json").is_file():
        return last + 1
    return last


def prompts_path_for_iteration(run_root: Path, K: int) -> Path:
    """The per-env prompts JSON the inference server consumes for iteration K.

    Convention: ``<run_root>/iteration{K:03d}/itr{K:02d}_prompt.json``.
    """
    return (
        Path(run_root) / f"iteration{K:03d}" / f"itr{K:02d}_prompt.json"
    )


def weights_path_for_iteration(
    run_root: Path,
    K: int,
    sft_weights: str | Path | None = None,
) -> Path:
    """The policy weights the inference server loads to run iteration K.

    For K=0 there is no prior PPO ckpt; returns ``sft_weights`` if provided.
    For K≥1, returns
    ``<run_root>/iteration{K-1:03d}/train_after_itr{K-1}_model.pt`` — the
    ckpt the trainer scp'd after training on iteration K-1's data.

    Raises FileNotFoundError if the resolved path does not exist on disk.
    Callers should poll / wait for the file if delivery is in flight.
    """
    if K == 0:
        if sft_weights is None:
            raise FileNotFoundError(
                "Iteration 0 requires inference.sft_weights to be set (no "
                "prior PPO ckpt exists yet). Pass an explicit SFT init path."
            )
        p = Path(sft_weights).expanduser()
        if not p.exists():
            raise FileNotFoundError(
                f"sft_weights path does not exist: {p}"
            )
        return p

    p = (
        Path(run_root)
        / f"iteration{K - 1:03d}"
        / f"train_after_itr{K - 1}_model.pt"
    )
    if not p.is_file():
        raise FileNotFoundError(
            f"Expected ckpt for iteration {K} at {p} but it does not exist. "
            "Wait for the trainer to deliver it (scp + atomic rename), then "
            "relaunch."
        )
    return p
