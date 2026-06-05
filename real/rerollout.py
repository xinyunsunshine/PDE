"""Re-rollout previously-deployed prompts under SDE-noise on the VLA server.

Walks ``~/real_prompt_opt/task<X>/round<Y>/prompt<Z>/`` for prompts that
achieved SR > 0 in the original (no-noise) deterministic rollouts, and
re-runs each under the noise-enabled VLA server. Records video + sidecar
JSON per prompt in a separate output tree so the prompt-opt artifacts are
left alone, then writes a summary CSV comparing baseline SR vs. noise SR.

Two structural differences from ``deploy.py``:

  * Outer loop over (task, round, prompt) tuples — one orchestrator drives
    the robot/VLA-server connection across all prompts (no per-prompt
    re-init). Per-prompt label collection (s/f/r) is unchanged.
  * Inside the trial loop, no ``valid_start`` trimming or wait-for-dt loop:
    we execute ``chunk[0 : execute_k]`` directly, matching ``real_rl.py``'s
    chunk semantics rather than ``deploy.py``'s inference-latency
    compensation. (The point of this script is to test how prompts behave
    under SDE noise during real-world RL, where the same chunk semantics
    apply.)

Launch (in three terminals, like deploy.py):

    # 1) OSC server
    bash real/run_osc_server.sh

    # 2) VLA server (noise enabled — see real/config/vla_server.yaml)
    python -m real.vla_server \\
        inference.weights=/path/to/full_weights.pt \\
        inference.norm_stats=/path/to/norm_stats.json

    # 3) This script (REQUIRED: rerollout.noise_level matches what the
    #    VLA server is configured with)
    python -m real.rerollout rerollout.noise_level=0.10
"""

from __future__ import annotations

import csv
import dataclasses
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from real.env import FR3RealEnv
from real.real_rl import clip_action_to_state
from real.real_util import UncutVideoRecorder
from real.round_task import (
    count_recorded_trials,
    iter_prompt_dirs,
    iter_round_dirs,
    iter_task_ids,
    load_prompts,
    load_task_yaml,
    prompt_dir,
)
from real.vla_client import connect_vla_client

GLOBAL_CAMERA_NAME = "global"
WRIST_CAMERA_NAME = "wrist"


# ---------------------------------------------------------------------------
# Stdin helpers (same as deploy.py — copied so we don't import its private bits)
# ---------------------------------------------------------------------------

@contextmanager
def raw_stdin():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def key_pressed() -> Optional[str]:
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def _obs_to_openpi_input(obs: dict, prompt: str) -> dict:
    return {
        "observation/image": np.asarray(obs["images"][GLOBAL_CAMERA_NAME], dtype=np.uint8),
        "observation/wrist_image": np.asarray(obs["images"][WRIST_CAMERA_NAME], dtype=np.uint8),
        "observation/state": np.asarray(obs["state"], dtype=np.float32),
        "prompt": prompt,
    }


# ---------------------------------------------------------------------------
# Enumeration: which prompts to re-run
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PromptToRerollout:
    task_id: int
    task_name: str
    round_num: int
    prompt_idx: int
    prompt_text: str
    original_successes: int
    original_total: int          # successes + failures
    original_sr: float
    output_dir: Path             # under cfg.rerollout.run_root


def _round_idx_from_path(p: Path) -> int:
    return int(p.name[len("round"):])


def _prompt_idx_from_path(p: Path) -> int:
    return int(p.name[len("prompt"):])


def select_nonzero_prompts(
    prompt_opt_base: Path,
    rerollout_base: Path,
    task_ids: list[int] | None = None,
    min_successes: int = 1,
) -> list[PromptToRerollout]:
    """Walk prompt-opt tree; for each (task, round, prompt) with original
    successes >= min_successes, build a record.
    """
    out: list[PromptToRerollout] = []
    candidate_task_ids = list(task_ids) if task_ids else iter_task_ids(prompt_opt_base)

    for task_id in candidate_task_ids:
        try:
            task_meta = load_task_yaml(task_id, prompt_opt_base)
        except FileNotFoundError:
            print(f"[rerollout] task{task_id}: no task.yaml, skipping.")
            continue
        task_name = str(task_meta.get("task_name", f"task{task_id}"))

        # Cache one prompts.json read per (task, round); load_prompts is cheap
        # but iter_prompt_dirs may yield many entries.
        for round_path in iter_round_dirs(task_id, prompt_opt_base):
            round_num = _round_idx_from_path(round_path)
            try:
                prompts_payload = load_prompts(task_id, round_num, prompt_opt_base)
            except (FileNotFoundError, ValueError) as e:
                print(f"[rerollout] task{task_id}/round{round_num}: prompts.json "
                      f"unreadable ({e}); skipping.")
                continue
            prompts_list = prompts_payload.get("prompts") or []

            for orig_prompt_path in iter_prompt_dirs(task_id, round_num, prompt_opt_base):
                prompt_idx = _prompt_idx_from_path(orig_prompt_path)
                counts = count_recorded_trials(orig_prompt_path)
                if counts["successes"] < min_successes:
                    continue
                if prompt_idx >= len(prompts_list):
                    print(f"[rerollout] task{task_id}/round{round_num}/prompt{prompt_idx}: "
                          f"index >= len(prompts) in prompts.json; skipping.")
                    continue
                entry = prompts_list[prompt_idx]
                prompt_text = str(entry.get("text", "") if isinstance(entry, dict) else entry)
                if not prompt_text:
                    print(f"[rerollout] task{task_id}/round{round_num}/prompt{prompt_idx}: "
                          f"empty prompt text; skipping.")
                    continue

                total = counts["successes"] + counts["failures"]
                sr = counts["successes"] / total if total > 0 else 0.0
                out.append(PromptToRerollout(
                    task_id=task_id,
                    task_name=task_name,
                    round_num=round_num,
                    prompt_idx=prompt_idx,
                    prompt_text=prompt_text,
                    original_successes=counts["successes"],
                    original_total=total,
                    original_sr=sr,
                    output_dir=prompt_dir(task_id, round_num, prompt_idx, rerollout_base),
                ))
    # Sort highest baseline SR first, then by (task, round, prompt) for a
    # stable, reproducible order across runs. Evaluating high-SR prompts
    # first lets you bail early if even the strongest prompts collapse
    # under noise, and keeps the most informative comparisons up front.
    out.sort(key=lambda p: (-p.original_sr, p.task_id, p.round_num, p.prompt_idx))
    return out


def filter_remaining(
    prompts: list[PromptToRerollout],
    trials_per_prompt: int,
) -> tuple[list[PromptToRerollout], list[PromptToRerollout]]:
    """Split prompts into (incomplete, complete) using the rerollout output dir."""
    incomplete: list[PromptToRerollout] = []
    complete: list[PromptToRerollout] = []
    for p in prompts:
        valid = count_recorded_trials(p.output_dir).get("valid", 0)
        if valid >= trials_per_prompt:
            complete.append(p)
        else:
            incomplete.append(p)
    return incomplete, complete


# ---------------------------------------------------------------------------
# Per-prompt trial loop (adapted from deploy.py's main loop body)
# ---------------------------------------------------------------------------

def run_one_prompt_trials(
    cfg: DictConfig,
    env: FR3RealEnv,
    client,
    target: PromptToRerollout,
    trials_per_prompt: int,
    noise_level: float,
) -> dict[str, int]:
    """Run up to ``trials_per_prompt`` trials for one prompt under the
    noise-enabled VLA server. Returns post-run trial counts.

    Trial-loop structure mirrors deploy.py but with two changes:
      * Chunk slicing uses ``chunk[0 : execute_k]`` (no valid_start trim).
      * Per-prompt sidecar tags include the noise_level + baseline SR
        so each recording is self-documenting.
    """
    target.output_dir.mkdir(parents=True, exist_ok=True)

    existing = count_recorded_trials(target.output_dir)
    remaining = max(0, trials_per_prompt - existing["valid"])

    print(f"\n[rerollout] === task{target.task_id}/round{target.round_num}"
          f"/prompt{target.prompt_idx} ===")
    print(f"[rerollout] prompt: {target.prompt_text!r}")
    print(f"[rerollout] baseline: {target.original_successes}/{target.original_total} "
          f"= {target.original_sr:.2f}")
    print(f"[rerollout] target trials={trials_per_prompt}; existing valid="
          f"{existing['valid']} (s={existing['successes']}, f={existing['failures']}, "
          f"r={existing['resets']}); remaining={remaining}")
    if remaining == 0:
        print(f"[rerollout] already complete; skipping.")
        return existing

    env.task = target.prompt_text  # ensure FR3 env knows the current task description

    # Recorder setup (same shape as deploy.py).
    recorder: Optional[UncutVideoRecorder] = None
    recording_cfg = cfg.get("recording", {})
    video_cfg = recording_cfg.get("video", {})
    if video_cfg.get("enabled", False):
        vis_keys = list(video_cfg.get("cameras", []))
        if vis_keys:
            def _get_frame() -> Optional[np.ndarray]:
                frames = env.cameras.get_frames()
                vis_frames = [
                    frames[k].color
                    for k in vis_keys
                    if k in frames and frames[k] is not None
                ]
                if not vis_frames:
                    return None
                return np.hstack(vis_frames) if len(vis_frames) > 1 else vis_frames[0]

            vis0_cfg = cfg.task.perception.get(vis_keys[0], {})
            single_res = vis0_cfg.get("resolution", [640, 480])
            resolution = (single_res[0] * len(vis_keys), single_res[1])
            fps = video_cfg.get("fps", 30)

            # Stamp noise_level + baseline into the sidecar via the
            # inference_config field so each uncut.json is self-documenting.
            inference_config = OmegaConf.to_container(cfg.inference, resolve=True)
            inference_config["noise_level"] = float(noise_level)
            inference_config["baseline_sr"] = float(target.original_sr)
            inference_config["baseline_successes"] = int(target.original_successes)
            inference_config["baseline_total"] = int(target.original_total)

            recorder = UncutVideoRecorder(
                output_dir=str(target.output_dir),
                frame_source=_get_frame,
                resolution=resolution,
                fps=fps,
                checkpoint_path=None,
                agent_config=None,
                inference_config=inference_config,
                task_id=target.task_id,
                task_name=target.task_name,
                round_num=target.round_num,
                prompt_idx=target.prompt_idx,
                prompt_text=target.prompt_text,
                expected_trials=int(trials_per_prompt),
            )
            recorder.start()

    success_count = existing["successes"]
    total_trials = existing["valid"]
    this_run_success = 0
    this_run_total = 0
    max_steps = int(cfg.get("max_steps", 1000))
    execute_k = int(cfg.inference.get("execute_k", 16))
    dt = 1.0 / float(cfg.robot.freq)  # not used for waiting; kept for parity

    aborted_due_to_crash = False
    try:
        for trial in tqdm(range(remaining), desc=f"task{target.task_id}/r{target.round_num}/p{target.prompt_idx}"):
            if recorder is not None:
                recorder.set_state("reset", episode=trial)

            cumulative_idx = total_trials + 1
            print(f"\n[rerollout] Resetting robot for trial {cumulative_idx}/{trials_per_prompt} "
                  f"(this run: {trial + 1}/{remaining})...")
            try:
                obs, _ = env.reset()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[rerollout] env.reset() failed: {e!r}. Aborting prompt; "
                      "re-run to resume.")
                aborted_due_to_crash = True
                break

            sr_so_far = success_count / total_trials if total_trials else 0.0
            print(f"[rerollout]   prompt:       {target.prompt_text!r}")
            print(f"[rerollout]   running SR:   {success_count}/{total_trials} "
                  f"= {sr_so_far:.2f}  (baseline: {target.original_successes}/"
                  f"{target.original_total} = {target.original_sr:.2f})")
            input(
                "[rerollout] Press [Enter] to start the trial "
                "(mid-rollout: s=success, f=failure, r=reset)... "
            )

            if recorder is not None:
                recorder.start_episode(trial)

            steps_completed = 0
            mid_label: Optional[str] = None
            crash_reason: Optional[str] = None

            try:
                with raw_stdin():
                    while steps_completed < max_steps:
                        k = key_pressed()
                        if k in ("s", "f", "r"):
                            mid_label = {"s": "success", "f": "failure", "r": "reset"}[k]
                            print(f"\n[rerollout] '{k}' pressed — ending trial early as "
                                  f"{mid_label.upper()}.")
                            break

                        f0 = time.monotonic()
                        chunk = np.asarray(
                            client.infer(_obs_to_openpi_input(obs, target.prompt_text))["actions"]
                        )
                        elapsed = time.monotonic() - f0

                        # No valid_start trim, no wait-for-dt: matches real_rl.py's
                        # chunk semantics. We always take the leading execute_k
                        # actions of the chunk and execute them sequentially.
                        actions_to_execute = np.asarray(chunk[0:execute_k])

                        interrupted = False
                        for a in actions_to_execute:
                            k2 = key_pressed()
                            if k2 in ("s", "f", "r"):
                                mid_label = {"s": "success", "f": "failure", "r": "reset"}[k2]
                                print(f"\n[rerollout] '{k2}' pressed mid-execute — ending as "
                                      f"{mid_label.upper()}.")
                                interrupted = True
                                break
                            a_clipped = clip_action_to_state(
                                np.asarray(a, dtype=np.float32),
                                np.asarray(obs["state"], dtype=np.float32),
                                cfg.inference.get("action_clip", None),
                            )
                            obs, _, _, _, _ = env.step(np.array(a_clipped))
                            steps_completed += 1
                            if recorder is not None:
                                recorder.set_state(
                                    "episode", episode=trial, step=steps_completed
                                )
                        if interrupted:
                            break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                crash_reason = repr(e)
                print(f"\n[rerollout] Trial {cumulative_idx} crashed mid-rollout after "
                      f"{steps_completed} steps: {crash_reason}")
                print("[rerollout] Logging this trial as FAILURE so the video is preserved.")
                mid_label = "failure"

            if recorder is not None:
                recorder.set_state("pause", episode=trial)

            def _record_label(label: str) -> None:
                nonlocal success_count, total_trials, this_run_success, this_run_total
                if label == "success":
                    success_count += 1
                    total_trials += 1
                    this_run_success += 1
                    this_run_total += 1
                elif label == "failure":
                    total_trials += 1
                    this_run_total += 1
                if recorder is not None:
                    recorder.end_episode(trial, label, steps_completed)

            if mid_label is not None:
                _record_label(mid_label)
            else:
                while True:
                    label = input(
                        "[rerollout] Label rollout (s)uccess / (f)ailure / (r)eset: "
                    ).strip().lower()
                    if label in ("s", "f", "r"):
                        _record_label({"s": "success", "f": "failure", "r": "reset"}[label])
                        break

            sr = success_count / total_trials if total_trials else 0.0
            print(f"[rerollout] SR (this prompt): {sr:.2f} "
                  f"({success_count}/{total_trials} valid; baseline was "
                  f"{target.original_successes}/{target.original_total})")

            if crash_reason:
                aborted_due_to_crash = True
                break
    finally:
        if recorder is not None:
            recorder.close()

    return count_recorded_trials(target.output_dir)


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def write_rerollout_summary(
    prompts: list[PromptToRerollout],
    run_root: Path,
    noise_level: float,
) -> Path:
    """Aggregate per-prompt SR_with_noise from the rerollout dirs and write a CSV."""
    out_path = run_root / "summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id", "task_name", "round_num", "prompt_idx", "prompt_text",
        "noise_level",
        "baseline_successes", "baseline_total", "baseline_sr",
        "noise_successes", "noise_total", "noise_sr", "delta_sr",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for p in prompts:
            counts = count_recorded_trials(p.output_dir)
            n_total = counts["successes"] + counts["failures"]
            n_sr = counts["successes"] / n_total if n_total > 0 else 0.0
            w.writerow({
                "task_id": p.task_id,
                "task_name": p.task_name,
                "round_num": p.round_num,
                "prompt_idx": p.prompt_idx,
                "prompt_text": p.prompt_text,
                "noise_level": noise_level,
                "baseline_successes": p.original_successes,
                "baseline_total": p.original_total,
                "baseline_sr": round(p.original_sr, 4),
                "noise_successes": counts["successes"],
                "noise_total": n_total,
                "noise_sr": round(n_sr, 4),
                "delta_sr": round(n_sr - p.original_sr, 4),
            })
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="config", config_name="rerollout", version_base=None)
def main(cfg: DictConfig) -> None:
    # noise_level is required (??? in YAML); Hydra will raise before reaching
    # here if the user forgot to pass it. Cast for sanity and log loudly so
    # there's no ambiguity about what noise scale the resulting data was
    # collected under.
    noise_level = float(cfg.rerollout.noise_level)
    trials_per_prompt = int(cfg.rerollout.trials_per_prompt)
    prompt_opt_base = Path(str(cfg.rerollout.prompt_opt_base)).expanduser().resolve()
    run_root = Path(str(cfg.rerollout.run_root)).expanduser().resolve()
    task_ids_cfg = cfg.rerollout.get("task_ids", None)
    task_ids = [int(t) for t in task_ids_cfg] if task_ids_cfg else None
    min_successes = int(cfg.rerollout.get("min_successes", 1))

    print(f"[rerollout] noise_level={noise_level}  (must match VLA server's "
          f"openpi_rl.noise_level)")
    print(f"[rerollout] trials_per_prompt={trials_per_prompt}")
    print(f"[rerollout] prompt_opt_base={prompt_opt_base}")
    print(f"[rerollout] run_root={run_root}")
    print(f"[rerollout] task_ids={'all' if task_ids is None else task_ids}")
    print(f"[rerollout] min_successes={min_successes}")

    if not prompt_opt_base.is_dir():
        raise FileNotFoundError(f"prompt_opt_base does not exist: {prompt_opt_base}")
    run_root.mkdir(parents=True, exist_ok=True)

    # Enumerate candidates, then split into incomplete vs already-done.
    all_prompts = select_nonzero_prompts(
        prompt_opt_base=prompt_opt_base,
        rerollout_base=run_root,
        task_ids=task_ids,
        min_successes=min_successes,
    )
    incomplete, complete = filter_remaining(all_prompts, trials_per_prompt)
    print(f"\n[rerollout] {len(all_prompts)} candidate prompts "
          f"({len(complete)} already complete, {len(incomplete)} remaining).")
    if not all_prompts:
        print("[rerollout] No nonzero prompts found. Exiting.")
        return

    print("\n[rerollout] Plan:")
    for p in all_prompts:
        valid = count_recorded_trials(p.output_dir).get("valid", 0)
        status = "DONE" if valid >= trials_per_prompt else f"{valid}/{trials_per_prompt}"
        print(f"  task{p.task_id}/round{p.round_num}/prompt{p.prompt_idx} "
              f"[{status}] baseline={p.original_successes}/{p.original_total}: "
              f"{p.prompt_text!r}")
    if not incomplete:
        print("\n[rerollout] All prompts already complete. Writing summary and exiting.")
        out = write_rerollout_summary(all_prompts, run_root, noise_level)
        print(f"[rerollout] Wrote {out}.")
        return

    # One-time hardware/server setup (persists across all prompts).
    print("\n[rerollout] Initializing FR3 env + connecting to VLA server...")
    env = FR3RealEnv(cfg, task=incomplete[0].prompt_text)
    env.start()
    obs, _ = env.reset()
    try:
        qpos = env.controller.get_qpos()
    except Exception as e:
        raise RuntimeError(
            "Robot health check failed right after env.start(). The OSC server "
            "at cfg.zmq.url is not responding, or the 1 kHz libfranka loop died "
            "(check the server's terminal for 'communication_constraints_violation'). "
            "Re-activate FCI from Desk UI and relaunch real/run_osc_server.sh, "
            f"then retry. Underlying: {e!r}"
        ) from e
    print(f"[rerollout] Robot health check OK (qpos={qpos}).")

    client = connect_vla_client(
        host=str(cfg.inference.server.host),
        port=int(cfg.inference.server.port),
    )

    input(f"[rerollout] Press <<Enter>> to begin "
          f"({len(incomplete)} prompts × ≤{trials_per_prompt} trials each)... ")

    # Drive each incomplete prompt's trials. The pre-prompt gate doubles
    # as a (a) "task change" reminder so the user can physically swap the
    # scene, and (b) skip option — `k <Enter>` skips this prompt entirely
    # without running any trials, useful if you decide a prompt isn't
    # worth burning robot time on.
    skipped: list[PromptToRerollout] = []
    prev_task_id: int | None = None
    for prompt_idx, target in enumerate(incomplete):
        print(f"\n[rerollout] >>> Prompt {prompt_idx + 1}/{len(incomplete)} "
              f"(task{target.task_id}/round{target.round_num}/prompt{target.prompt_idx})")
        print(f"[rerollout]   text:     {target.prompt_text!r}")
        print(f"[rerollout]   baseline: {target.original_successes}/{target.original_total} "
              f"= {target.original_sr:.2f}")
        if prev_task_id is not None and target.task_id != prev_task_id:
            print(f"[rerollout] " + "=" * 60)
            print(f"[rerollout] !! TASK CHANGE: task{prev_task_id} -> "
                  f"task{target.task_id} ({target.task_name})")
            print(f"[rerollout] !! Swap the scene to: {target.task_name!r}")
            print(f"[rerollout] " + "=" * 60)

        choice = input(
            f"[rerollout] Press <<Enter>> to run trials, "
            f"'k' <Enter> to skip this prompt: "
        ).strip().lower()
        if choice == "k":
            print(f"[rerollout] SKIPPED task{target.task_id}/round{target.round_num}"
                  f"/prompt{target.prompt_idx}.")
            skipped.append(target)
            prev_task_id = target.task_id
            continue

        try:
            run_one_prompt_trials(
                cfg=cfg,
                env=env,
                client=client,
                target=target,
                trials_per_prompt=trials_per_prompt,
                noise_level=noise_level,
            )
        except KeyboardInterrupt:
            print("\n[rerollout] KeyboardInterrupt — bailing. Re-run to resume.")
            break
        prev_task_id = target.task_id

    # Always re-aggregate the full candidate list (incomplete + already-done).
    out = write_rerollout_summary(all_prompts, run_root, noise_level)
    print(f"\n[rerollout] Done. Summary: {out}")


if __name__ == "__main__":
    main()
