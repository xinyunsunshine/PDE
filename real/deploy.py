"""Deploy a pi0(.5) SFT checkpoint on the FR3 + AgileX rig.

Handles uncut video and statistic metric recording for real eval.


(in separate terminals):
    # 1) OSC robot server (owns the 1 kHz libfranka loop):
    python -m real.server.osc_server

    # 2) VLA inference server:
    python -m real.vla_server \
        inference.weights=/path/to/full_weights.pt \
        inference.norm_stats=/path/to/norm_stats.json

Launch:
    python -m real.deploy task.prompt="pick up the red block"

    # Override server location / port if the VLA server isn't on localhost:8000:
    python -m real.deploy \
        task.prompt="pick up the red block" \
        inference.server.host=192.168.1.42 \
        inference.server.port=8000
"""

from __future__ import annotations

import select
import sys
import termios
import tty
from contextlib import contextmanager
from math import ceil
from typing import Optional
from collections import deque

import hydra
import time
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from pathlib import Path

from real.env import FR3RealEnv
from real.real_rl import clip_action_to_state
from real.real_util import UncutVideoRecorder
from real.round_task import count_recorded_trials, load_prompts, load_task_yaml, prompt_dir
from real.vla_client import connect_vla_client

GLOBAL_CAMERA_NAME = "global"
WRIST_CAMERA_NAME = "wrist"


@contextmanager
def raw_stdin():
    """Put stdin into raw mode so single keypresses are available immediately."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def key_pressed() -> Optional[str]:
    """Return the key if one is waiting on stdin, else None. Non-blocking."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def _obs_to_openpi_input(obs: dict, prompt: str) -> dict:
    """Pack an FR3RealEnv observation into pi0's expected input keys."""
    return {
        "observation/image": np.asarray(obs["images"][GLOBAL_CAMERA_NAME], dtype=np.uint8),
        "observation/wrist_image": np.asarray(obs["images"][WRIST_CAMERA_NAME], dtype=np.uint8),
        "observation/state": np.asarray(obs["state"], dtype=np.float32),
        "prompt": prompt,
    }


def _resolve_pipeline_coords(cfg: DictConfig) -> tuple[Optional[int], Optional[int], Optional[int], Optional[str], str, Optional[str]]:
    """Resolve (task_id, round_num, prompt_idx, task_name, prompt_text, output_dir).

    If task_id/round_num/prompt_idx are all set, look up prompts.json and pull the
    text + task name. CLI `task.prompt=...` overrides whatever's in prompts.json.
    Otherwise, fall back to the bare `task.prompt` string and a manually set
    `recording.output_dir`.
    """
    task_id = cfg.get("task_id", None)
    round_num = cfg.get("round_num", None)
    prompt_idx = cfg.get("prompt_idx", None)

    cli_prompt = str(cfg.task.get("prompt", "") or "")
    cli_output_dir = cfg.get("recording", {}).get("output_dir", None)

    if task_id is None and round_num is None and prompt_idx is None:
        if not cli_prompt:
            raise ValueError(
                "Provide either (task_id, round_num, prompt_idx) so deploy.py can "
                "resolve the prompt from prompts.json, or pass `task.prompt='...'` "
                "directly on the CLI."
            )
        return None, None, None, None, cli_prompt, cli_output_dir or "./recordings"

    if task_id is None or round_num is None or prompt_idx is None:
        raise ValueError(
            "task_id, round_num, and prompt_idx must all be set together (got "
            f"{task_id=}, {round_num=}, {prompt_idx=})."
        )

    task_id = int(task_id)
    round_num = int(round_num)
    prompt_idx = int(prompt_idx)

    task_meta = load_task_yaml(task_id)
    task_name = str(task_meta.get("task_name", ""))

    prompts_payload = load_prompts(task_id, round_num)
    prompts = prompts_payload.get("prompts", [])
    if prompt_idx < 0 or prompt_idx >= len(prompts):
        raise IndexError(
            f"prompt_idx={prompt_idx} out of range for round{round_num}/prompts.json "
            f"with {len(prompts)} prompt(s)."
        )
    resolved = str(prompts[prompt_idx].get("text", ""))
    if not resolved:
        raise ValueError(
            f"prompts.json entry {prompt_idx} has no 'text' field at "
            f"round{round_num}/prompts.json."
        )

    prompt_text = cli_prompt or resolved
    output_dir = str(cli_output_dir) if cli_output_dir else str(prompt_dir(task_id, round_num, prompt_idx))
    return task_id, round_num, prompt_idx, task_name, prompt_text, output_dir


@hydra.main(config_path="config", config_name="deploy", version_base=None)
def main(cfg: DictConfig) -> None:
    task_id, round_num, prompt_idx, task_name, task_prompt, recording_output_dir = (
        _resolve_pipeline_coords(cfg)
    )

    print(f"[deploy] task_id={task_id} task_name={task_name!r} round_num={round_num} "
          f"prompt_idx={prompt_idx}")
    print(f"[deploy] Task prompt: {task_prompt!r}")
    print(f"[deploy] Recording output dir: {recording_output_dir}")

    # Resume support: if the prompt dir already has valid trials from a prior run
    # (server crash, manual interrupt, etc.), top up to cfg.trials instead of
    # starting over. cfg.trials is the target; remaining = max(0, target - existing).
    target_trials = int(cfg.trials)
    existing = count_recorded_trials(Path(recording_output_dir))
    remaining = max(0, target_trials - existing["valid"])

    if existing["valid"] > 0:
        print(
            f"[deploy] Found {existing['valid']} valid trials already recorded at "
            f"{recording_output_dir} (s={existing['successes']}, f={existing['failures']}, "
            f"r={existing['resets']}); target={target_trials}, remaining={remaining}."
        )
    if remaining == 0:
        print("[deploy] Already complete; nothing to do. Exiting.")
        return

    if existing["valid"] > 0:
        input(f"Press <<Enter>> to resume with {remaining} more trial(s)...")
    else:
        input("Press <<Enter>> to continue...")

    env = FR3RealEnv(cfg, task=task_prompt)
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
    print(f"[deploy] Robot health check OK (qpos={qpos}).")

    # Connect to the VLA inference server. Blocks until the server accepts the
    # connection (5 s retry loop inside WebsocketClientPolicy).
    client = connect_vla_client(
        host=str(cfg.inference.server.host),
        port=int(cfg.inference.server.port),
    )

    # Uncut-video recording
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

            recorder = UncutVideoRecorder(
                output_dir=recording_output_dir,
                frame_source=_get_frame,
                resolution=resolution,
                fps=fps,
                checkpoint_path=None,
                agent_config=None,
                inference_config=OmegaConf.to_container(cfg.inference, resolve=True),
                task_id=task_id,
                task_name=task_name,
                round_num=round_num,
                prompt_idx=prompt_idx,
                prompt_text=task_prompt,
                expected_trials=int(cfg.trials),
            )
            recorder.start()
            print(f"[deploy] Video recording: {vis_keys}")

    # `success_count` / `total_trials` are CUMULATIVE (existing + this run) so the
    # printed SR reflects the full prompt dir. `this_run_*` track this invocation.
    success_count = existing["successes"]
    total_trials = existing["valid"]
    this_run_success = 0
    this_run_total = 0
    max_steps = int(cfg.get("max_steps", 300))
    execute_k = int(cfg.inference.get("execute_k", 25))

    # env.step → controller.step → wait_for_next_tick paces each action to dt,
    # so no deadline tracking is needed here. Per trial: infer on current obs,
    # execute up to execute_k actions, re-infer on fresh obs.
    dt = 1.0 / float(cfg.robot.freq)
    aborted_due_to_crash = False
    try:
        for trial in tqdm(range(remaining), desc="trials"):
            if recorder is not None:
                recorder.set_state("reset", episode=trial)

            cumulative_idx = total_trials + 1  # 1-indexed for display
            print(f"\n[deploy] Resetting robot for trial {cumulative_idx}/{target_trials} "
                  f"(this run: {trial + 1}/{remaining})...")
            try:
                obs, _ = env.reset()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"\n[deploy] env.reset() failed before trial {cumulative_idx}: {e!r}")
                print("[deploy] Aborting deploy (no episode recorded for this trial). "
                      "Re-run the same command to resume after fixing the issue.")
                aborted_due_to_crash = True
                break
            input(
                "[deploy] Press [Enter] to start the trial "
                "(mid-rollout: s=success, f=failure, r=reset)... "
            )

            if recorder is not None:
                recorder.start_episode(trial)

            steps_completed = 0
            mid_label: Optional[str] = None  # "success" | "failure" | "reset" if user interrupts
            crash_reason: Optional[str] = None  # set if env.step / client.infer raised mid-rollout

            action_queue: deque = deque()
            try:
                with raw_stdin():
                    while steps_completed < max_steps:
                        k = key_pressed()
                        if k in ("s", "f", "r"):
                            mid_label = {"s": "success", "f": "failure", "r": "reset"}[k]
                            print(f"\n[deploy] '{k}' pressed — ending trial early as {mid_label.upper()}.")
                            action_queue.clear()
                            break

                        if not action_queue:
                            f0 = time.monotonic()
                            chunk = np.asarray(
                                client.infer(_obs_to_openpi_input(obs, task_prompt))["actions"]
                            )
                            elapsed = time.monotonic() - f0
                            print(f"elapsed: {elapsed}")

                            valid_start = max(1, ceil(elapsed / dt))
                            wait_for = valid_start * dt - elapsed
                            if wait_for > 0:
                                ts = time.monotonic()
                                while time.monotonic() - ts < wait_for:
                                    time.sleep(0.0005)

                            action_queue.extend(chunk[valid_start:valid_start + execute_k].tolist())

                        action = np.array(action_queue.popleft())
                        action = clip_action_to_state(
                            action,
                            np.asarray(obs["state"], dtype=np.float32),
                            cfg.inference.get("action_clip", None),
                        )
                        obs, _, _, _, _ = env.step(action)
                        steps_completed += 1
                        if recorder is not None:
                            recorder.set_state(
                                "episode", episode=trial, step=steps_completed
                            )
            except KeyboardInterrupt:
                raise
            except Exception as e:
                crash_reason = repr(e)
                print(f"\n[deploy] Trial {cumulative_idx} crashed mid-rollout after "
                      f"{steps_completed} steps: {crash_reason}")
                print("[deploy] Logging this trial as FAILURE so the video is preserved.")
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
                # "reset" doesn't increment either count.
                if recorder is not None:
                    recorder.end_episode(trial, label, steps_completed)

            if mid_label is not None:
                _record_label(mid_label)
                if crash_reason:
                    print(f"[deploy] Trial {cumulative_idx} labeled FAILURE (crash) after "
                          f"{steps_completed} steps. Reason: {crash_reason}")
                else:
                    print(f"[deploy] Trial {cumulative_idx} labeled {mid_label.upper()} mid-rollout "
                          f"after {steps_completed} steps.")
            else:
                while True:
                    label = input(
                        "[deploy] Label rollout (s)uccess / (f)ailure / (r)eset: "
                    ).strip().lower()
                    if label in ("s", "f", "r"):
                        _record_label({"s": "success", "f": "failure", "r": "reset"}[label])
                        break

            sr = success_count / total_trials if total_trials else 0.0
            print(f"[deploy] SR: {sr:.2f} ({success_count}/{total_trials} valid cumulative; "
                  f"this run: {this_run_total} valid / {trial + 1} attempted)")

            if crash_reason:
                print("[deploy] Aborting deploy. Re-run the same command to resume "
                      "after fixing the underlying issue.")
                aborted_due_to_crash = True
                break
    finally:
        if recorder is not None:
            recorder.close()


if __name__ == "__main__":
    main()
