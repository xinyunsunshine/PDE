# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Embodied runner subclass for the bifranka real-world RL pipeline.

Two responsibilities on top of EmbodiedRunner:

1. **Channel rewiring.** The parent EmbodiedRunner orchestrates rollout
   workers with `rollout.generate(input_channel=rollout_channel,
   output_channel=env_channel)` and reads trajectories from
   `self.actor_channel` (env worker normally pushes there). The realworld
   pipeline has no env worker (data comes from bifranka HDF5 via rsync)
   and the rollout worker pushes trajectories directly. We alias
   `self.env_channel = self.actor_channel` in __init__ so that
   `rollout.generate(..., output_channel=self.env_channel)` actually
   sends to the actor sink. `RealworldRolloutWorker.generate` treats its
   `output_channel` argument as the actor channel (see worker docstring).

2. **Post-save ckpt scp to bifranka.** After the parent runner finishes
   writing the FSDP checkpoint to disk (`global_step_<N>/actor/
   model_state_dict/full_weights.pt`), this subclass launches a
   non-blocking subprocess that ships only the consolidated
   `full_weights.pt` to the bifranka inference server. Atomic rename on
   the remote side so the inference server never reads a partial weights
   file. A heartbeat ssh keeps the multiplex socket warm so we don't
   trigger Duo prompts mid-run.

Activated by setting `runner.runner_class: realworld` in the config; the
entry script `train_realworld_agent.py` reads that field and instantiates
this class instead of EmbodiedRunner.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import threading
import time

from rlinf.runners.embodied_runner import EmbodiedRunner


class RealworldEmbodiedRunner(EmbodiedRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # See class docstring: realworld worker pushes trajectories into its
        # output_channel, but the parent run() loop binds output_channel to
        # env_channel. Aliasing makes those trajectories land in the actor
        # sink without modifying the parent's run() loop.
        self.env_channel = self.actor_channel

        self._heartbeat_started = False
        self._scp_procs: list[subprocess.Popen] = []
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    # ---- public hook overrides --------------------------------------------

    def _save_checkpoint(self):
        super()._save_checkpoint()

        if not self._is_rank_zero():
            return

        ckpt_dir = self._latest_actor_ckpt_dir()
        weights_path = os.path.join(ckpt_dir, "model_state_dict", "full_weights.pt")
        if not os.path.isfile(weights_path):
            self._safe_log(
                f"realworld: full_weights.pt not found at {weights_path}; skip scp"
            )
            return

        dest = self.cfg.runner.get("bifranka_dest", None)
        if not dest:
            self._safe_log("realworld: runner.bifranka_dest unset; skipping scp")
            return

        self._ensure_heartbeat_running()
        self._scp_full_weights_to_bifranka(weights_path, dest)

    # ---- helpers ----------------------------------------------------------

    def _is_rank_zero(self) -> bool:
        return getattr(self, "global_rank", 0) == 0 or getattr(self, "_rank", 0) == 0

    def _current_iteration_idx(self) -> int:
        """Iteration the rollout worker last delivered for training.

        Reads the rollout worker's persisted ``.consumed_iterations`` file
        and returns the highest iteration index recorded there. This is the
        iteration whose rollouts produced the gradients in the current
        actor checkpoint.

        Do NOT use ``max(rollout_dir/iteration*)``: bifranka rsyncs new
        iteration dirs into the local rollout root incrementally — often
        several iterations ahead of what the trainer has consumed — so
        that max can outrun the iteration actually trained on, producing
        a mislabeled ``train_after_itrN_model.pt``.

        Returns -1 if no iterations have been consumed yet.
        """
        from pathlib import Path

        consumed_path = (
            Path(self.cfg.rollout.realworld.rollout_dir) / ".consumed_iterations"
        )
        if not consumed_path.is_file():
            return -1
        candidates = []
        try:
            with open(consumed_path) as f:
                for line in f:
                    name = line.strip()
                    if name.startswith("iteration"):
                        try:
                            candidates.append(int(name[len("iteration") :]))
                        except ValueError:
                            pass
        except Exception:
            return -1
        return max(candidates) if candidates else -1

    def _latest_actor_ckpt_dir(self) -> str:
        return os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
            "actor",
        )

    def _scp_full_weights_to_bifranka(self, src_path: str, dest: str):
        # dest expected as user@host:/abs/path/to/run-root.
        # The ckpt that was trained on iteration N's data lands at:
        #   <dest>/iteration{N:03d}/train_after_itr{N}_model.pt
        if ":" not in dest:
            self._safe_log(
                f"realworld: bifranka_dest='{dest}' missing colon (user@host:/path); skip"
            )
            return
        iter_idx = self._current_iteration_idx()
        if iter_idx < 0:
            self._safe_log(
                "realworld: cannot determine current iteration from local "
                "rollout_dir; skipping ckpt scp"
            )
            return
        host, run_root = dest.split(":", 1)
        run_root = run_root.rstrip("/")
        remote_dir = f"{run_root}/iteration{iter_idx:03d}"
        remote_filename = f"train_after_itr{iter_idx}_model.pt"
        remote_tmp = f"{remote_dir}/{remote_filename}.tmp"
        remote_final = f"{remote_dir}/{remote_filename}"

        cipher = self.cfg.runner.get("scp_cipher", "aes128-gcm@openssh.com")
        scp_cmd = [
            "scp",
            "-c", cipher,
            "-o", "ConnectTimeout=30",
            src_path,
            f"{host}:{remote_tmp}",
        ]
        # No-overwrite finalize:
        #   - mkdir the iteration dir.
        #   - If train_after_itrN_model.pt already exists, leave it alone
        #     and delete the freshly-uploaded .tmp; emit a clear marker so
        #     the user can manually rm the existing file to allow the new
        #     correct write to land.
        #   - Otherwise atomic-rename .tmp → final.
        q_dir = shlex.quote(remote_dir)
        q_tmp = shlex.quote(remote_tmp)
        q_final = shlex.quote(remote_final)
        remote_script = (
            f"mkdir -p {q_dir} && "
            f"if [ -e {q_final} ]; then "
            f"  echo \"NO_OVERWRITE: {remote_final} already exists; \""
            f"  echo \"removing freshly uploaded {remote_tmp}\"; "
            f"  rm -f {q_tmp}; "
            f"else "
            f"  mv {q_tmp} {q_final}; "
            f"  echo \"WROTE: {remote_final}\"; "
            f"fi"
        )
        finalize_cmd = ["ssh", host, remote_script]

        chained = " && ".join(
            [
                " ".join(shlex.quote(a) for a in scp_cmd),
                " ".join(shlex.quote(a) for a in finalize_cmd),
            ]
        )
        log_path = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            "scp_to_bifranka.log",
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        log_fh = open(log_path, "ab", buffering=0)
        log_fh.write(
            f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} step={self.global_step} ===\n"
            f"{chained}\n".encode()
        )
        proc = subprocess.Popen(
            ["bash", "-c", chained],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
        self._scp_procs.append(proc)
        self._scp_procs = [p for p in self._scp_procs if p.poll() is None]
        self._safe_log(
            f"realworld: scp pid={proc.pid} step={self.global_step} → {dest}; "
            f"log → {log_path}"
        )

    # ---- heartbeat to keep the ssh multiplex socket warm ------------------

    def _ensure_heartbeat_running(self):
        if self._heartbeat_started:
            return
        self._heartbeat_started = True

        dest = self.cfg.runner.bifranka_dest
        host = dest.split(":", 1)[0] if ":" in dest else dest
        interval_s = float(self.cfg.runner.get("heartbeat_interval_s", 25 * 60))

        def _loop():
            while not self._heartbeat_stop.wait(interval_s):
                try:
                    subprocess.run(
                        ["ssh", "-o", "ConnectTimeout=10", host, "true"],
                        timeout=30,
                        check=False,
                    )
                except Exception:
                    pass

        self._heartbeat_thread = threading.Thread(
            target=_loop, name="bifranka-heartbeat", daemon=True
        )
        self._heartbeat_thread.start()

    def _safe_log(self, msg: str):
        log = getattr(self, "logger", None)
        if log is not None and hasattr(log, "info"):
            log.info(f"[RealworldEmbodiedRunner] {msg}")
        else:
            print(f"[RealworldEmbodiedRunner] {msg}", flush=True)
