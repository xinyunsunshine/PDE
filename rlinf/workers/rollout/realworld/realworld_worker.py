"""RealworldRolloutWorker — HDF5 rollout consumer for pi0.5 / OpenPI.

Drop-in replacement for MultiStepRolloutWorker when training on real-robot
rollouts produced by `real/real_rl.py` on bifranka.

Per outer iteration the worker:

1. Polls bifranka via ssh for a new iterationNNN/ that has the
   iteration_complete.json marker (atomic — appears only after every
   rollout in that iteration is committed).
2. rsyncs the iteration dir to local storage.
3. Loads all rollout*.h5 files for that iteration (= num_envs rollouts).
4. For each rollout, runs a no-grad forward pass through
   OpenPi0ForRLActionPrediction.default_forward with
   forward_inputs["action_mask"] reconstructed from valid_start +
   execute_k → masked-sum logprob path → per-chunk-step prev_logprobs.
5. Pads to T_max chunk-steps along T and stacks the num_envs rollouts
   into a single Trajectory of shape [T_max+1, B, ...] (with the
   trailing-position convention the FSDP actor expects).
6. Pushes the Trajectory onto actor_channel.

Termination vs. truncation:
    The HDF5 has explicit /terminations and /truncations bool flags.
    The worker writes Trajectory.dones = terminations only (NOT
    terminations | truncations) — this is the "encode terminations as
    dones" trick that makes the unchanged shared GAE bootstrap V on
    truncated episodes correctly without any GAE patches. See
    realworld/curriculum_runner_diff.md and
    realworld/rollout_dump_schema.md for the full rationale.

interface contract (matches MultiStepRolloutWorker so the runner is
identical):

    async def generate(input_channel, output_channel, actor_channel)
    async def sync_model_from_actor()
    def     set_global_step(global_step)
    async def evaluate(...)         # no-op; eval happens on the robot
"""

from __future__ import annotations

import copy
import gc
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, open_dict

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.models import get_model
from rlinf.scheduler import Channel, Cluster, CollectiveGroupOptions, Worker
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.utils import get_model_weights_id
from rlinf.workers.rollout.realworld.format import (
    is_iteration_complete,
    load_iteration,
    reconstruct_action_mask,
)


class SshColdError(RuntimeError):
    """Raised when ssh to bifranka fails with auth/banner/connection signatures
    indicating the ControlMaster socket has expired and the user must
    re-warm the session interactively."""


class RealworldRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.device = torch.cuda.current_device()
        self.actor_group_name = cfg.actor.group_name

        self.placement = HybridComponentPlacement(cfg, Cluster())
        actor_world_size = self.placement.get_world_size("actor")
        self.actor_weight_src_rank = self._rank % actor_world_size

        rw = cfg.rollout.realworld
        self.local_root: Path = Path(rw.rollout_dir)
        self.episodes_per_iteration: int = int(rw.episodes_per_iteration)
        self.max_episode_steps: int = int(rw.max_episode_steps)
        self.poll_interval_s: float = float(rw.get("poll_interval_s", 60.0))
        self.accept_after_id: str = str(rw.get("accept_model_weights_id_after", ""))

        # Remote rsync source. Format: user@host:/abs/path
        self.remote_dest: str = str(rw.get("remote_dest", ""))
        self.scp_cipher: str = str(rw.get("scp_cipher", "aes128-gcm@openssh.com"))

        # Action-mask reconstruction parameters. action_env_dim = 10 for
        # FR3 + AgileX gripper (matches upstream /actions third dim).
        self.chunk_size: int = int(cfg.actor.model.openpi.get("action_horizon", 50))
        self.action_dim_full: int = int(cfg.actor.model.openpi.get("action_dim", 32))
        self.action_env_dim: int = int(cfg.actor.model.openpi.get("action_env_dim", 10))

        self.model_weights_id = ""
        self.count_update = 0
        self.global_step = 0

        # Track which remote iteration names we've already pulled+processed.
        # Persisted to disk so a fresh process resuming from a checkpoint
        # doesn't re-train on already-consumed iterations.
        self._consumed_state_path = self.local_root / ".consumed_iterations"
        self._consumed_iterations: set[str] = self._load_consumed_state()

        max_ctas = cfg.rollout.get("sync_weight_nccl_max_ctas", None)
        min_ctas = cfg.rollout.get("sync_weight_nccl_min_ctas", None)
        self._sync_weight_comm_options = CollectiveGroupOptions(
            accel_max_ctas=max_ctas, accel_min_ctas=min_ctas
        )

    # ---- lifecycle (called by runner) -------------------------------------

    def init_worker(self):
        rollout_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.model_path = self.cfg.rollout.model.model_path

        self.hf_model = get_model(rollout_model_config)

        if self.cfg.runner.get("ckpt_path", None):
            state = torch.load(self.cfg.runner.ckpt_path, map_location="cpu")
            self.hf_model.load_state_dict(state)

        self.hf_model.eval()
        self.hf_model = self.hf_model.to(self.device)

        self.local_root.mkdir(parents=True, exist_ok=True)

        # Per-(task, prompt) success stats accumulated during generate() and
        # drained by get_and_clear_epoch_stats() at the end of each iteration.
        # Schema: list[{epoch_idx, task_desc, prompt, n_trials, n_success}].
        self._epoch_stats: list[dict] = []

        # Heartbeat disabled per user request — was hammering the CSAIL
        # jumphost and triggering fail2ban. Re-enable only if we have a
        # plan to share a single ControlMaster across all ssh callers
        # AND keep the connection-rate well below the jumphost's
        # rate-limit threshold.
        self._heartbeat_thread = None
        self._heartbeat_stop = None

    def _start_ssh_heartbeat(self) -> None:
        import threading

        host = self.remote_dest.split(":", 1)[0]
        interval_s = float(
            self.cfg.rollout.realworld.get("heartbeat_interval_s", 25 * 60)
        )
        self._heartbeat_stop = threading.Event()

        def _loop():
            while not self._heartbeat_stop.wait(interval_s):
                try:
                    subprocess.run(
                        ["ssh", "-o", "ConnectTimeout=10", host, "true"],
                        timeout=30,
                        check=False,
                    )
                except Exception:
                    pass  # transient errors are fine; the next real call will surface them

        self._heartbeat_thread = threading.Thread(
            target=_loop, name="bifranka-heartbeat-rollout", daemon=True
        )
        self._heartbeat_thread.start()
        self.log_info(
            f"realworld: rollout rank-0 ssh heartbeat to {host} every {interval_s:.0f}s"
        )

    def set_global_step(self, global_step):
        self.global_step = int(global_step)
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)

    async def sync_model_from_actor(self):
        param_state_dict = await self.recv(
            self.actor_group_name,
            src_rank=self.actor_weight_src_rank,
            async_op=True,
            options=self._sync_weight_comm_options,
        ).async_wait()

        self.hf_model.load_state_dict(param_state_dict)
        self.model_weights_id = (
            str(get_model_weights_id(self.hf_model)) + f"_{self.count_update}"
        )
        self.count_update += 1
        del param_state_dict
        gc.collect()
        torch.cuda.empty_cache()

    # ---- main generate loop -----------------------------------------------

    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
    ):
        """Wait for one bifranka iteration, recompute logprobs, push to actor.

        Channel contract for the current vla-rl runner: rollout.generate
        is called with 2 channels (input_channel, output_channel). The
        realworld pipeline doesn't run an env loop, so `input_channel`
        carries nothing and is unused. We treat `output_channel` as the
        actor sink — RealworldEmbodiedRunner passes its `actor_channel`
        as the `output_channel` argument so trajectories land where
        `actor.recv_rollout_trajectories` reads.

        With rollout placement=all, this method runs on every rank
        (typically world_size=8). To avoid 8× duplicate work and a
        batch-size mismatch at the actor, the rollouts are sharded:

          - rank 0 polls bifranka and rsyncs the iteration dir to local;
            other ranks wait for the local iteration_complete.json to
            appear (rsync delivers it as part of the iteration dir).
          - every rank loads the full iteration's rollout list, then
            handles only rollouts[rank::world_size] (e.g., 4 of 32 per
            rank for world_size=8). Each rank pushes a Trajectory of
            shape [T_max, 32/world_size, ...].
          - the actor's recv_rollout_trajectories receives N=world_size
            trajectories and convert_trajectories_to_batch concatenates
            them along dim=1 → batch of [T_max, 32, ...].
        """
        del input_channel  # unused in realworld
        actor_channel = output_channel  # output_channel IS the actor sink

        iter_dir = self._wait_for_next_iteration()
        all_rollouts = load_iteration(iter_dir)
        n_total = min(len(all_rollouts), self.episodes_per_iteration)
        all_rollouts = all_rollouts[:n_total]

        # Sharded slice: rollouts[rank::world_size]
        rs = self._rank
        ws = max(self.placement.get_world_size("rollout"), 1)
        my_rollouts = all_rollouts[rs::ws]
        self.log_info(
            f"realworld: rank={rs}/{ws} loaded iteration {iter_dir.name} "
            f"({len(my_rollouts)} of {n_total} rollouts)"
        )

        # Read iteration_meta.json to get per-env prompt assignments. Bifranka's
        # inference server records `prompts` (the candidate phrasings) and
        # `prompt_indices` (length=num_envs, per-env which prompt was assigned).
        # We use this to build per-prompt success stats from the rollout rewards
        # so the curriculum runner can update its EMA UCB without needing a live
        # eval env.
        from rlinf.workers.rollout.realworld.format import load_iteration_meta
        iter_meta = load_iteration_meta(iter_dir)
        # Bifranka has used two schemas:
        #   (old) prompts: [unique_strings], prompt_indices: [int per env]
        #   (new) per_env_prompts: [string per env]
        # Normalize to (prompts_list, prompt_indices) where prompts_list is the
        # unique prompt vocabulary and prompt_indices maps env -> prompts_list.
        if "per_env_prompts" in iter_meta:
            per_env = list(iter_meta["per_env_prompts"])
            seen: dict[str, int] = {}
            prompts_list: list[str] = []
            prompt_indices: list[int] = []
            for s in per_env:
                if s not in seen:
                    seen[s] = len(prompts_list)
                    prompts_list.append(s)
                prompt_indices.append(seen[s])
        else:
            prompts_list = list(iter_meta.get("prompts", []))
            prompt_indices = list(iter_meta.get("prompt_indices", []))

        # Console-log this rank's per-env prompt assignment so users can see
        # which prompts each GPU is training on for the current iteration.
        # Aggregated per-prompt success goes to wandb via the runner's
        # _log_per_prompt_success / _log_ucb_pool_stats path.
        my_global_envs = list(range(rs, n_total, ws))
        rank_assignments: list[tuple[int, int, str]] = []
        for global_idx in my_global_envs:
            if 0 <= global_idx < len(prompt_indices):
                pidx = int(prompt_indices[global_idx])
                prompt_str = (
                    prompts_list[pidx]
                    if 0 <= pidx < len(prompts_list)
                    else f"<unknown {pidx}>"
                )
                rank_assignments.append((global_idx, pidx, prompt_str))
        if rank_assignments:
            lines = [
                f"realworld: rank={rs}/{ws} iteration={iter_dir.name} "
                f"prompt assignments ({len(rank_assignments)} envs):"
            ]
            for global_idx, pidx, prompt_str in rank_assignments:
                lines.append(f"   env={global_idx:02d} prompt_idx={pidx}: {prompt_str!r}")
            self.log_info("\n".join(lines))

        episodes = []
        per_prompt_counts: dict[int, dict[str, int]] = {}
        for i, rollout in enumerate(my_rollouts):
            ep = self._rollout_to_episode(rollout)
            cache_path = self._logprob_cache_path(rollout)
            cache_hit = self._maybe_load_logprob_cache(ep, cache_path)
            if not cache_hit:
                ep = self._recompute_logprobs(ep)
                self._save_logprob_cache(ep, cache_path)
            episodes.append(ep)
            self.log_info(
                f"realworld: rank={rs} episode {i + 1}/{len(my_rollouts)} "
                f"{'cache' if cache_hit else 'recompute'} done (T={len(ep['steps'])})"
            )

            # Per-prompt success aggregation. Global rollout index in the
            # iteration is rs + i*ws (since my_rollouts = all_rollouts[rs::ws]).
            global_rollout_idx = rs + i * ws
            if global_rollout_idx < len(prompt_indices):
                pidx = int(prompt_indices[global_rollout_idx])
                # Success := the rollout received any positive reward at any
                # chunk-step. For our binary reward, this matches the
                # "did the task complete?" definition used elsewhere.
                rewards = rollout.get("rewards")
                if rewards is not None:
                    success = bool(np.asarray(rewards).max() > 0)
                else:
                    success = False
                slot = per_prompt_counts.setdefault(
                    pidx, {"n_trials": 0, "n_success": 0}
                )
                slot["n_trials"] += 1
                slot["n_success"] += int(success)

        # Surface per-(task, prompt) stats for the runner's UCB EMA. Each
        # entry mirrors the schema PromptOptRolloutWorker uses
        # (epoch_idx, task_desc, prompt, n_success, n_trials) so the
        # downstream parent runner aggregates across ranks identically.
        # For realworld single-task we tag every entry with the same
        # task_desc — the curriculum tracks evolution within the one
        # task's prompt pool.
        task_desc = self._curriculum_task_desc()
        for pidx, counts in per_prompt_counts.items():
            prompt = (
                prompts_list[pidx]
                if 0 <= pidx < len(prompts_list)
                else f"<unknown prompt {pidx}>"
            )
            self._epoch_stats.append({
                "epoch_idx": int(self.global_step),
                "task_desc": task_desc,
                "prompt":    prompt,
                "n_trials":  counts["n_trials"],
                "n_success": counts["n_success"],
            })

        trajectory = self._pad_and_pack(episodes)
        actor_channel.put(trajectory, async_op=True)

    async def evaluate(
        self, input_channel: Channel, output_channel: Channel, *args, **kwargs
    ):
        del input_channel, output_channel
        return

    def update_prompt_epoch_map(self, prompt_epoch_map):
        """No-op for realworld: prompts are managed by the curriculum runner
        and shipped to bifranka, not consumed inside this worker."""
        del prompt_epoch_map
        return None

    def get_and_clear_epoch_stats(self) -> list:
        """Return per-(task, prompt) success counts collected during the
        most recent generate() call, then clear the buffer. Each entry:
            {epoch_idx, task_desc, prompt, n_trials, n_success}
        The runner aggregates across worker ranks and feeds it into the
        EMA UCB tracker, which then produces the next round's prompt
        assignments via _write_next_round_prompts_json.

        Without this, the curriculum-update step skips because
        epoch_stats_per_worker comes back empty.
        """
        stats = list(self._epoch_stats)
        self._epoch_stats.clear()
        return stats

    # ---- persistent _consumed_iterations -----------------------------------

    def _load_consumed_state(self) -> set:
        """Read the on-disk record of consumed iteration names. The file is
        a plain text list, one iteration name per line. Returns an empty
        set if the file doesn't exist (first launch).
        """
        try:
            if self._consumed_state_path.is_file():
                with open(self._consumed_state_path) as f:
                    return {line.strip() for line in f if line.strip()}
        except Exception:
            pass
        return set()

    def _persist_consumed_state(self) -> None:
        """Atomically write the current _consumed_iterations to disk."""
        try:
            self._consumed_state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._consumed_state_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                for name in sorted(self._consumed_iterations):
                    f.write(name + "\n")
            os.replace(tmp, self._consumed_state_path)
        except Exception as e:
            self.log_warning(f"realworld: failed to persist consumed state: {e}")

    def _curriculum_task_desc(self) -> str:
        """Return the canonical task descriptor used as the EMA-tracking key.

        Realworld is single-task in our setup (all 6 prompts in
        iteration_meta.json are reformulations of the same goal). We
        prefer cfg.curriculum.task_descriptions[0] if set; otherwise we
        fall back to the most-frequent (or first) prompt as a stable key.
        """
        try:
            tds = self.cfg.runner.prompt_opt.task_descriptions  # ListConfig
            if tds and len(tds) > 0:
                return str(tds[0])
        except Exception:
            pass
        return "realworld_single_task"

    # ---- bifranka polling + rsync -----------------------------------------

    def _wait_for_next_iteration(self) -> Path:
        """Poll bifranka every poll_interval_s seconds for a new iteration.

        Rank-aware: only rank 0 does the actual ssh poll + rsync. Other
        ranks wait until the local iteration_complete.json appears (the
        rsync from rank 0 delivers it as part of the iteration dir).
        Avoids 8× redundant ssh and rsync ops, and avoids race conditions
        on the local filesystem.

        Returns the local Path to the synced iteration dir.
        """
        if not self.remote_dest:
            # No remote configured — assume rollouts already on local fs.
            return self._wait_for_next_iteration_local_only()

        if self._rank != 0:
            # Non-leader ranks: just poll local fs for the marker file.
            return self._wait_for_next_iteration_local_only()

        # Rank 0 does the bifranka poll + rsync.
        #
        # Strategy: overlap rsync with bifranka's writes. As soon as
        # bifranka starts an iteration (drops iteration_meta.json), we
        # incrementally rsync whatever rollouts are present so the data
        # transfer doesn't have to happen all-at-once after
        # iteration_complete.json appears. We only RETURN (and start the
        # actual training compute) when iteration_complete.json AND all
        # expected rollout*.h5 files are present locally.
        #
        # rsync is incremental — files already in local with matching
        # mtime/size are skipped, so this is cheap to run every poll.
        poll_start = time.monotonic()
        attempt = 0
        while True:
            attempt += 1
            try:
                # All iterations bifranka has started (have iteration_meta).
                started_iters = self._list_remote_started_iterations()
                # Incremental rsync for every started, not-yet-consumed iter.
                # Don't enforce verify here — bifranka may still be writing.
                for iter_name in started_iters:
                    if iter_name in self._consumed_iterations:
                        continue
                    self._rsync_iteration(iter_name, verify=False)

                # Now check if any iter is finalized and complete locally.
                for iter_name in started_iters:
                    if iter_name in self._consumed_iterations:
                        continue
                    local_iter = self.local_root / iter_name
                    if not is_iteration_complete(local_iter):
                        # bifranka still writing rollouts; keep polling.
                        continue
                    ok, expected, found = self._verify_iteration_files(local_iter)
                    if not ok:
                        # marker present but files short — one more strict
                        # rsync pass to fill in any stragglers.
                        self.log_info(
                            f"realworld: {iter_name} complete-marker present but "
                            f"only {found}/{expected} rollout*.h5 — final rsync"
                        )
                        self._rsync_iteration(iter_name, verify=True)
                        ok, expected, found = self._verify_iteration_files(local_iter)
                    if ok:
                        self._consumed_iterations.add(iter_name)
                        self._persist_consumed_state()
                        return local_iter

                pending = [
                    n for n in started_iters if n not in self._consumed_iterations
                ]
                elapsed = int(time.monotonic() - poll_start)
                self.log_info(
                    f"realworld: [poll #{attempt}, waited {elapsed}s] "
                    f"polling bifranka for rollout data — "
                    f"started={len(started_iters)} pending={pending or '[]'}; "
                    f"sleeping {self.poll_interval_s}s"
                )
            except SshColdError as e:
                # Distinctive marker the launcher Monitor greps for. The
                # rollout worker can't surface a desktop notification by
                # itself, so we make the log line shout. The user re-warms
                # ssh from the cluster session and the next poll succeeds.
                self.log_warning(
                    "REWARM_SSH_REQUIRED: ssh to bifranka cold — please run "
                    "'! ssh user@robot.example.com echo ok' from your "
                    f"cluster session. Will retry every {self.poll_interval_s}s. "
                    f"stderr_tail={e}"
                )
            except subprocess.TimeoutExpired:
                self.log_warning(
                    "REWARM_SSH_REQUIRED: ssh to bifranka timed out — please run "
                    "'! ssh user@robot.example.com echo ok' from your "
                    f"cluster session. Will retry every {self.poll_interval_s}s."
                )
            except Exception as e:
                self.log_warning(f"realworld: poll error '{e}'; retry in {self.poll_interval_s}s")
            time.sleep(self.poll_interval_s)

    def _wait_for_next_iteration_local_only(self) -> Path:
        """Fallback: poll local_root only (no bifranka rsync)."""
        poll_start = time.monotonic()
        attempt = 0
        while True:
            attempt += 1
            for iter_dir in sorted(self._iter_dirs(self.local_root)):
                if iter_dir.name in self._consumed_iterations:
                    continue
                if is_iteration_complete(iter_dir):
                    self._consumed_iterations.add(iter_dir.name)
                    self._persist_consumed_state()
                    return iter_dir
            elapsed = int(time.monotonic() - poll_start)
            self.log_info(
                f"realworld: [poll #{attempt}, waited {elapsed}s] "
                f"polling local fs for rollout data "
                f"({self.local_root}); sleeping {self.poll_interval_s}s"
            )
            time.sleep(self.poll_interval_s)

    @staticmethod
    def _iter_dirs(root: Path) -> list[Path]:
        if not root.is_dir():
            return []
        out: list[Path] = []
        for entry in root.iterdir():
            if entry.is_dir() and entry.name.startswith("iteration"):
                out.append(entry)
        return out

    def _list_remote_started_iterations(self) -> list[str]:
        """Return iteration_name list for which iteration_meta.json exists on bifranka.

        Includes both in-progress (meta but no complete) and finalized
        (meta + complete) iterations. Used to drive incremental rsyncs
        as bifranka writes rollouts, before iteration_complete.json
        appears.
        """
        host, remote_root = self._split_remote_dest()
        cmd = (
            f"find {self._shq(remote_root)} -maxdepth 2 -name iteration_meta.json "
            f"2>/dev/null"
        )
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=15", host, cmd],
            capture_output=True, text=True, timeout=60, check=False,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            cold_signatures = (
                "Permission denied",
                "Connection timed out during banner",
                "kex_exchange_identification",
                "Connection closed by",
                "Could not resolve hostname",
                "Host key verification failed",
                "ssh: connect to host",
            )
            if any(sig in stderr for sig in cold_signatures):
                raise SshColdError(stderr[-400:] or f"rc={result.returncode}")
            raise RuntimeError(
                f"ssh to {host} returned rc={result.returncode}; "
                f"stderr={stderr[-400:]}"
            )
        names: set[str] = set()
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            names.add(Path(line).parent.name)
        return sorted(names)

    def _list_remote_complete_iterations(self) -> list[str]:
        """Return sorted iteration_name list for which iteration_complete.json exists on bifranka.

        Raises SshColdError if ssh's stderr indicates the multiplex socket
        / auth state has died. The outer poll loop catches that and emits
        a distinctive REWARM marker so the user knows what to do.
        """
        host, remote_root = self._split_remote_dest()
        cmd = (
            f"find {self._shq(remote_root)} -maxdepth 2 -name iteration_complete.json "
            f"2>/dev/null"
        )
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=15", host, cmd],
            capture_output=True, text=True, timeout=60, check=False,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            cold_signatures = (
                "Permission denied",
                "Connection timed out during banner",
                "kex_exchange_identification",
                "Connection closed by",
                "Could not resolve hostname",
                "Host key verification failed",
                "ssh: connect to host",
            )
            if any(sig in stderr for sig in cold_signatures):
                raise SshColdError(stderr[-400:] or f"rc={result.returncode}")
            # Some other ssh failure — surface it but don't claim ssh is cold.
            raise RuntimeError(
                f"ssh to {host} returned rc={result.returncode}; "
                f"stderr={stderr[-400:]}"
            )
        names: set[str] = set()
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            names.add(Path(line).parent.name)
        return sorted(names)

    def _run_rsync_streaming(
        self, cmd: list, iter_name: str, timeout: int = 1800
    ) -> "subprocess.CompletedProcess":
        """Run rsync, streaming throttled progress snapshots to log_info.

        Expects ``--info=progress2`` in cmd so rsync emits a single carriage-
        return-updated aggregate line (bytes / pct / rate / eta). Snapshots
        are throttled to one log line every PROGRESS_THROTTLE_S. Final
        summary lines (from ``--info=stats2``) and any per-file or error
        lines pass through unthrottled.
        """
        PROGRESS_THROTTLE_S = 5.0

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        stdout_chunks: list[bytes] = []
        buf = b""
        last_progress_emit = 0.0

        def _emit(line: str, is_progress: bool) -> None:
            nonlocal last_progress_emit
            line = line.strip()
            if not line:
                return
            if line.startswith(("sending ", "receiving ")):
                return
            if is_progress:
                now = time.monotonic()
                if now - last_progress_emit < PROGRESS_THROTTLE_S:
                    return
                last_progress_emit = now
                self.log_info(f"realworld: rsync {iter_name} [progress] {line}")
            else:
                self.log_info(f"realworld: rsync {iter_name} > {line}")

        try:
            assert proc.stdout is not None
            fd = proc.stdout.fileno()
            while True:
                chunk = os.read(fd, 4096)
                if not chunk:
                    break
                stdout_chunks.append(chunk)
                buf += chunk
                # Split on \r or \n. \r-terminated chunks are rsync's in-place
                # progress refreshes; \n-terminated are final lines.
                while True:
                    cr = buf.find(b"\r")
                    nl = buf.find(b"\n")
                    if cr == -1 and nl == -1:
                        break
                    if cr == -1:
                        idx, is_progress = nl, False
                    elif nl == -1:
                        idx, is_progress = cr, True
                    else:
                        idx = min(cr, nl)
                        is_progress = idx == cr and idx != nl
                    line = buf[:idx].decode("utf-8", errors="replace")
                    buf = buf[idx + 1 :]
                    _emit(line, is_progress)
            proc.wait(timeout=timeout)
            if buf:
                _emit(buf.decode("utf-8", errors="replace"), False)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
            raise
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        stdout_text = b"".join(stdout_chunks).decode("utf-8", errors="replace")
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout_text, stderr)

    def _rsync_iteration(self, iter_name: str, verify: bool = True) -> None:
        """rsync one iteration dir from bifranka.

        verify=True (the strict path used right before training):
            After rsync, require that local has `num_envs` rollout*.h5
            files (per iteration_meta.json). Retry up to
            `rsync_max_attempts` times. Raise on non-convergence so we
            never train on partial data.

        verify=False (the incremental path used while bifranka is still
            writing the iteration): one rsync attempt; partial data is
            expected and fine — the next poll will rsync again. Useful
            to overlap data transfer with bifranka's rollout production.
        """
        host, remote_root = self._split_remote_dest()
        local_iter = self.local_root / iter_name
        local_iter.mkdir(parents=True, exist_ok=True)
        cmd = [
            "rsync",
            "-a",
            # progress2: aggregate bytes/pct/rate/eta on one \r-updated line;
            # stats2: final summary at end; flist0/name0: drop noisy file list.
            "--info=progress2,stats2,flist0,name0",
            "-e", f"ssh -c {self.scp_cipher} -o ConnectTimeout=30",
            f"{host}:{remote_root}/{iter_name}/",
            str(local_iter) + "/",
        ]

        # Incremental (non-verifying) call: one rsync pass, partial OK.
        # The polling loop will call us again next tick to fetch any
        # rollouts bifranka has written since.
        if not verify:
            try:
                result = self._run_rsync_streaming(cmd, iter_name, timeout=1800)
            except subprocess.TimeoutExpired:
                self.log_warning(
                    f"realworld: incremental rsync {iter_name} timed out; "
                    "will retry next poll"
                )
                return
            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                # Bubble up ssh-cold so the polling loop emits the REWARM
                # marker; everything else is just noise to log + ignore.
                cold_signatures = (
                    "Permission denied",
                    "Connection timed out during banner",
                    "kex_exchange_identification",
                    "Connection closed by",
                )
                if any(sig in stderr for sig in cold_signatures):
                    raise SshColdError(stderr[-400:] or f"rc={result.returncode}")
                self.log_warning(
                    f"realworld: incremental rsync {iter_name} rc={result.returncode}; "
                    f"stderr_tail={stderr[-300:]}"
                )
                return
            # Logging the partial state can be useful to see progress
            # toward all-files-present without flooding the log.
            found = sum(1 for _ in local_iter.glob("rollout*.h5"))
            self.log_info(
                f"realworld: incremental rsync {iter_name} ← bifranka "
                f"(now have {found} rollout*.h5)"
            )
            return

        max_attempts = int(self.cfg.rollout.realworld.get("rsync_max_attempts", 4))
        retry_sleep_s = float(
            self.cfg.rollout.realworld.get("rsync_retry_sleep_s", 5.0)
        )

        for attempt in range(1, max_attempts + 1):
            self.log_info(
                f"realworld: rsync {iter_name} ← bifranka (attempt {attempt}/{max_attempts})"
            )
            try:
                result = self._run_rsync_streaming(cmd, iter_name, timeout=1800)
            except subprocess.TimeoutExpired:
                self.log_warning(f"realworld: rsync {iter_name} timed out (attempt {attempt})")
                if attempt == max_attempts:
                    raise
                time.sleep(retry_sleep_s)
                continue

            if result.returncode != 0:
                self.log_warning(
                    f"realworld: rsync {iter_name} non-zero rc={result.returncode}; "
                    f"stderr_tail={result.stderr[-500:] if result.stderr else ''}"
                )
                if attempt == max_attempts:
                    raise subprocess.CalledProcessError(
                        result.returncode, cmd,
                        output=result.stdout, stderr=result.stderr,
                    )
                time.sleep(retry_sleep_s)
                continue

            ok, expected, found = self._verify_iteration_files(local_iter)
            if ok:
                self.log_info(
                    f"realworld: rsync {iter_name} OK ({found}/{expected} files)"
                )
                return
            self.log_warning(
                f"realworld: rsync {iter_name} INCOMPLETE — got {found}/{expected} "
                f"rollout*.h5 (attempt {attempt}/{max_attempts}); will retry"
            )
            time.sleep(retry_sleep_s)

        # Out of attempts — fail loudly so we don't train on partial data.
        ok, expected, found = self._verify_iteration_files(local_iter)
        raise RuntimeError(
            f"realworld: rsync {iter_name} did not converge after "
            f"{max_attempts} attempts: have {found}/{expected} rollout*.h5 "
            f"in {local_iter}. Refusing to start training on partial data."
        )

    def _verify_iteration_files(
        self, local_iter: "Path"
    ) -> tuple[bool, int, int]:
        """Return (is_complete, expected_count, found_count)."""
        from rlinf.workers.rollout.realworld.format import load_iteration_meta

        meta = load_iteration_meta(local_iter)
        expected = int(meta.get("num_envs", 0)) or int(self.episodes_per_iteration)
        found = sum(1 for _ in local_iter.glob("rollout*.h5"))
        return (found >= expected), expected, found

    def _split_remote_dest(self) -> tuple[str, str]:
        if ":" not in self.remote_dest:
            raise ValueError(
                f"rollout.realworld.remote_dest='{self.remote_dest}' missing ':' "
                "(expected user@host:/abs/path)"
            )
        host, root = self.remote_dest.split(":", 1)
        return host, root.rstrip("/")

    @staticmethod
    def _shq(s: str) -> str:
        # cheap shell-quoting for paths going into ssh remote command strings
        return "'" + s.replace("'", "'\"'\"'") + "'"

    # ---- HDF5 rollout → per-step episode dict ------------------------------

    def _rollout_to_episode(self, rollout: dict[str, Any]) -> dict[str, Any]:
        """Wrap an HDF5 rollout into the per-step structure _pad_and_pack expects.

        Each step gets:
          - forward_inputs: every key from /forward_inputs (no action_mask —
            we use the sim fixed-slice path so the model returns logprobs
            of shape (B, action_chunk, action_env_dim), matching sim).
            Currently safe because all executed chunks in our data have
            valid_start=0 and execute_k=16; the static slice
            [0:action_chunk, :action_env_dim] equals the executed window.
            Re-enable the masked-sum branch (pass action_mask) only if
            future data has variable per-step windows.
          - rewards, terminations, truncations: scalar wrapped to [1, 1]
        """
        T = rollout["horizon"]

        steps = []
        for t in range(T):
            fwd = {}
            for k, arr in rollout["forward_inputs"].items():
                # Add a leading batch dim of 1 to match what the rest of the
                # worker expects (shape_fwd uses arr.shape[1:] in _pad_and_pack).
                fwd[k] = torch.from_numpy(np.asarray(arr[t : t + 1]))

            steps.append({
                "forward_inputs": fwd,
                "rewards":      torch.tensor([[float(rollout["rewards"][t])]],
                                              dtype=torch.float32),
                "terminations": torch.tensor([[bool(rollout["terminations"][t])]],
                                              dtype=torch.bool),
                "truncations":  torch.tensor([[bool(rollout["truncations"][t])]],
                                              dtype=torch.bool),
            })

        return {
            "horizon": T,
            "steps":   steps,
            "model_weights_id": rollout.get("model_weights_id", ""),
        }

    # ---- streaming logprob recompute --------------------------------------

    @torch.no_grad()
    def _recompute_logprobs(self, ep: dict[str, Any]) -> dict[str, Any]:
        for step in ep["steps"]:
            forward_inputs = self._to_device(step["forward_inputs"])
            out = self.hf_model.default_forward(
                forward_inputs=forward_inputs,
                compute_values=True,
            )
            step["prev_logprobs"] = out["logprobs"].detach().cpu().contiguous()
            # The value head returns a scalar per batch (shape [B]); the GAE
            # preprocessor expects a chunk_size dim at the end, matching the
            # 3D shape of rewards/dones ([T+1, B, 1]). Add the trailing
            # singleton so per-step shape becomes [1, 1] and after pack
            # prev_values is [T+1, B, 1] just like rewards/terminations.
            values = out["values"].detach().cpu().contiguous()
            if values.dim() == 1:
                values = values.unsqueeze(-1)  # [B] -> [B, 1]
            step["prev_values"] = values
        return ep

    def _to_device(self, d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in d.items()}

    # ---- logprob/value cache (skip recompute on restart) ------------------

    def _logprob_cache_path(self, rollout: dict[str, Any]) -> Path | None:
        """Return path of the cache file for this rollout, or None.

        Caches are scoped by `global_step` so that after a PPO update the
        recompute is forced to use the new weights. The rollout dict
        carries `__source_path` set by load_iteration; if absent, caching
        is disabled.
        """
        src = rollout.get("__source_path")
        if not src:
            return None
        src = Path(src)
        # rollout005.h5 -> rollout005.h5.logprobs_step{N}.pt
        return src.with_name(src.name + f".logprobs_step{self.global_step}.pt")

    def _maybe_load_logprob_cache(
        self, ep: dict[str, Any], cache_path: Path | None
    ) -> bool:
        """Populate ep['steps'][i]['prev_logprobs'/'prev_values'] from cache.

        Returns True on a clean cache hit (caller skips recompute), False
        otherwise. A cache miss includes: no file, version mismatch, step
        count mismatch, or a torch.load exception.
        """
        if cache_path is None or not cache_path.is_file():
            return False
        try:
            cached = torch.load(cache_path, map_location="cpu")
            steps = ep["steps"]
            if (
                cached.get("schema") != self._LOGPROB_CACHE_SCHEMA
                or len(cached.get("logprobs", [])) != len(steps)
                or len(cached.get("values", [])) != len(steps)
            ):
                self.log_warning(
                    f"realworld: logprob cache schema/length mismatch at "
                    f"{cache_path.name}; recomputing"
                )
                return False
            for i, step in enumerate(steps):
                step["prev_logprobs"] = cached["logprobs"][i]
                step["prev_values"] = cached["values"][i]
            return True
        except Exception as e:
            self.log_warning(
                f"realworld: failed to load logprob cache {cache_path.name} "
                f"({e}); recomputing"
            )
            return False

    def _save_logprob_cache(
        self, ep: dict[str, Any], cache_path: Path | None
    ) -> None:
        if cache_path is None:
            return
        try:
            torch.save(
                {
                    "schema": self._LOGPROB_CACHE_SCHEMA,
                    "global_step": self.global_step,
                    "logprobs": [s["prev_logprobs"] for s in ep["steps"]],
                    "values":   [s["prev_values"]   for s in ep["steps"]],
                },
                cache_path,
            )
        except Exception as e:
            self.log_warning(
                f"realworld: failed to save logprob cache {cache_path.name} ({e})"
            )

    # Bumped to 2: switched from masked-sum to fixed-slice forward, so
    # cached prev_logprobs have a different shape ((B, 1, 1) vs (B, 16, 10)).
    # Old caches will fail the schema check and be recomputed.
    _LOGPROB_CACHE_SCHEMA = 2

    # ---- pad and pack 32 episodes into one Trajectory ---------------------

    def _pad_and_pack(self, episodes: list[dict[str, Any]]) -> Trajectory:
        """Stack episodes into a Trajectory matching the actor's shape contract.

        Length conventions:
          forward_inputs[k], prev_logprobs, rewards : [T_max,   B, ...]
          dones, terminations, truncations,
          prev_values                                : [T_max+1, B, ...]

        dones encodes TERMINATIONS ONLY (not termination | truncation).
        See class docstring for why this lets the unchanged shared GAE
        handle truncations correctly.
        """
        B = len(episodes)
        if B == 0:
            # Should be unreachable: _wait_for_next_iteration verifies all
            # `episodes_per_iteration` rollouts are present before returning,
            # so each rank gets >= 1 episode. If this fires, there is a
            # sharding bug or the iteration's `num_envs` is < world_size.
            raise RuntimeError(
                "realworld: rank received 0 episodes from sharding — "
                "iteration is incomplete or num_envs < world_size; "
                "refusing to push an empty trajectory"
            )
        # Always pad to the configured max_episode_steps so EVERY rank's
        # trajectory has the same shape[0]. Different per-rank T_max leads
        # to rollout_size = T*B varying per actor rank, breaking the
        # `rollout_size % batch_size_per_rank` divisibility check in
        # fsdp_actor_worker.py:953. The loss mask zeros out padded
        # positions, so math is identical.
        observed_max = max(len(ep["steps"]) for ep in episodes)
        if observed_max > self.max_episode_steps:
            self.log_warning(
                f"realworld: observed T_max={observed_max} exceeds configured "
                f"max_episode_steps={self.max_episode_steps}; truncating"
            )
        T_max = self.max_episode_steps

        sample = episodes[0]["steps"][0]
        shape_fwd = {k: v.shape[1:] for k, v in sample["forward_inputs"].items()}
        dtype_fwd = {k: v.dtype for k, v in sample["forward_inputs"].items()}
        rew_shape = sample["rewards"].shape[1:]
        rew_dtype = sample["rewards"].dtype
        term_shape = sample["terminations"].shape[1:]
        prev_logprobs_shape = sample["prev_logprobs"].shape[1:]
        prev_logprobs_dtype = sample["prev_logprobs"].dtype
        prev_values_shape = sample["prev_values"].shape[1:]
        prev_values_dtype = sample["prev_values"].dtype

        forward_inputs = {
            k: torch.zeros((T_max, B, *shape_fwd[k]), dtype=dtype_fwd[k])
            for k in shape_fwd
        }
        rewards = torch.zeros((T_max, B, *rew_shape), dtype=rew_dtype)
        prev_logprobs = torch.zeros(
            (T_max, B, *prev_logprobs_shape), dtype=prev_logprobs_dtype
        )

        T_eff = T_max + 1
        terminations = torch.zeros((T_eff, B, *term_shape), dtype=torch.bool)
        truncations = torch.zeros((T_eff, B, *term_shape), dtype=torch.bool)
        prev_values = torch.zeros(
            (T_eff, B, *prev_values_shape), dtype=prev_values_dtype
        )

        for b, ep in enumerate(episodes):
            T_b = min(len(ep["steps"]), T_max)
            if T_b == 0:
                continue
            for t in range(T_b):
                step = ep["steps"][t]
                for k in shape_fwd:
                    forward_inputs[k][t, b] = step["forward_inputs"][k][0]
                rewards[t, b] = step["rewards"][0]
                prev_logprobs[t, b] = step["prev_logprobs"][0]
                prev_values[t, b] = step["prev_values"][0]
                terminations[t, b] = step["terminations"][0]
                truncations[t, b] = step["truncations"][0]

            # Trailing-position termination/truncation flags. The HDF5
            # writer already sets them at the actual terminal step (see
            # real_rl.py:set_terminal); we hoist them onto position T_b
            # which is the slot GAE indexes via [step+1].
            last = ep["steps"][T_b - 1]
            terminations[T_b, b] = last["terminations"][0].any()
            truncations[T_b, b] = last["truncations"][0].any()
            # Bootstrap value: use last step's prev_values as a proxy.
            prev_values[T_b, b] = last["prev_values"][0]

        # KEY TRICK: dones encodes TERMINATIONS ONLY (not term|trunc).
        # Lets the unchanged shared GAE bootstrap V on truncated episodes.
        dones = terminations.clone().contiguous()

        traj = Trajectory(
            max_episode_length=T_max,
            model_weights_id=self.model_weights_id,
            rewards=rewards.contiguous(),
            dones=dones,
            terminations=terminations.contiguous(),  # telemetry only
            truncations=truncations.contiguous(),     # telemetry only
            prev_logprobs=prev_logprobs.contiguous(),
            prev_values=prev_values.contiguous(),
            forward_inputs={k: v.contiguous() for k, v in forward_inputs.items()},
        )
        return traj
