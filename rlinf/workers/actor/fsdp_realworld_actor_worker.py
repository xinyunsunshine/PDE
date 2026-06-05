"""Realworld variant of EmbodiedFSDPActor.

Inserts a one-shot prev_logprobs / prev_values recompute right after
receiving trajectories. The rollout worker computes prev_logprobs at B=1
(per chunk-step). The PPO inner loop runs forward at B=micro_batch_size.
The model's bf16 reductions are batch-size dependent (verified empirically
in scratch/check_old_logprob.py: B=1 vs B=4 differ by ~1.4e-2 per element),
so without this recompute prev/new logprobs disagree at iter 0 by enough
to drive approx_kl into the 10^14 range even with matched weights.

This recompute runs the actor's own forward at micro_batch_size, in flat
T*B order, and overwrites prev_logprobs/prev_values in rollout_batch.
After the overwrite, PPO's first inner-iter forward (also at
micro_batch_size) produces byte-equal logprobs → ratio = 1 → KL ≈ 0 at
iter 0. After the first PPO update, weights drift naturally and KL grows
as standard PPO expects.

EmbodiedFSDPActor (used by all sim runs) is unchanged.
"""

from __future__ import annotations

import torch

from rlinf.data.embodied_io_struct import Trajectory, convert_trajectories_to_batch
from rlinf.scheduler import Channel
from rlinf.utils.metric_utils import compute_loss_mask, compute_split_num
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


class RealworldFSDPActor(EmbodiedFSDPActor):
    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        # NOTE: this body mirrors EmbodiedFSDPActor.recv_rollout_trajectories
        # so we can insert the recompute step BETWEEN convert_trajectories_to_batch
        # and _process_received_rollout_batch. We deliberately do NOT call
        # super().recv_rollout_trajectories — splitting the parent method
        # is the only way to insert mid-flow without modifying it.
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        self.rollout_batch = convert_trajectories_to_batch(recv_list)

        self._recompute_prev_with_actor_forward()

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

        # Parent's _preprocess_rollout_batch skips loss_mask creation when
        # env.train.ignore_terminations=True (which realworld must set, per
        # CLAUDE.md, so the actor doesn't zero out loss_mask on episodes that
        # ended naturally on the robot). GAE accepts loss_mask=None, but
        # GRPO / REINPP / raw advantage functions dereference it and crash.
        # The realworld worker stores per-chunk-step terminations in `dones`
        # (encoding terminations only, not truncations — see worker docstring),
        # so compute_loss_mask(dones) yields 1s at valid positions and 0s at
        # padded steps after the actual episode end. Build it here so any
        # adv_type works.
        if self.rollout_batch.get("loss_mask", None) is None:
            dones = self.rollout_batch["dones"]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)
            if self.cfg.algorithm.reward_type == "chunk_level":
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]
            self.rollout_batch["loss_mask"] = loss_mask
            self.rollout_batch["loss_mask_sum"] = loss_mask_sum
            self.log_info(
                f"realworld: synthesized loss_mask from dones, shape="
                f"{tuple(loss_mask.shape)}, valid_frac="
                f"{loss_mask.float().mean().item():.3f}"
            )

        # For non-GAE advantage paths under ignore_terminations=True, zero out
        # `dones` before the runner calls compute_advantages_and_returns.
        # The bifranka HDF5 marks task-success with terminations=True at the
        # same chunk-step that carries reward=1; the realworld rollout worker
        # then copies terminations into dones ("KEY TRICK: dones encodes
        # TERMINATIONS ONLY"). algorithms/utils.py:calculate_scores iterates
        # backward over the trajectory and resets its accumulator whenever
        # dones[step+1] is True — which wipes the success reward immediately
        # after it lands, leaving every per-episode score at 0 (and therefore
        # every GRPO advantage at 0, every gradient at 0, every cell's ckpt
        # identical to the SFT init). ignore_terminations=True is supposed to
        # neutralize end-of-episode dones, but that logic only fires in the
        # sim env_worker; realworld uses NoOpEnv so dones come through raw.
        # loss_mask was already computed from the original dones above, so
        # padding-after-episode-end is still masked out for the actor loss.
        # GAE legitimately needs dones to know where to stop bootstrapping V,
        # so we leave it alone for adv_type=gae.
        if (
            self.cfg.env.train.get("ignore_terminations", False)
            and self.cfg.algorithm.adv_type != "gae"
        ):
            self.rollout_batch["dones"] = torch.zeros_like(
                self.rollout_batch["dones"]
            )
            self.log_info(
                "realworld: zeroed dones under ignore_terminations=True for "
                f"adv_type={self.cfg.algorithm.adv_type}; calculate_scores "
                "will accumulate full-episode reward sums into per-episode scores."
            )

    @torch.no_grad()
    def _recompute_prev_with_actor_forward(self) -> None:
        """Overwrite rollout_batch['prev_logprobs', 'prev_values'] using the
        actor's own forward at micro_batch_size, in flat T*B order.
        """
        if "forward_inputs" not in self.rollout_batch:
            return

        fwd_in = self.rollout_batch["forward_inputs"]
        T = self.rollout_batch["prev_logprobs"].shape[0]
        B = self.rollout_batch["prev_logprobs"].shape[1]
        TB = T * B

        # Flatten the (T, B, ...) leading dims to (TB, ...) for each tensor.
        flat_fwd: dict[str, torch.Tensor] = {}
        for k, v in fwd_in.items():
            if isinstance(v, torch.Tensor):
                flat_fwd[k] = v.reshape(TB, *v.shape[2:]).contiguous()

        micro = int(self.cfg.actor.micro_batch_size)
        new_logprobs_chunks: list[torch.Tensor] = []
        new_values_chunks: list[torch.Tensor] = []
        device = torch.cuda.current_device()
        for start in range(0, TB, micro):
            end = min(start + micro, TB)
            chunk = {
                k: v[start:end].to(device, non_blocking=True)
                for k, v in flat_fwd.items()
            }
            with self.amp_context:
                out = self.model(
                    forward_inputs=chunk,
                    compute_logprobs=True,
                    compute_entropy=False,
                    compute_values=(self.cfg.algorithm.adv_type == "gae"),
                    use_cache=False,
                )
            new_logprobs_chunks.append(out["logprobs"].detach().cpu())
            if "values" in out and out["values"] is not None:
                new_values_chunks.append(out["values"].detach().cpu())

        # Reshape (TB, *) back to (T, B, *) and overwrite prev_logprobs.
        new_logprobs = torch.cat(new_logprobs_chunks, dim=0)
        new_logprobs = new_logprobs.reshape(T, B, *new_logprobs.shape[1:])
        old_lp = self.rollout_batch["prev_logprobs"]
        self.rollout_batch["prev_logprobs"] = new_logprobs.to(
            dtype=old_lp.dtype, device=old_lp.device
        )

        # Overwrite prev_values for the executed range [0, T). Keep the
        # bootstrap row at position T from the rollout worker.
        if new_values_chunks:
            new_values = torch.cat(new_values_chunks, dim=0)
            new_values = new_values.reshape(T, B, *new_values.shape[1:])
            old_pv = self.rollout_batch.get("prev_values", None)
            if old_pv is not None:
                old_pv = old_pv.clone()
                # Match trailing singleton dims if old_pv has them
                # (rollout worker stores prev_values as (T+1, B, 1)).
                target_tail = old_pv.shape[2:]
                while new_values.dim() - 2 < len(target_tail):
                    new_values = new_values.unsqueeze(-1)
                old_pv[:T] = new_values.to(dtype=old_pv.dtype, device=old_pv.device)
                self.rollout_batch["prev_values"] = old_pv

        self.log_info(
            f"realworld: actor recomputed prev_logprobs at micro_batch_size={micro} "
            f"(TB={TB}, T={T}, B={B}); prev_logprobs shape="
            f"{tuple(self.rollout_batch['prev_logprobs'].shape)}"
        )
