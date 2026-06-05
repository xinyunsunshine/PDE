# Copyright 2025 The RLinf Authors.
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

"""EmbodiedFSDPActorHERDualSeparateGrad: separated gradient collection pipeline.

Like ``EmbodiedFSDPActorHERDual`` but runs the orig and HER microbatch phases
sequentially, with independent reduce-scatter / unscale / clip between them.
This enables per-branch gradient norm logging and independent gradient clipping.
"""

import os

import torch

from rlinf.scheduler import Worker
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.workers.actor.her_dual_actor_worker import EmbodiedFSDPActorHERDual


class EmbodiedFSDPActorHERDualSeparateGrad(EmbodiedFSDPActorHERDual):
    """HER-dual variant with separated gradient collection per branch.

    Instead of interleaving orig+HER microbatches with a single reduce-scatter,
    this runs all orig microbatches first (reduce-scatter, unscale, clip, snapshot),
    then all HER microbatches (reduce-scatter, unscale, clip), and sums the
    snapshotted orig grads back before the optimizer step.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        her_grad_norm_scale = cfg.algorithm.get("her_grad_norm_scale", None)
        if her_grad_norm_scale is not None:
            self.her_max_grad_norm = (
                float(cfg.actor.optim.clip_grad) * float(her_grad_norm_scale)
            )
        else:
            self.her_max_grad_norm = None

    def _run_single_update_epoch_dual(self, metrics: dict) -> None:
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        her_rollout_size = self.her_rollout_batch["prev_logprobs"].size(0)
        assert rollout_size == her_rollout_size, (
            f"rollout/her rollout size mismatch: {rollout_size} vs {her_rollout_size}"
        )
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        n_global = rollout_size // batch_size_per_rank

        orig_global_iter = split_dict_to_chunk(self.rollout_batch, n_global)
        her_global_iter = split_dict_to_chunk(self.her_rollout_batch, n_global)

        device_str = (
            f"{Worker.torch_device_type}:{int(os.environ['LOCAL_RANK'])}"
        )

        for orig_global, her_global in zip(orig_global_iter, her_global_iter):
            size = orig_global["prev_logprobs"].shape[0]
            assert size == her_global["prev_logprobs"].shape[0]
            assert size % self.cfg.actor.micro_batch_size == 0
            micro_cnt = size // self.cfg.actor.micro_batch_size

            orig_micros = split_dict_to_chunk(orig_global, micro_cnt)
            her_micros = split_dict_to_chunk(her_global, micro_cnt)

            self.optimizer.zero_grad()

            # -- Phase 1: original RL microbatches --
            orig_micro_data: list[tuple[float, dict]] = []
            for idx, m_orig in enumerate(orig_micros):
                is_last_orig = (idx + 1) == len(orig_micros)
                m_orig = put_tensor_device(m_orig, device_str)
                orig_ctx = self.before_micro_batch(
                    self.model, is_last_micro_batch=is_last_orig
                )
                orig_loss, orig_md = self._microbatch_forward_and_loss(
                    m_orig, kl_beta_override=self.kl_beta
                )
                scaled_orig = (
                    self.rl_loss_weight * orig_loss
                ) / self.gradient_accumulation
                with orig_ctx:
                    self.grad_scaler.scale(scaled_orig).backward()
                orig_micro_data.append((orig_loss.detach().item(), orig_md))
                del orig_loss, scaled_orig

            # Reduce-scatter has fired on the last orig backward. Unscale +
            # clip the orig grads, then steal them so the HER phase
            # accumulates on a clean slate.
            self.grad_scaler.unscale_(self.optimizer)
            orig_grad_norm = self._strategy.clip_grad_norm_(model=self.model)
            orig_grad_snapshot = self._steal_param_grads()
            if self.grad_scaler._enabled:
                self.grad_scaler._per_optimizer_states.pop(
                    id(self.optimizer), None
                )

            # -- Phase 2: HER microbatches --
            her_micro_data: list[tuple[float, dict]] = []
            for idx, m_her in enumerate(her_micros):
                is_last_her = (idx + 1) == len(her_micros)
                m_her = put_tensor_device(m_her, device_str)
                her_ctx = self.before_micro_batch(
                    self.model, is_last_micro_batch=is_last_her
                )
                her_loss, her_md = self._microbatch_forward_and_loss(
                    m_her,
                    kl_beta_override=self.kl_beta_her,
                    clip_ratio_overrides={
                        "high": self.her_clip_ratio_high,
                        "low": self.her_clip_ratio_low,
                    },
                )
                scaled_her = (
                    self.her_loss_weight * her_loss
                ) / self.gradient_accumulation
                if (
                    self.her_loss_clip is not None
                    and her_loss.detach().item() > self.her_loss_clip
                ):
                    scaled_her = scaled_her * (
                        self.her_loss_clip / her_loss.detach().item()
                    )
                with her_ctx:
                    self.grad_scaler.scale(scaled_her).backward()
                her_micro_data.append((her_loss.detach().item(), her_md))
                del her_loss, scaled_her

            # Unscale + clip HER grads, then add the (already-clipped) orig
            # grads back so the optimizer step sees the sum.
            self.grad_scaler.unscale_(self.optimizer)
            her_grad_norm = self._strategy.clip_grad_norm_(
                model=self.model, max_norm=self.her_max_grad_norm
            )
            self._add_grads_from_snapshot(orig_grad_snapshot)
            del orig_grad_snapshot

            # Per-microbatch metrics.
            for (orig_loss_val, orig_md), (her_loss_val, her_md) in zip(
                orig_micro_data, her_micro_data
            ):
                merged_md: dict = dict(orig_md)
                merged_md.update({f"her/{k}": v for k, v in her_md.items()})
                merged_md["actor/orig_loss"] = orig_loss_val
                merged_md["actor/her_loss"] = her_loss_val
                merged_md["actor/total_loss"] = (
                    self.rl_loss_weight * orig_loss_val
                    + self.her_loss_weight * her_loss_val
                ) / self.gradient_accumulation
                append_to_dict(metrics, merged_md)

            self.torch_platform.empty_cache()

            grad_norm, lr_list = self.optimizer_step(already_unscaled=True)
            step_data = {
                "actor/grad_norm": grad_norm,
                "actor/orig_grad_norm": orig_grad_norm,
                "actor/her_grad_norm": her_grad_norm,
                "actor/lr": lr_list[0],
            }
            if len(lr_list) > 1:
                step_data["critic/lr"] = lr_list[1]
            append_to_dict(metrics, step_data)

    # -- Gradient snapshot helpers --

    def _steal_param_grads(
        self,
    ) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
        """Move each param's current grad into a snapshot list.

        Sets ``p.grad = None`` — avoids a clone+zero transient 2x memory spike.
        """
        snapshot: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        for p in self.model.parameters():
            if p.grad is not None:
                snapshot.append((p, p.grad))
                p.grad = None
        return snapshot

    def _add_grads_from_snapshot(
        self,
        snapshot: list[tuple[torch.nn.Parameter, torch.Tensor]],
    ) -> None:
        """In-place add snapshotted grads into each param's current grad."""
        for p, g in snapshot:
            if p.grad is None:
                p.grad = g
            else:
                p.grad.add_(g)
