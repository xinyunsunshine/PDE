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

"""EmbodiedFSDPActorHERDualPCGrad: PCGrad-style HER gradient projection.

Extends ``EmbodiedFSDPActorHERDualSeparateGrad`` by projecting the clipped HER
gradient to remove its component along the orig RL gradient when the two
conflict (cos < 0).  When they agree (cos >= 0), the HER gradient is kept
as-is.  This preserves the orthogonal component of HER even when gradients
conflict, unlike the cosine-gating approach which drops g_her entirely.

The projection and cosine are computed over the full (FSDP-sharded) model
gradient via an all_reduce of the local dot-product and squared-norm
accumulators.
"""

import os

import torch
import torch.distributed as dist

from rlinf.scheduler import Worker
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.workers.actor.her_dual_separate_grad_actor_worker import (
    EmbodiedFSDPActorHERDualSeparateGrad,
)


class EmbodiedFSDPActorHERDualPCGrad(EmbodiedFSDPActorHERDualSeparateGrad):
    """HER-dual variant with PCGrad-style gradient projection.

    After each phase's independent clip step, computes cos(g_orig, g_her) over
    the full model gradient.  When cos < 0 (conflict), projects g_her onto the
    plane orthogonal to g_orig, removing only the conflicting component.  When
    cos >= 0, g_her is kept unchanged.
    """

    def _compute_grad_cos_and_proj(
        self,
        orig_snapshot: list[tuple[torch.nn.Parameter, torch.Tensor]],
    ) -> tuple[float, float]:
        """Global cosine similarity and projection coefficient.

        Returns (cos_sim, proj_coeff) where proj_coeff = dot / ||g_orig||².
        Uses a single all_reduce of a 3-element buffer.
        """
        buf = torch.zeros(3, device=orig_snapshot[0][1].device)
        for p, g_orig in orig_snapshot:
            if p.grad is None:
                continue
            buf[0] += (g_orig * p.grad).sum()  # dot(g_orig, g_her)
            buf[1] += g_orig.pow(2).sum()  # ||g_orig||²
            buf[2] += p.grad.pow(2).sum()  # ||g_her||²
        dist.all_reduce(buf)
        dot, sq_o, sq_h = buf.tolist()
        cos_sim = float(dot / (sq_o**0.5 * sq_h**0.5 + 1e-8))
        proj_coeff = float(dot / (sq_o + 1e-8))
        return cos_sim, proj_coeff

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

        device_str = f"{Worker.torch_device_type}:{int(os.environ['LOCAL_RANK'])}"

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
                with her_ctx:
                    self.grad_scaler.scale(scaled_her).backward()
                her_micro_data.append((her_loss.detach().item(), her_md))
                del her_loss, scaled_her

            # Unscale + clip HER grads, then apply PCGrad projection when
            # g_orig and g_her conflict.
            self.grad_scaler.unscale_(self.optimizer)
            her_grad_norm = self._strategy.clip_grad_norm_(model=self.model)

            cos_sim, proj_coeff = self._compute_grad_cos_and_proj(
                orig_grad_snapshot
            )
            projected = 0.0
            if cos_sim < 0:
                # g_her_proj = g_her - (dot / ||g_orig||²) * g_orig
                for p, g_orig in orig_grad_snapshot:
                    if p.grad is not None:
                        p.grad.add_(g_orig, alpha=-proj_coeff)
                projected = 1.0

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
                "actor/her_grad_cos_sim": cos_sim,
                "actor/her_grad_projected": projected,
                "actor/lr": lr_list[0],
            }
            if len(lr_list) > 1:
                step_data["critic/lr"] = lr_list[1]
            append_to_dict(metrics, step_data)
