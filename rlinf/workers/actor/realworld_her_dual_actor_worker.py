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

"""RealworldFSDPActorHERDual: HER-dual training on bifranka real-world rollouts.

Composes RealworldFSDPActor (one-shot prev_logprobs recompute, loss_mask
synthesis under ignore_terminations=True) with EmbodiedFSDPActorHERDual
(paired original+HER mini-batches with summed losses).

MRO: RealworldFSDPActorHERDual → RealworldFSDPActor → EmbodiedFSDPActorHERDual
→ EmbodiedFSDPActorHER → EmbodiedFSDPActor.

Net behavior per outer iter:
  1. recv_rollout_trajectories (Realworld) recvs HDF5-converted Trajectories,
     recomputes prev_logprobs at micro_batch_size, then calls
     _process_received_rollout_batch.
  2. _process_received_rollout_batch (HERDual) calls _preprocess_rollout_batch
     (overridden below to synthesize loss_mask from `dones` when missing — so
     both the original and HER clone carry one), clones the batch, relabels
     instructions / rewards on the clone via the HER processor, then filters
     both branches independently.
  3. compute_advantages_and_returns (HERDual) produces orig + her/* metrics.
  4. run_training (HERDual) iterates paired mini-batches and sums the two
     branch losses (HER scaled by her_loss_weight) before each backward.
"""

from __future__ import annotations

import torch

from rlinf.utils.metric_utils import compute_loss_mask
from rlinf.workers.actor.fsdp_realworld_actor_worker import RealworldFSDPActor
from rlinf.workers.actor.her_dual_actor_worker import EmbodiedFSDPActorHERDual


class RealworldFSDPActorHERDual(RealworldFSDPActor, EmbodiedFSDPActorHERDual):
    def _preprocess_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Reshape + reorder string fields, then synthesize loss_mask if missing.

        The base EmbodiedFSDPActor only builds loss_mask when
        ``not auto_reset and not ignore_terminations``. Realworld training runs
        with ``ignore_terminations=True`` (CLAUDE.md: required for GRPO/HER so
        the VLM reward at the last step isn't zeroed), so the base skips it.
        HER dual then clones the batch and may invoke
        ``_apply_her_nonrelabeled_mask``, which asserts loss_mask is present.
        Synthesizing it here (before HERDual clones) keeps loss_mask on both
        copies and matches the post-hoc synthesis in
        RealworldFSDPActor.recv_rollout_trajectories.
        """
        rollout_batch = super()._preprocess_rollout_batch(rollout_batch)

        if rollout_batch.get("loss_mask", None) is None:
            dones = rollout_batch["dones"]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)
            if self.cfg.algorithm.reward_type == "chunk_level":
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]
            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum
            self.log_info(
                f"realworld_her_dual: synthesized loss_mask from dones in "
                f"preprocess, shape={tuple(loss_mask.shape)}, valid_frac="
                f"{loss_mask.float().mean().item():.3f}"
            )

        return rollout_batch
