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

"""EmbodiedFSDPActorHER: EmbodiedFSDPActor extended with Hindsight Experience Replay."""

import torch

from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.actor.her_processor import HERProcessor


class EmbodiedFSDPActorHER(EmbodiedFSDPActor):
    """EmbodiedFSDPActor extended with HER reward relabeling.

    * ``_process_received_rollout_batch`` — preserves string fields across the
      tensor reshape (base class only handles tensors), runs HER, then applies
      the shared postprocess hook.
    * ``compute_advantages_and_returns`` — runs HER first (relabels instructions,
      re-evaluates rewards), then calls super() so
      advantages are computed on the relabeled rewards. HER metrics are merged
      into the returned metrics dict.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        modification_type = cfg.algorithm.get("modification_type")
        if modification_type == "group_ranking":
            from rlinf.workers.actor.group_ranking_processor import (
                GroupRankingProcessor,
            )

            self._her_processor = GroupRankingProcessor(
                cfg=cfg,
                rank=self._rank,
                get_model_fn=lambda: self.model,
                log_warning_fn=self.log_warning,
            )
        elif modification_type == "majority_voting":
            from rlinf.workers.actor.her_majority_voting_processor import (
                HERMajorityVotingProcessor,
            )

            self._her_processor = HERMajorityVotingProcessor(
                cfg=cfg,
                rank=self._rank,
                get_model_fn=lambda: self.model,
                log_warning_fn=self.log_warning,
            )
        else:
            self._her_processor = HERProcessor(
                cfg=cfg,
                rank=self._rank,
                get_model_fn=lambda: self.model,
                log_warning_fn=self.log_warning,
            )
        self._her_metrics: dict = {}

    def _preprocess_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Reshape tensors, compute loss_mask, and reorder string fields."""
        rollout_epoch = self.cfg.algorithm.rollout_epoch

        # Pop non-tensor fields that process_nested_dict_for_adv ignores.
        task_descs = rollout_batch.pop("task_descriptions", None)
        scene_objs = rollout_batch.pop("scene_objects", None)

        rollout_batch = super()._preprocess_rollout_batch(rollout_batch)

        # Re-insert string fields with epoch-major reordering to match the
        # tensor reshape: original layout is [env_j * rollout_epoch + epoch_i],
        # target layout is [epoch_i * n_envs + env_j].
        if task_descs is not None:
            bsz = len(task_descs) // rollout_epoch
            rollout_batch["task_descriptions"] = [
                task_descs[env_j * rollout_epoch + epoch_i]
                for epoch_i in range(rollout_epoch)
                for env_j in range(bsz)
            ]
        if scene_objs is not None:
            bsz = len(scene_objs) // rollout_epoch
            rollout_batch["scene_objects"] = [
                scene_objs[env_j * rollout_epoch + epoch_i]
                for epoch_i in range(rollout_epoch)
                for env_j in range(bsz)
            ]

        return rollout_batch

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        rollout_batch = self._preprocess_rollout_batch(rollout_batch)

        # Run HER: relabels instructions and rewards in-place; store metrics for
        # compute_advantages_and_returns which runs after this method returns.
        her_result = self._her_processor(rollout_batch)
        self._her_metrics = her_result["metrics"]
        return self._postprocess_rollout_batch(rollout_batch)

    def compute_advantages_and_returns(self) -> dict:
        rollout_metrics = super().compute_advantages_and_returns()
        rollout_metrics.update(self._her_metrics)
        return rollout_metrics

    def set_global_step(self, global_step: int) -> None:
        super().set_global_step(global_step)
        self._her_processor.version = global_step
