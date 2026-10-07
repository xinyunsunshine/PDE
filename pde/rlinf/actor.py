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

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

from pde.rlinf.data import convert_trajectories_to_batch
from pde.rlinf.model import promote_openpi_model
from pde.rlinf.relabeler import HERProcessor


class PDEActor(EmbodiedFSDPActor):
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
        self._her_processor = HERProcessor(
            cfg=cfg,
            rank=self._rank,
            get_model_fn=lambda: self.model,
            log_warning_fn=self.log_warning,
        )
        self._her_metrics: dict = {}

    def model_provider_func(self):
        """Build the RLinf policy and promote supported models to PDE subclasses."""
        return promote_openpi_model(super().model_provider_func())

    async def recv_rollout_trajectories(self, input_channel) -> None:
        """Receive RLinf trajectories while preserving PDE language metadata."""
        clear_memory(sync=False)
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)
        trajectories: list[Trajectory] = []
        for _ in range(split_num):
            trajectory = await input_channel.get(async_op=True).async_wait()
            trajectories.append(trajectory)
        self.rollout_batch = convert_trajectories_to_batch(trajectories)
        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

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


# Backward-compatible name used by early experiment configs.
EmbodiedFSDPActorHER = PDEActor
