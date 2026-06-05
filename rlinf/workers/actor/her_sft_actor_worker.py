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

"""EmbodiedFSDPActorHERSFT: joint RL + Hindsight Experience Replay SFT training.

Overview
--------
After each rollout the actor:
  1. Runs HERProcessor to relabel trajectories (group-based VLM calls).
  2. Extracts relabeled (forward_inputs, action_tokens) and stores them in a
     bounded CPU replay buffer, then restores the original batch so GRPO
     trains on unmodified data.
  3. During RL training, samples a micro-batch from the buffer and adds a
     supervised SFT loss term to the RL loss:

       loss = rl_loss + her_sft_loss_weight * sft_loss

This wastes no trajectory data – trajectories discarded by RL (off-policy after
the first gradient step) continue contributing to the SFT objective.
"""

import os
from typing import Optional

import torch

from rlinf.config import SupportedModel
from rlinf.scheduler import Worker
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.workers.actor.her_buffer_actor_worker import EmbodiedFSDPActorHERBuffer


class EmbodiedFSDPActorHERSFT(EmbodiedFSDPActorHERBuffer):
    """EmbodiedFSDPActorHERBuffer extended with an SFT auxiliary loss.

    Uses ``HERProcessor`` for group-based VLM relabeling (via the buffer
    base class), storing relabeled data in a CPU replay buffer.  During
    RL training, samples from the buffer and adds a supervised SFT loss
    (negative mean log-likelihood) after the GRPO backward.

    New config keys under ``cfg.actor.her_sft``:
      replay_buffer_size (int, default 8): max number of rollout batches to keep.
      sft_loss_weight (float, default 0.1): weight λ for the SFT loss term.
      relabel_per_group (int, default 1): max trajectories per group to store
          in the buffer (limits buffer memory while preserving group coverage).
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        her_cfg = cfg.actor.get("her_sft", {})
        self.her_sft_loss_weight = float(her_cfg.get("sft_loss_weight", 0.1))

    def _get_buffer_config(self, cfg) -> dict:
        return cfg.actor.get("her_sft", {})

    # ------------------------------------------------------------------
    # Microbatch backward override + HER-SFT backward
    # ------------------------------------------------------------------

    def _run_microbatch_backward(
        self,
        batch: dict,
        loss: torch.Tensor,
        backward_ctx,
        idx: int,
        metrics_data: dict,
    ) -> float:
        """RL backward + HER-SFT backward."""
        rl_loss = super()._run_microbatch_backward(
            batch, loss, backward_ctx, idx, metrics_data
        )
        is_last = (idx + 1) == self.gradient_accumulation
        sft_loss = self._her_sft_backward(batch, metrics_data, is_last)
        return rl_loss + (sft_loss if sft_loss is not None else 0.0)

    def _her_sft_backward(
        self,
        batch: dict,
        metrics_data: dict,
        is_last_micro_batch: bool,
    ) -> Optional[float]:
        """Run a separate SFT forward+backward after the RL backward.

        Because this runs after ``rl_loss.backward()`` has returned, RL
        activations have already been freed.  Only SFT activations are live
        during the SFT backward, halving peak GPU memory vs. combining both
        losses into one backward pass.
        """
        if not self._her_replay_buffer:
            return None

        micro_batch_size = batch["prev_logprobs"].shape[0]
        sft_batch_cpu = self._sample_her_buffer_microbatch(micro_batch_size)
        if sft_batch_cpu is None:
            return None

        device = f"{Worker.torch_device_type}:{int(os.environ['LOCAL_RANK'])}"
        sft_forward_inputs = put_tensor_device(sft_batch_cpu, device)

        # Same kwargs as the RL forward pass so the model behaves identically.
        kwargs = {}
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ]:
            kwargs["temperature"] = self.cfg.algorithm.sampling_params.temperature_train
            kwargs["top_k"] = self.cfg.algorithm.sampling_params.top_k

        with self.amp_context:
            sft_output = self.model(
                forward_inputs=sft_forward_inputs,
                compute_logprobs=True,
                compute_entropy=False,
                compute_values=False,
                use_cache=False,
                **kwargs,
            )

        # sft_logprobs shape: [B, action_dim] or [B, action_dim, num_chunks].
        # Negative mean log-likelihood = supervised cross-entropy loss.
        sft_logprobs: torch.Tensor = sft_output["logprobs"]
        sft_loss = -sft_logprobs.mean()

        sft_loss_val = sft_loss.detach().item()
        weighted_loss_val = self.her_sft_loss_weight * sft_loss_val
        if sft_loss_val > 10.0:
            self.log_warning(
                f"[HER-SFT] sft_loss={sft_loss_val:.4f} (weighted: {weighted_loss_val:.4f}) is "
                "unusually large — check buffer contents and sft_loss_weight."
            )
        else:
            self.log_info(
                f"[HER-SFT] sft_loss={sft_loss_val:.4f} (weighted: {weighted_loss_val:.4f})"
            )

        metrics_data["her_sft/sft_loss"] = sft_loss_val
        metrics_data["her_sft/buffer_size"] = float(len(self._her_replay_buffer))

        sft_loss_scaled = (
            self.her_sft_loss_weight * sft_loss / self.gradient_accumulation
        )
        backward_ctx = self.before_micro_batch(self.model, is_last_micro_batch)
        with backward_ctx:
            self.grad_scaler.scale(sft_loss_scaled).backward()
        return sft_loss_scaled.detach().item()
