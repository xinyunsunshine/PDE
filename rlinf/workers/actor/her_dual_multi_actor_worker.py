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

"""EmbodiedFSDPActorHERDualMulti: M independently-relabeled HER batches.

Extends ``EmbodiedFSDPActorHERDual`` to train on 1 original batch plus M
HER-relabeled copies.  The combined loss is:

    rl_loss_weight * L(orig) + her_loss_weight * (1/M) * sum_i L(her_i)

Each HER batch is independently cloned and relabeled (stochastic group
selection + VLM variance), providing more diverse hindsight signal.

Config keys (under ``cfg.algorithm``):
  num_her_batches (int, default 1): number of independent HER batches (M).
  her_loss_weight (float, default 1.0): coefficient on the HER branch loss.
"""

import os
from typing import Any

import numpy as np
import torch

from rlinf.scheduler import Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.fsdp_actor_worker import (
    EmbodiedFSDPActor,
    process_nested_dict_for_train,
)
from rlinf.workers.actor.her_dual_actor_worker import (
    EmbodiedFSDPActorHERDual,
    _clone_rollout_batch,
)


class EmbodiedFSDPActorHERDualMulti(EmbodiedFSDPActorHERDual):
    """Train on 1 original + M independently HER-relabeled batches.

    Overrides the dual actor to clone and relabel M times instead of once.
    The HER loss contribution is averaged across the M batches before being
    scaled by ``her_loss_weight``.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_her_batches: int = int(cfg.algorithm.get("num_her_batches", 1))
        assert self.num_her_batches >= 1, (
            f"num_her_batches must be >= 1, got {self.num_her_batches}"
        )
        self.her_rollout_batches: list[dict[str, Any]] = []
        self._filter_metrics_her_list: list[dict[str, float]] = []
        self._her_metrics_list: list[dict[str, Any]] = []

    # ── Rollout batch reception ──────────────────────────────────────────────

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        rollout_batch = self._preprocess_rollout_batch(rollout_batch)

        M = self.num_her_batches
        self.her_rollout_batches = []
        self._her_metrics_list = []
        for i in range(M):
            her_clone = _clone_rollout_batch(rollout_batch)
            her_result = self._her_processor(
                her_clone, force_all_groups=self.her_dual_force_all_groups
            )
            self._her_metrics_list.append(her_result["metrics"])
            self.her_rollout_batches.append(her_clone)

        rollout_batch = self._postprocess_rollout_batch(rollout_batch)
        self._filter_metrics_orig = dict(self._rollout_postprocess_metrics)

        self._filter_metrics_her_list = []
        for i in range(M):
            self.her_rollout_batches[i] = self._postprocess_rollout_batch(
                self.her_rollout_batches[i]
            )
            self._filter_metrics_her_list.append(
                dict(self._rollout_postprocess_metrics)
            )

        self.her_rollout_batch = self.her_rollout_batches[0]
        return rollout_batch

    # ── Advantages & returns ─────────────────────────────────────────────────

    def compute_advantages_and_returns(self) -> dict:
        rollout_metrics: dict = dict(
            EmbodiedFSDPActor.compute_advantages_and_returns(self)
        )
        for k, v in self._filter_metrics_orig.items():
            rollout_metrics[f"filter_orig/{k}"] = v

        M = self.num_her_batches
        saved_rollout_batch = self.rollout_batch
        saved_filter_metrics = self._rollout_postprocess_metrics

        for i in range(M):
            self.rollout_batch = self.her_rollout_batches[i]
            try:
                her_metrics: dict = dict(
                    EmbodiedFSDPActor.compute_advantages_and_returns(self)
                )
            finally:
                self.her_rollout_batches[i] = self.rollout_batch
                self.rollout_batch = saved_rollout_batch
                self._rollout_postprocess_metrics = saved_filter_metrics

            prefix = "her" if M == 1 else f"her_{i}"
            for k, v in her_metrics.items():
                rollout_metrics[f"{prefix}/{k}"] = v
            for k, v in self._filter_metrics_her_list[i].items():
                rollout_metrics[f"filter_{prefix}/{k}"] = v
            for k, v in self._her_metrics_list[i].items():
                rollout_metrics[f"{prefix}/{k}"] = v

        self.her_rollout_batch = self.her_rollout_batches[0]
        return rollout_metrics

    # ── Training ─────────────────────────────────────────────────────────────

    @Worker.timer("run_training")
    def run_training(self):
        """Multi-HER training loop: 1 orig + M HER batches per micro-batch."""
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        self.model.train()

        M = self.num_her_batches
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )
            for i in range(M):
                self.her_rollout_batches[i] = process_nested_dict_for_train(
                    self.her_rollout_batches[i], shuffle_id
                )

        self._compute_ref_logprobs()
        for i in range(M):
            self._compute_ref_logprobs_for_batch(i)

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        metrics: dict = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for epoch_idx in range(update_epoch):
            print(
                f"[Actor-HERDualMulti] Update epoch {epoch_idx + 1}/{update_epoch} "
                f"(M={M})."
            )
            self._run_single_update_epoch_multi(metrics)

        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()

        mean_metric_dict = {key: np.nanmean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        return mean_metric_dict

    def _compute_ref_logprobs_for_batch(self, batch_idx: int) -> None:
        """Compute ref logprobs for her_rollout_batches[batch_idx]."""
        if self.kl_beta <= 0 or self.ref_policy_state_dict is None:
            return
        saved = self.rollout_batch
        self.rollout_batch = self.her_rollout_batches[batch_idx]
        try:
            self._compute_ref_logprobs()
        finally:
            self.her_rollout_batches[batch_idx] = self.rollout_batch
            self.rollout_batch = saved

    # ── Multi-HER micro-batch loop ──────────────────────────────────────────

    def _run_single_update_epoch_multi(self, metrics: dict) -> None:
        M = self.num_her_batches
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)

        for i in range(M):
            assert rollout_size == self.her_rollout_batches[i]["prev_logprobs"].size(0)

        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        n_global = rollout_size // batch_size_per_rank

        orig_global_iter = split_dict_to_chunk(self.rollout_batch, n_global)
        her_global_iters = [
            split_dict_to_chunk(self.her_rollout_batches[i], n_global) for i in range(M)
        ]

        for global_idx in range(n_global):
            orig_global = orig_global_iter[global_idx]
            her_globals = [her_global_iters[i][global_idx] for i in range(M)]

            size = orig_global["prev_logprobs"].shape[0]
            assert size % self.cfg.actor.micro_batch_size == 0
            micro_cnt = size // self.cfg.actor.micro_batch_size

            orig_micros = split_dict_to_chunk(orig_global, micro_cnt)
            her_micros_list = [
                split_dict_to_chunk(her_globals[i], micro_cnt) for i in range(M)
            ]

            self.optimizer.zero_grad()
            for idx in range(micro_cnt):
                is_last_mb = (idx + 1) == self.gradient_accumulation
                device = f"{Worker.torch_device_type}:{int(os.environ['LOCAL_RANK'])}"

                m_orig = put_tensor_device(orig_micros[idx], device)

                # Original branch: always no_sync
                no_sync_ctx = self.before_micro_batch(
                    self.model, is_last_micro_batch=False
                )
                orig_loss, orig_md = self._microbatch_forward_and_loss(m_orig)
                scaled_orig = (
                    self.rl_loss_weight * orig_loss
                ) / self.gradient_accumulation
                with no_sync_ctx:
                    self.grad_scaler.scale(scaled_orig).backward()
                orig_loss_val = orig_loss.detach().item()
                del orig_loss, scaled_orig, m_orig

                # HER branches
                her_loss_vals: list[float] = []
                her_mds: list[dict] = []
                for i in range(M):
                    m_her = put_tensor_device(her_micros_list[i][idx], device)

                    is_final_backward = i == M - 1
                    if is_final_backward:
                        her_ctx = self.before_micro_batch(
                            self.model, is_last_micro_batch=is_last_mb
                        )
                    else:
                        her_ctx = self.before_micro_batch(
                            self.model, is_last_micro_batch=False
                        )

                    her_loss, her_md = self._microbatch_forward_and_loss(
                        m_her,
                        clip_ratio_overrides={
                            "high": self.her_clip_ratio_high,
                            "low": self.her_clip_ratio_low,
                        },
                    )
                    scaled_her = (
                        self.her_loss_weight * her_loss / M
                    ) / self.gradient_accumulation
                    with her_ctx:
                        self.grad_scaler.scale(scaled_her).backward()
                    her_loss_vals.append(her_loss.detach().item())
                    her_mds.append(her_md)
                    del her_loss, scaled_her, m_her

                # Merge metrics
                merged_md: dict = dict(orig_md)
                for i in range(M):
                    prefix = "her" if M == 1 else f"her_{i}"
                    merged_md.update(
                        {f"{prefix}/{k}": v for k, v in her_mds[i].items()}
                    )
                merged_md["actor/orig_loss"] = orig_loss_val
                merged_md["actor/her_loss_mean"] = sum(her_loss_vals) / M
                if M > 1:
                    for i in range(M):
                        merged_md[f"actor/her_{i}_loss"] = her_loss_vals[i]
                else:
                    merged_md["actor/her_loss"] = her_loss_vals[0]
                merged_md["actor/total_loss"] = (
                    self.rl_loss_weight * orig_loss_val
                    + self.her_loss_weight * sum(her_loss_vals) / M
                ) / self.gradient_accumulation
                append_to_dict(metrics, merged_md)

            self.torch_platform.empty_cache()

            grad_norm, lr_list = self.optimizer_step()
            step_data = {
                "actor/grad_norm": grad_norm,
                "actor/lr": lr_list[0],
            }
            if len(lr_list) > 1:
                step_data["critic/lr"] = lr_list[1]
            append_to_dict(metrics, step_data)
