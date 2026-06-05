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

"""EmbodiedFSDPActorHERShuffledDual: orig + HER + shuffled-HER joint training.

Inherits from ``EmbodiedFSDPActorHERDual`` and threads a third "shuffled HER"
batch through the parent's training loop without overriding ``run_training``.
The shuffled batch reuses the parent's data flow via four small overrides:

  * ``_process_received_rollout_batch`` adds a third clone (process_shuffled).
  * ``compute_advantages_and_returns`` calls ``super()`` and appends the
    shuffled batch's advantages.
  * ``_compute_ref_logprobs_her`` chains ``super()`` and re-derives the parent's
    seed-based shuffle to keep the shuffled batch paired with the other two.
  * ``_run_single_update_epoch_dual`` (the per-epoch hook called by parent's
    ``run_training``) is redirected to a triple-branch version.

Config keys (under ``cfg.algorithm``):
  her_shuffled_instruction_pool (list[str], required): candidate instructions.
  her_shuffled_loss_weight (float, default 1.0): coefficient on the shuffled loss.
"""

import os
from typing import Any

import torch

from rlinf.scheduler import Worker
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.workers.actor.fsdp_actor_worker import (
    EmbodiedFSDPActor,
    process_nested_dict_for_train,
)
from rlinf.workers.actor.her_dual_actor_worker import (
    EmbodiedFSDPActorHERDual,
    _clone_rollout_batch,
)


class EmbodiedFSDPActorHERShuffledDual(EmbodiedFSDPActorHERDual):
    """Train on orig + VLM-relabeled HER + shuffled-instruction HER.

    Shuffled HER replaces each group's instruction with a random sample from
    ``her_shuffled_instruction_pool`` (excluding the group's own task) and runs
    VLM reward eval on the result. Combined loss::

        rl_loss_weight * L(orig)
        + her_loss_weight * L(her)
        + her_shuffled_loss_weight * L(shuffled)
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.her_shuffled_loss_weight: float = float(
            cfg.algorithm.get("her_shuffled_loss_weight", 1.0)
        )
        instruction_pool = cfg.algorithm.get("her_shuffled_instruction_pool", None)
        if not instruction_pool:
            raise ValueError(
                "cfg.algorithm.her_shuffled_instruction_pool must be set and non-empty "
                "when using EmbodiedFSDPActorHERShuffledDual."
            )
        self._her_shuffled_instruction_pool: list[str] = [
            str(t) for t in instruction_pool
        ]
        self.her_shuffled_rollout_batch: dict[str, Any] = {}
        self._filter_metrics_shuffled: dict[str, float] = {}
        self._her_shuffled_metrics: dict[str, Any] = {}

    # ── Generic helpers ──────────────────────────────────────────────────────

    def _compute_ref_logprobs_for_attr(self, attr_name: str) -> None:
        """Run ref-logprob computation against ``self.{attr_name}`` via temp swap."""
        if self.kl_beta <= 0 or self.ref_policy_state_dict is None:
            return
        saved = self.rollout_batch
        self.rollout_batch = getattr(self, attr_name)
        try:
            self._compute_ref_logprobs()
        finally:
            setattr(self, attr_name, self.rollout_batch)
            self.rollout_batch = saved

    def _compute_advantages_for_attr(self, attr_name: str) -> dict:
        """Compute advantages on the batch at ``self.{attr_name}`` via temp swap."""
        saved_batch = self.rollout_batch
        saved_filter = self._rollout_postprocess_metrics
        self.rollout_batch = getattr(self, attr_name)
        try:
            metrics = dict(EmbodiedFSDPActor.compute_advantages_and_returns(self))
        finally:
            setattr(self, attr_name, self.rollout_batch)
            self.rollout_batch = saved_batch
            self._rollout_postprocess_metrics = saved_filter
        return metrics

    def _backward_branches(
        self,
        branches: list[tuple[dict, float]],
        is_last_mb: bool,
        her_clip_ratio_overrides: dict[str, float] | None = None,
    ) -> tuple[list[float], list[dict]]:
        """Forward + scaled backward for an arbitrary number of paired branches.

        All but the last branch use ``no_sync``; the last uses the real last-mb
        context so FSDP's reduce-scatter fires once with grads from every branch.

        ``her_clip_ratio_overrides`` applies to all branches after the first
        (i.e. HER and shuffled-HER). The first branch (original) always uses
        the global clip ratios.
        """
        loss_vals: list[float] = []
        mds: list[dict] = []
        n = len(branches)
        for i, (micro, weight) in enumerate(branches):
            is_final_branch = i == n - 1
            ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(is_final_branch and is_last_mb),
            )
            overrides = her_clip_ratio_overrides if i > 0 else None
            loss, md = self._microbatch_forward_and_loss(
                micro, clip_ratio_overrides=overrides
            )
            scaled = (weight * loss) / self.gradient_accumulation
            with ctx:
                self.grad_scaler.scale(scaled).backward()
            loss_vals.append(loss.detach().item())
            mds.append(md)
            del loss, scaled
        return loss_vals, mds

    def _parent_shuffle_id(self) -> torch.Tensor:
        """Re-derive the shuffle the parent's run_training applied to orig + HER.

        Must match parent's recipe exactly: rollout_size = T * B from an
        un-shuffled batch, generator seeded with ``cfg.actor.seed + _rank``.
        """
        # her_shuffled_rollout_batch is still un-shuffled at the time we call
        # this (we hook in *after* parent shuffles the other two); use its T,B.
        shuf_pl = self.her_shuffled_rollout_batch["prev_logprobs"]
        rollout_size = shuf_pl.shape[0] * shuf_pl.shape[1]
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        return torch.randperm(rollout_size, generator=g)

    # ── Rollout batch reception ──────────────────────────────────────────────

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        rollout_batch = self._preprocess_rollout_batch(rollout_batch)

        # Run HER and shuffled HER on independent clones of the preprocessed batch.
        her_clone = _clone_rollout_batch(rollout_batch)
        self._her_metrics = self._her_processor(
            her_clone, force_all_groups=self.her_dual_force_all_groups
        )["metrics"]

        shuffled_clone = _clone_rollout_batch(rollout_batch)
        self._her_shuffled_metrics = self._her_processor.process_shuffled(
            shuffled_clone, self._her_shuffled_instruction_pool
        )["metrics"]

        # Filter rewards independently and snapshot filter metrics for each.
        def _filter(batch: dict) -> tuple[dict, dict]:
            out = self._postprocess_rollout_batch(batch)
            return out, dict(self._rollout_postprocess_metrics)

        rollout_batch, self._filter_metrics_orig = _filter(rollout_batch)
        self.her_rollout_batch, self._filter_metrics_her = _filter(her_clone)
        (
            self.her_shuffled_rollout_batch,
            self._filter_metrics_shuffled,
        ) = _filter(shuffled_clone)

        return rollout_batch

    # ── Advantages & returns ─────────────────────────────────────────────────

    def compute_advantages_and_returns(self) -> dict:
        # Parent (dual actor) handles original + VLM-relabeled HER.
        rollout_metrics: dict = super().compute_advantages_and_returns()

        # Append the shuffled branch.
        shuffled_metrics = self._compute_advantages_for_attr("her_shuffled_rollout_batch")
        for k, v in shuffled_metrics.items():
            rollout_metrics[f"her_shuffled/{k}"] = v
        for k, v in self._filter_metrics_shuffled.items():
            rollout_metrics[f"filter_shuffled/{k}"] = v
        # Prefix trajectory_log so it doesn't clobber the HER one (the runner
        # picks up keys matching "trajectory_log" or "*/trajectory_log").
        for k, v in self._her_shuffled_metrics.items():
            key = f"her_shuffled/{k}" if k == "trajectory_log" else k
            rollout_metrics[key] = v

        return rollout_metrics

    # ── Hooks into parent's run_training ─────────────────────────────────────

    def _compute_ref_logprobs_her(self) -> None:
        """Override: also shuffle + ref-logprob the third (shuffled) batch.

        Parent's ``run_training`` shuffles ``rollout_batch`` and
        ``her_rollout_batch`` with a seed-derived permutation just before this
        method runs. We piggyback by re-deriving the same shuffle and applying
        it to ``her_shuffled_rollout_batch`` so all three batches stay paired.
        """
        super()._compute_ref_logprobs_her()

        shuffle_id = self._parent_shuffle_id()
        with torch.no_grad():
            self.her_shuffled_rollout_batch = process_nested_dict_for_train(
                self.her_shuffled_rollout_batch, shuffle_id
            )
        self._compute_ref_logprobs_for_attr("her_shuffled_rollout_batch")

    def _run_single_update_epoch_dual(self, metrics: dict) -> None:
        """Override: parent's run_training calls this — route to the triple version."""
        print("[Actor-HERShuffledDual] Running triple-branch epoch (orig + HER + shuffled).")
        self._run_single_update_epoch_triple(metrics)

    # ── Triple micro-batch loop ───────────────────────────────────────────────

    def _run_single_update_epoch_triple(self, metrics: dict) -> None:
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        assert rollout_size == self.her_rollout_batch["prev_logprobs"].size(0), (
            f"rollout/her size mismatch: {rollout_size} "
            f"vs {self.her_rollout_batch['prev_logprobs'].size(0)}"
        )
        assert rollout_size == self.her_shuffled_rollout_batch["prev_logprobs"].size(0), (
            f"rollout/her_shuffled size mismatch: {rollout_size} "
            f"vs {self.her_shuffled_rollout_batch['prev_logprobs'].size(0)}"
        )

        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        n_global = rollout_size // batch_size_per_rank

        batches = (
            self.rollout_batch,
            self.her_rollout_batch,
            self.her_shuffled_rollout_batch,
        )
        weights = (
            self.rl_loss_weight,
            self.her_loss_weight,
            self.her_shuffled_loss_weight,
        )
        md_prefixes = ("", "her/", "her_shuffled/")
        loss_keys = ("actor/orig_loss", "actor/her_loss", "actor/her_shuffled_loss")
        n_branches = len(batches)

        global_iters = [split_dict_to_chunk(b, n_global) for b in batches]
        device = f"{Worker.torch_device_type}:{int(os.environ['LOCAL_RANK'])}"

        for global_idx in range(n_global):
            globals_per_branch = [global_iters[j][global_idx] for j in range(n_branches)]

            size = globals_per_branch[0]["prev_logprobs"].shape[0]
            assert size % self.cfg.actor.micro_batch_size == 0
            micro_cnt = size // self.cfg.actor.micro_batch_size

            micros_per_branch = [
                split_dict_to_chunk(globals_per_branch[j], micro_cnt)
                for j in range(n_branches)
            ]

            self.optimizer.zero_grad()
            for idx in range(micro_cnt):
                is_last_mb = (idx + 1) == self.gradient_accumulation
                branches = [
                    (
                        put_tensor_device(micros_per_branch[j][idx], device),
                        weights[j],
                    )
                    for j in range(n_branches)
                ]

                loss_vals, mds = self._backward_branches(
                    branches,
                    is_last_mb,
                    her_clip_ratio_overrides={
                        "high": self.her_clip_ratio_high,
                        "low": self.her_clip_ratio_low,
                    },
                )

                merged_md: dict = dict(mds[0])  # orig — no prefix
                for j in range(1, n_branches):
                    merged_md.update(
                        {f"{md_prefixes[j]}{k}": v for k, v in mds[j].items()}
                    )
                for j in range(n_branches):
                    merged_md[loss_keys[j]] = loss_vals[j]
                merged_md["actor/total_loss"] = (
                    sum(weights[j] * loss_vals[j] for j in range(n_branches))
                    / self.gradient_accumulation
                )
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
