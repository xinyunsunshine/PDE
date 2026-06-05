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

"""EmbodiedFSDPActorHERDual: joint RL training on original + HER-relabeled rollouts.

Each training step:
  1. Duplicate the preprocessed ``rollout_batch`` into ``her_rollout_batch``.
  2. Run HER relabeling on the duplicate only (leaves the original untouched).
  3. Apply reward filtering (``_postprocess_rollout_batch``) on both batches
     independently.
  4. Split both into mini-batches of equal shape and, for each paired micro-batch,
     compute the RL loss on both, sum them with the HER loss re-weighted by
     ``her_loss_weight``, and backprop the combined gradient.

Config keys (under ``cfg.algorithm``):
  her_loss_weight (float, default 1.0): coefficient on the HER-branch loss.
"""

import os
from typing import Any

import numpy as np
import torch

from rlinf.algorithms.registry import policy_loss
from rlinf.algorithms.utils import kl_penalty
from rlinf.config import SupportedModel
from rlinf.scheduler import Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.utils.utils import clear_memory, masked_mean, reshape_entropy
from rlinf.workers.actor.fsdp_actor_worker import (
    EmbodiedFSDPActor,
    process_nested_dict_for_train,
)
from rlinf.workers.actor.her_embodied_actor_worker import EmbodiedFSDPActorHER


def _clone_rollout_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Deep-clone a rollout batch so in-place HER patches do not leak.

    Tensors are ``.clone()``d, dicts recursed into, and lists shallow-copied so
    per-trajectory string fields (task_descriptions, scene_objects) can be
    mutated independently on each copy.
    """
    cloned: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            cloned[k] = v.clone()
        elif isinstance(v, dict):
            cloned[k] = _clone_rollout_batch(v)
        elif isinstance(v, list):
            cloned[k] = list(v)
        else:
            cloned[k] = v
    return cloned


class EmbodiedFSDPActorHERDual(EmbodiedFSDPActorHER):
    """EmbodiedFSDPActorHER variant that trains on paired (original, HER) batches.

    Overrides
    ---------
    ``_process_received_rollout_batch``
        Preprocess once, duplicate, run HER on the duplicate, then filter
        rewards on both.
    ``compute_advantages_and_returns``
        Compute advantages for both batches; metrics from the HER branch are
        prefixed with ``her/`` and filter metrics with ``filter_{orig,her}/``.
    ``run_training``
        Ref-logprob + reshape both batches, then iterate paired mini-batches
        and sum the two RL losses (HER branch scaled by ``her_loss_weight``)
        before each backward.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.her_loss_weight: float = float(
            cfg.algorithm.get("her_loss_weight", 1.0)
        )
        self.her_dual_force_all_groups: bool = bool(
            cfg.algorithm.get("her_dual_force_all_groups", False)
        )
        _rsp = cfg.algorithm.get("her_dual_random_select_prob", None)
        self.her_dual_random_select_prob: float | None = (
            float(_rsp) if _rsp is not None else None
        )
        self.kl_beta_her: float = float(
            cfg.algorithm.get("kl_beta_her", self.kl_beta)
        )
        self.her_clip_ratio_low: float = float(
            cfg.algorithm.get("her_clip_ratio_low", cfg.algorithm.clip_ratio_low)
        )
        self.her_clip_ratio_high: float = float(
            cfg.algorithm.get("her_clip_ratio_high", cfg.algorithm.clip_ratio_high)
        )
        self.her_loss_clip: float | None = (
            float(cfg.algorithm.her_loss_clip)
            if cfg.algorithm.get("her_loss_clip", None) is not None
            else None
        )
        self._filter_metrics_orig: dict[str, float] = {}
        self._filter_metrics_her: dict[str, float] = {}
        self.her_rollout_batch: dict[str, Any] = {}

    # ── Rollout batch reception ──────────────────────────────────────────────

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Reshape tensors and reorder string fields (parent's preprocess).
        rollout_batch = self._preprocess_rollout_batch(rollout_batch)

        # 1. Duplicate the preprocessed batch so HER's in-place patches stay on the copy.
        her_rollout_batch = _clone_rollout_batch(rollout_batch)

        # 2. Relabel the duplicate only.
        her_result = self._her_processor(
            her_rollout_batch,
            force_all_groups=self.her_dual_force_all_groups,
            random_select_prob=self.her_dual_random_select_prob,
        )
        self._her_metrics = her_result["metrics"]

        # 3. Filter rewards on both batches independently.
        #    _postprocess_rollout_batch writes to self._rollout_postprocess_metrics
        #    as a side effect, so snapshot after each call.
        rollout_batch = self._postprocess_rollout_batch(rollout_batch)
        self._filter_metrics_orig = dict(self._rollout_postprocess_metrics)

        self.her_rollout_batch = self._postprocess_rollout_batch(her_rollout_batch)
        self._filter_metrics_her = dict(self._rollout_postprocess_metrics)

        # 4. Mask trajectories that did not go through the full HER pipeline
        #    (i.e., trajs in non-selected or fallback groups) from the HER loss.
        if self.cfg.algorithm.get("mask_her_nonrelabeled", False):
            eval_traj_indices = her_result.get("eval_traj_indices", [])
            self._apply_her_nonrelabeled_mask(self.her_rollout_batch, eval_traj_indices)

        return rollout_batch

    def _apply_her_nonrelabeled_mask(
        self,
        her_batch: dict[str, Any],
        eval_traj_indices: list[int],
    ) -> None:
        """Zero out loss_mask for trajs that did not go through the full HER pipeline.

        ``eval_traj_indices`` are the trajs the VLM was asked to score — i.e.,
        trajs in selected, non-fallback groups. This INCLUDES trajs the VLM
        ultimately rejected (vlm_reward == 0); it is not a "positive only" filter.
        """
        T, B, _ = her_batch["rewards"].shape
        loss_mask = her_batch.get("loss_mask")
        assert loss_mask is not None, (
            "_apply_her_nonrelabeled_mask requires loss_mask to be set upstream "
            "(enable filter_rewards or disable ignore_terminations)."
        )

        eval_set = set(eval_traj_indices)
        n_masked = 0
        for b in range(B):
            if b not in eval_set:
                loss_mask[:, b, :].fill_(0)
                n_masked += 1

        # Do not recompute loss_mask_sum. Masked trajs already contribute zero
        # in masked_mean_ratio via where(mask=False, 0, ...), so a stale per-traj
        # count is harmless. Recomputing from the current loss_mask would be wrong
        # under reward_type=chunk_level: upstream loss_mask_sum is summed over (T, A)
        # before chunk-level reduction (scale ~T*A, matching max_episode_steps),
        # while the post-reduction loss_mask is [T, B, 1] — re-summing it would
        # produce T-scale counts and break the loss_mask_ratio normalization.

        # Refresh filter_her/effective_tokens to reflect the post-mask state so
        # W&B reports the actual contributing-token count, not the pre-mask one.
        effective = int(loss_mask.sum().item())
        total = int(loss_mask.numel())
        self._filter_metrics_her["effective_tokens"] = effective
        self._filter_metrics_her["effective_tokens_frac"] = (
            effective / total if total > 0 else 0.0
        )

        print(
            f"[her] mask_her_nonrelabeled: masked {n_masked}/{B} trajs, "
            f"kept {len(eval_set)} VLM-evaluated (incl. vlm_reward=0)",
            flush=True,
        )

    # ── Advantages & returns ─────────────────────────────────────────────────

    def compute_advantages_and_returns(self) -> dict:
        # EmbodiedFSDPActor.compute_advantages_and_returns merges
        # self._rollout_postprocess_metrics into its return dict. Swap the right
        # snapshot in for each branch so unprefixed keys (e.g. effective_tokens)
        # match the batch they describe instead of leaking across branches.
        saved_filter_metrics = self._rollout_postprocess_metrics

        # Advantages for the original batch (uses self.rollout_batch).
        self._rollout_postprocess_metrics = self._filter_metrics_orig
        try:
            rollout_metrics: dict = dict(
                EmbodiedFSDPActor.compute_advantages_and_returns(self)
            )
        finally:
            self._rollout_postprocess_metrics = saved_filter_metrics
        for k, v in self._filter_metrics_orig.items():
            rollout_metrics[f"filter_orig/{k}"] = v

        # Advantages for the HER batch via temporary swap.
        saved_rollout_batch = self.rollout_batch
        self.rollout_batch = self.her_rollout_batch
        self._rollout_postprocess_metrics = self._filter_metrics_her
        try:
            her_metrics: dict = dict(
                EmbodiedFSDPActor.compute_advantages_and_returns(self)
            )
        finally:
            self.her_rollout_batch = self.rollout_batch
            self.rollout_batch = saved_rollout_batch
            self._rollout_postprocess_metrics = saved_filter_metrics

        for k, v in her_metrics.items():
            rollout_metrics[f"her/{k}"] = v
        for k, v in self._filter_metrics_her.items():
            rollout_metrics[f"filter_her/{k}"] = v
        for k, v in self._her_metrics.items():
            rollout_metrics[k] = v
        return rollout_metrics

    # ── Training ─────────────────────────────────────────────────────────────

    @Worker.timer("run_training")
    def run_training(self):
        """Dual-batch training loop: paired mini-batches, summed loss, one backward."""
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        self.model.train()

        # Shuffle both batches with the same permutation so paired samples stay aligned.
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
            self.her_rollout_batch = process_nested_dict_for_train(
                self.her_rollout_batch, shuffle_id
            )

        # Ref logprobs for both (no-op when kl_beta == 0).
        self._compute_ref_logprobs()
        self._compute_ref_logprobs_her()

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
            print(f"[Actor-HERDual] Update epoch {epoch_idx + 1}/{update_epoch}.")
            self._run_single_update_epoch_dual(metrics)

        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()

        mean_metric_dict = {key: np.nanmean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        return mean_metric_dict

    def _compute_ref_logprobs_her(self) -> None:
        """Run parent's ref-logprob computation against ``self.her_rollout_batch``."""
        if (
            max(self.kl_beta, self.kl_beta_her) <= 0
            or self.ref_policy_state_dict is None
        ):
            return
        saved = self.rollout_batch
        saved_kl_beta = self.kl_beta
        self.rollout_batch = self.her_rollout_batch
        self.kl_beta = self.kl_beta_her
        try:
            self._compute_ref_logprobs()
        finally:
            self.her_rollout_batch = self.rollout_batch
            self.rollout_batch = saved
            self.kl_beta = saved_kl_beta

    # ── Paired mini-batch / micro-batch loop ─────────────────────────────────

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

        for orig_global, her_global in zip(orig_global_iter, her_global_iter):
            size = orig_global["prev_logprobs"].shape[0]
            assert size == her_global["prev_logprobs"].shape[0]
            assert size % self.cfg.actor.micro_batch_size == 0
            micro_cnt = size // self.cfg.actor.micro_batch_size

            orig_micros = split_dict_to_chunk(orig_global, micro_cnt)
            her_micros = split_dict_to_chunk(her_global, micro_cnt)

            self.optimizer.zero_grad()
            for idx, (m_orig, m_her) in enumerate(zip(orig_micros, her_micros)):
                is_last_mb = (idx + 1) == self.gradient_accumulation

                m_orig = put_tensor_device(
                    m_orig,
                    f"{Worker.torch_device_type}:{int(os.environ['LOCAL_RANK'])}",
                )
                m_her = put_tensor_device(
                    m_her,
                    f"{Worker.torch_device_type}:{int(os.environ['LOCAL_RANK'])}",
                )

                # Original branch — backward under no_sync so orig activations
                # can be freed before the HER forward. Grads accumulate locally
                # in param.grad (FSDP reduce-scatter is deferred to the HER
                # backward below).
                no_sync_ctx = self.before_micro_batch(
                    self.model, is_last_micro_batch=False
                )
                orig_loss, orig_md = self._microbatch_forward_and_loss(
                    m_orig, kl_beta_override=self.kl_beta
                )
                scaled_orig = (
                    self.rl_loss_weight * orig_loss
                ) / self.gradient_accumulation
                with no_sync_ctx:
                    self.grad_scaler.scale(scaled_orig).backward()
                orig_loss_val = orig_loss.detach().item()
                del orig_loss, scaled_orig

                # HER branch — backward under the real last-mb context so the
                # final reduce-scatter fires exactly once per mini-batch,
                # capturing grads from both branches.
                her_ctx = self.before_micro_batch(
                    self.model, is_last_micro_batch=is_last_mb
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
                her_loss_val = her_loss.detach().item()
                del her_loss, scaled_her

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

            grad_norm, lr_list = self.optimizer_step()
            step_data = {
                "actor/grad_norm": grad_norm,
                "actor/lr": lr_list[0],
            }
            if len(lr_list) > 1:
                step_data["critic/lr"] = lr_list[1]
            append_to_dict(metrics, step_data)

    # ── Shared micro-batch forward + loss (no backward) ──────────────────────

    def _microbatch_forward_and_loss(
        self,
        batch: dict[str, torch.Tensor],
        kl_beta_override: float | None = None,
        clip_ratio_overrides: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Forward + policy loss + entropy + KL for one micro-batch.

        Mirrors the body of ``EmbodiedFSDPActor._run_single_update_epoch`` but
        skips the backward so the caller can combine losses across batches.

        ``kl_beta_override`` lets each branch pick its own KL coefficient
        (e.g. ``self.kl_beta_her`` for the HER branch). When ``None``, falls
        back to ``self.kl_beta``.

        ``clip_ratio_overrides`` lets each branch use its own clip bounds
        (keys: ``"high"``, ``"low"``). When ``None``, uses the global config
        values.
        """
        kl_beta = self.kl_beta if kl_beta_override is None else kl_beta_override
        advantages = batch["advantages"]
        prev_logprobs = batch["prev_logprobs"]
        returns = batch.get("returns", None)
        prev_values = batch.get("prev_values", None)
        loss_mask = batch.get("loss_mask", None)
        loss_mask_sum = batch.get("loss_mask_sum", None)
        forward_inputs = batch.get("forward_inputs", None)

        kwargs: dict = {}
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ]:
            kwargs["temperature"] = (
                self.cfg.algorithm.sampling_params.temperature_train
            )
            kwargs["top_k"] = self.cfg.algorithm.sampling_params.top_k
        elif SupportedModel(self.cfg.actor.model.model_type) == SupportedModel.GR00T:
            kwargs["prev_logprobs"] = prev_logprobs

        compute_values = self.cfg.algorithm.adv_type == "gae"

        with self.amp_context:
            output_dict = self.model(
                forward_inputs=forward_inputs,
                compute_logprobs=True,
                compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                compute_values=compute_values,
                use_cache=False,
                **kwargs,
            )

        if SupportedModel(self.cfg.actor.model.model_type) == SupportedModel.GR00T:
            prev_logprobs = output_dict["prev_logprobs"]

        ploss_kwargs = {
            "loss_type": self.cfg.algorithm.loss_type,
            "logprob_type": self.cfg.algorithm.logprob_type,
            "reward_type": self.cfg.algorithm.reward_type,
            "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
            "logprobs": output_dict["logprobs"],
            "values": output_dict.get("values", None),
            "old_logprobs": prev_logprobs,
            "advantages": advantages,
            "returns": returns,
            "prev_values": prev_values,
            "clip_ratio_high": clip_ratio_overrides["high"] if clip_ratio_overrides else self.cfg.algorithm.clip_ratio_high,
            "clip_ratio_low": clip_ratio_overrides["low"] if clip_ratio_overrides else self.cfg.algorithm.clip_ratio_low,
            "value_clip": self.cfg.algorithm.get("value_clip", None),
            "huber_delta": self.cfg.algorithm.get("huber_delta", None),
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
            "max_episode_steps": self.cfg.env.train.max_episode_steps,
            "task_type": self.cfg.runner.task_type,
            "critic_warmup": self.optimizer_steps < self.critic_warmup_steps,
        }
        loss, metrics_data = policy_loss(**ploss_kwargs)

        entropy_loss = torch.tensor(
            0.0, device=Worker.torch_platform.current_device()
        )
        if (
            self.cfg.algorithm.entropy_bonus > 0
            and not ploss_kwargs["critic_warmup"]
        ):
            entropy = output_dict["entropy"]
            entropy = reshape_entropy(
                entropy,
                entropy_type=self.cfg.algorithm.entropy_type,
                action_dim=self.cfg.actor.model.get("action_dim", 7),
                batch_size=output_dict["logprobs"].shape[0],
            )
            entropy_loss = masked_mean(entropy, mask=loss_mask)
            loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss
        metrics_data["actor/entropy_loss"] = entropy_loss.detach().item()

        kl_loss = torch.tensor(0.0, device=Worker.torch_platform.current_device())
        ref_logprobs = batch.get("ref_logprobs", None)
        if kl_beta > 0 and ref_logprobs is not None:
            kld = kl_penalty(
                output_dict["logprobs"], ref_logprobs, self.kl_penalty_type
            )
            bsz = kld.shape[0]
            action_dim = self.cfg.actor.model.get("action_dim", 7)
            logprob_type = self.cfg.algorithm.logprob_type
            if logprob_type == "token_level":
                kld = kld.reshape(bsz, -1, action_dim)
                kl_loss_mask = (
                    loss_mask.unsqueeze(-1) if loss_mask is not None else None
                )
            elif logprob_type == "action_level":
                kld = kld.reshape(bsz, -1, action_dim).sum(dim=-1)
                kl_loss_mask = loss_mask
            elif logprob_type == "chunk_level":
                kld = kld.reshape(bsz, -1, action_dim).sum(dim=[1, 2])
                kl_loss_mask = (
                    loss_mask.flatten() if loss_mask is not None else None
                )
            else:
                kl_loss_mask = loss_mask
            kl_loss = masked_mean(kld, mask=kl_loss_mask)
            loss = loss + kl_beta * kl_loss
        metrics_data["actor/kl_loss"] = kl_loss.detach().item()
        metrics_data["actor/kl_beta"] = float(kl_beta)

        return loss, metrics_data
