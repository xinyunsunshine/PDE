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

"""EmbodiedFSDPActorHERContrastive: joint RL + VLM-hard-negative contrastive training.

Overview
--------
Extends EmbodiedFSDPActorHERBuffer with a per-trajectory binary contrastive loss.

For each trajectory stored in the replay buffer, we have:
  - positive_instruction  — what the robot actually did (from HER VLM relabeling)
  - negative_instruction  — a plausible but wrong alternative (from a second VLM call)

The pairwise contrastive loss for trajectory i:

    L_i = -log [ exp(sim(pos_i, τ_i)/T) / (exp(sim(pos_i, τ_i)/T) + exp(sim(neg_i, τ_i)/T)) ]

where sim(l, τ_i) = mean_t log p_θ(a_t | l, o_i).

This trains the model to assign higher action likelihood to the correct instruction
than to the VLM-generated foil, without any cross-trajectory comparison — so there
are no false negatives regardless of batch composition.

The negative instructions are generated via ``_augment_relabeled_buf``, a hook in the
buffer base class called just before each buffer entry is appended.  The contrastive
backward runs after the RL backward so activations do not overlap in memory.

Training loop per microbatch:
  1. GRPO backward (on raw rollout buffer)
  2. Contrastive backward (on relabeled replay buffer with hard negatives)
"""

import os
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import torch
import torch.distributed as dist

from rlinf.algorithms.losses import pairwise_contrastive_loss
from rlinf.config import SupportedModel
from rlinf.scheduler import Worker
from rlinf.workers.actor.her_buffer_actor_worker import EmbodiedFSDPActorHERBuffer
from rlinf.workers.actor.her_utils import HER_NEGATIVE_PROMPT_TEMPLATE


class EmbodiedFSDPActorHERContrastive(EmbodiedFSDPActorHERBuffer):
    """EmbodiedFSDPActorHERBuffer extended with a VLM hard-negative contrastive loss.

    For each relabeled trajectory, a second VLM call generates a plausible but
    incorrect alternative task description (hard negative).  The pairwise
    contrastive loss then trains the model to rank the correct instruction above
    the foil for each trajectory independently — no cross-batch comparison, no
    false negatives.

    New config keys under ``cfg.actor.her_contrastive``:
      contrastive_batch_size (int, default 4): B trajectories sampled per
          contrastive step.  Sampling is trajectory-level: one timestep per
          unique trajectory.
      contrastive_loss_weight (float, default 0.1): weight λ for the loss.
      temperature (float, default 0.07): softmax temperature T.
      replay_buffer_size (int, default 4): forwarded to buffer base.
      filter_by_reward (bool, default False): skip low-reward trajs before VLM.
      reward_filter_threshold (float, default 0.0): reward cutoff for filtering.
      relabel_per_group (int, default 1): VLM calls per GRPO group.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        c_cfg = cfg.actor.get("her_contrastive", {})
        self._contrastive_batch_size: int = int(c_cfg.get("contrastive_batch_size", 4))
        self._contrastive_loss_weight: float = float(
            c_cfg.get("contrastive_loss_weight", 0.1)
        )
        self._contrastive_temperature: float = float(c_cfg.get("temperature", 0.07))

    def _get_buffer_config(self, cfg) -> dict:
        return cfg.actor.get("her_contrastive", {})

    # ------------------------------------------------------------------
    # VLM negative generation
    # ------------------------------------------------------------------

    def _request_negative_instruction(
        self,
        frame_tensors: list,
        original_instruction: str,
        positive_instruction: str,
    ) -> str:
        """Call VLM to generate a plausible but incorrect alternative task description.

        Uses the same VLM backend as positive HER relabeling.  Retries until the
        response parses correctly and differs from the positive instruction.
        """
        prompt_text = HER_NEGATIVE_PROMPT_TEMPLATE.format(
            original_instruction=original_instruction,
            positive_instruction=positive_instruction,
        )
        max_attempts = 10
        _attempt = 0
        while _attempt < max_attempts:
            _attempt += 1
            try:
                raw = self._her_processor._vlm.call(
                    frame_tensors, prompt_text, max_tokens=1024
                )
            except RuntimeError as e:
                self.log_warning(
                    f"[HER-Contrastive] neg attempt {_attempt}: "
                    f"VLM call failed ({e}), retrying"
                )
                continue
            # Strip <think>...</think> wrapper if present.
            cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            matched = re.search(r"<final>\s*(.*?)\s*</final>", cleaned, flags=re.DOTALL)
            if matched:
                instruction = matched.group(1).strip().rstrip(".,;:!?")
            else:
                # Fallback: the thinking model often returns the instruction as
                # the last short line after its reasoning. Try to extract it.
                lines = [ln.strip() for ln in cleaned.split("\n") if ln.strip()]
                if not lines:
                    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
                candidate = lines[-1] if lines else ""
                # Strip markdown bold/quotes.
                candidate = re.sub(r"[\*\"`]", "", candidate).strip().rstrip(".,;:!?")
                if candidate and 3 <= len(candidate.split()) <= 20:
                    instruction = candidate
                else:
                    if _attempt == 1:
                        self.log_warning(
                            f"[HER-Contrastive] neg attempt {_attempt}: no <final> tag. "
                            f"Raw response (first 500 chars): {raw[:500]!r}"
                        )
                    else:
                        self.log_warning(
                            f"[HER-Contrastive] neg attempt {_attempt}: "
                            "no <final> tag, retrying"
                        )
                    continue
            if instruction:
                instruction = instruction[0].upper() + instruction[1:]
            if len(instruction.split()) > 20:
                self.log_warning(
                    f"[HER-Contrastive] neg attempt {_attempt}: "
                    f"too long ({len(instruction.split())} words), retrying"
                )
                continue
            if instruction.lower() == positive_instruction.lower():
                self.log_warning(
                    f"[HER-Contrastive] neg attempt {_attempt}: "
                    "identical to positive, retrying"
                )
                continue
            return instruction
        raise RuntimeError(
            f"[HER-Contrastive] failed to generate negative instruction after "
            f"{max_attempts} attempts for positive={positive_instruction!r}"
        )

    # ------------------------------------------------------------------
    # Buffer augmentation hook (overrides parent no-op)
    # ------------------------------------------------------------------

    def _augment_relabeled_buf(
        self,
        buf: dict,
        relabeled_bs: list,
        vlm_inputs_cache: dict,
        traj_instructions: dict,
    ) -> dict:
        """Generate one hard negative per relabeled trajectory and add to buf.

        Runs VLM calls in parallel (same as positive relabeling) then stores
        ``negative_instruction`` in the buffer entry with the same flat layout
        as ``relabeled_instruction``: one string per (timestep, trajectory) pair.
        """

        def _get_neg(b: int) -> tuple[int, str]:
            frames, orig_instruction = vlm_inputs_cache[b]
            neg = self._request_negative_instruction(
                frames, orig_instruction, traj_instructions[b]
            )
            return b, neg

        neg_instructions: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=min(32, len(relabeled_bs) or 1)) as pool:
            futures = {pool.submit(_get_neg, b): b for b in relabeled_bs}
            for fut in as_completed(futures):
                b, neg = fut.result()
                neg_instructions[b] = neg

        # Infer T from the flat buffer size and the number of relabeled trajectories.
        n_flat = len(buf.get("relabeled_instruction", []))
        n_rel = len(relabeled_bs)
        T = n_flat // n_rel if n_rel > 0 else 1
        buf["negative_instruction"] = [neg_instructions[b] for b in relabeled_bs] * T

        if self._rank == 0:
            for b in relabeled_bs:
                self.log_info(
                    f"[HER-Contrastive] traj {b}  "
                    f"pos={traj_instructions[b]!r}  |  "
                    f"neg={neg_instructions[b]!r}"
                )
        return buf

    # ------------------------------------------------------------------
    # Trajectory-level sampling
    # ------------------------------------------------------------------

    def _sample_contrastive_minibatch(self, B: int) -> Optional[dict]:
        """Sample B items, one per unique trajectory, from buffer entries that
        have ``negative_instruction``.

        Groups all flat items by their (positive_instruction, negative_instruction)
        pair — which uniquely identifies a trajectory — then picks one random
        timestep per trajectory and returns up to B trajectories.
        """
        entries = [e for e in self._her_replay_buffer if "negative_instruction" in e]
        if not entries:
            return None

        # Group items by trajectory key = (pos_instr, neg_instr).
        groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
        for entry_i, entry in enumerate(entries):
            pos_list: list[str] = entry["relabeled_instruction"]
            neg_list: list[str] = entry["negative_instruction"]
            for local_idx in range(len(pos_list)):
                key = (pos_list[local_idx], neg_list[local_idx])
                groups[key].append((entry_i, local_idx))

        if not groups:
            return None

        # Shuffle trajectories and keep up to B.
        traj_keys = list(groups.keys())
        random.shuffle(traj_keys)
        selected_keys = traj_keys[:B]

        # For each selected trajectory pick one random timestep.
        all_keys = list(entries[0].keys())
        result: dict[str, list] = {k: [] for k in all_keys}
        for key in selected_keys:
            entry_i, local_idx = random.choice(groups[key])
            entry = entries[entry_i]
            for k in all_keys:
                result[k].append(entry[k][local_idx])

        stacked: dict = {}
        for k, v in result.items():
            if isinstance(v[0], torch.Tensor):
                stacked[k] = torch.stack(v, dim=0)
            else:
                stacked[k] = v  # list of strings
        return stacked

    # ------------------------------------------------------------------
    # Pairwise score computation (2 forward passes)
    # ------------------------------------------------------------------

    def _retokenize_instructions_for_item(
        self,
        instructions: list[str],
        ref_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize K instructions into the OpenVLA-OFT prompt format.

        Returns:
            new_input_ids:      [K, ref_seq_len] (CPU)
            new_attention_mask: [K, ref_seq_len] (CPU)
        """
        tokenizer = self._her_processor._get_tokenizer()
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        bos_id = tokenizer.bos_token_id

        prompts = [
            f"In: What action should the robot take to {instr.lower()}?\nOut: "
            for instr in instructions
        ]
        orig_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        encoded = tokenizer(
            prompts,
            add_special_tokens=True,
            truncation=True,
            max_length=ref_seq_len,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        tokenizer.padding_side = orig_side

        ids_t = encoded["input_ids"]  # [K, ref_seq_len]
        mask_t = encoded["attention_mask"]
        # Fix BOS placement after left-padding (same as her_sft_actor_worker).
        first_nonpad = mask_t.to(torch.int64).argmax(dim=1, keepdim=True)
        ids_t.scatter_(1, first_nonpad, pad_id)
        mask_t.scatter_(1, first_nonpad, 0)
        ids_t[:, 0] = bos_id
        mask_t[:, 0] = 1
        return ids_t, mask_t

    def _compute_pairwise_scores(
        self,
        sampled: dict,
        device: str,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-trajectory scores under positive and negative instructions.

        Runs two batched forward passes — one with positive instructions, one
        with negative — each of batch size B.

        Args:
            sampled: dict with stacked CPU tensors [B, ...] and instruction lists.
            device: target device string.
            kwargs: extra model kwargs (temperature, top_k).

        Returns:
            scores_pos: [B] mean action log-prob under positive instruction.
            scores_neg: [B] mean action log-prob under negative instruction.
        """
        pos_instructions: list[str] = sampled["relabeled_instruction"]
        neg_instructions: list[str] = sampled["negative_instruction"]
        B = len(pos_instructions)

        is_openpi = "tokenized_prompt" in sampled

        if is_openpi:
            scores_pos = self._compute_openpi_scores(
                sampled, pos_instructions, B, device, kwargs
            )
            scores_neg = self._compute_openpi_scores(
                sampled, neg_instructions, B, device, kwargs
            )
        else:
            scores_pos = self._compute_openvla_scores(
                sampled, pos_instructions, B, device, kwargs
            )
            scores_neg = self._compute_openvla_scores(
                sampled, neg_instructions, B, device, kwargs
            )
        return scores_pos, scores_neg

    def _compute_openvla_scores(
        self,
        sampled: dict,
        instructions: list[str],
        B: int,
        device: str,
        kwargs: dict,
    ) -> torch.Tensor:
        """Score instructions against trajectories for OpenVLA / OpenVLA-OFT."""
        seq_len = sampled["input_ids"].shape[-1]
        ids, mask = self._retokenize_instructions_for_item(instructions, seq_len)

        fwd = {
            "input_ids": ids.to(device),
            "attention_mask": mask.to(device),
            "pixel_values": sampled["pixel_values"].to(device),
            "action_tokens": sampled["action_tokens"].to(device),
        }
        with self.amp_context:
            out = self.model(
                forward_inputs=fwd,
                compute_logprobs=True,
                compute_entropy=False,
                compute_values=False,
                use_cache=False,
                **kwargs,
            )
        lp: torch.Tensor = out["logprobs"]  # [B, action_dim*chunks]
        return lp.view(B, -1).mean(dim=-1)  # [B]

    def _compute_openpi_scores(
        self,
        sampled: dict,
        instructions: list[str],
        B: int,
        device: str,
        kwargs: dict,
    ) -> torch.Tensor:
        """Score instructions against trajectories for OpenPI (pi05)."""
        model = self.model.module if hasattr(self.model, "module") else self.model
        ref_inputs = {"tokenized_prompt": sampled["tokenized_prompt"].to(device)}
        for key in (
            "observation/image",
            "observation/state",
            "observation/wrist_image",
        ):
            if key in sampled:
                ref_inputs[key] = sampled[key].to(device)

        new_tp, new_tpm = model.retokenize_prompts_for_second_pass(
            instructions, ref_inputs
        )

        fwd: dict = {
            "tokenized_prompt": new_tp,
            "tokenized_prompt_mask": new_tpm,
        }
        for key in sampled:
            if key.startswith("observation/") or key in ("chains", "denoise_inds"):
                fwd[key] = sampled[key].to(device)

        with self.amp_context:
            out = self.model(
                forward_inputs=fwd,
                compute_logprobs=True,
                compute_entropy=False,
                compute_values=False,
                use_cache=False,
                **kwargs,
            )
        lp: torch.Tensor = out["logprobs"]  # [B, ...]
        return lp.view(B, -1).mean(dim=-1)  # [B]

    # ------------------------------------------------------------------
    # Training loop overrides
    # ------------------------------------------------------------------

    def _run_microbatch_backward(
        self,
        batch: dict,
        loss: torch.Tensor,
        backward_ctx,
        idx: int,
        metrics_data: dict,
    ) -> float:
        """RL backward + contrastive backward."""
        rl_loss = super()._run_microbatch_backward(
            batch, loss, backward_ctx, idx, metrics_data
        )
        is_last = (idx + 1) == self.gradient_accumulation
        contrastive_loss = self._her_contrastive_backward(batch, metrics_data, is_last)
        return rl_loss + (contrastive_loss if contrastive_loss is not None else 0.0)

    def _her_contrastive_backward(
        self,
        batch: dict,
        metrics_data: dict,
        is_last_micro_batch: bool,
    ) -> Optional[float]:
        """Run a separate contrastive forward+backward after the RL backward.

        Because this runs after ``rl_loss.backward()`` has returned, RL
        activations have already been freed.  Only contrastive activations are
        live during the contrastive backward, halving peak GPU memory vs.
        combining both losses into one backward pass.
        """
        device = f"{Worker.torch_device_type}:{int(os.environ['LOCAL_RANK'])}"

        # All FSDP ranks must agree on whether to run the contrastive forward.
        # Disagreement (e.g. some ranks missed a VLM call) causes an AllGather hang.
        entries_with_neg = [
            e for e in self._her_replay_buffer if "negative_instruction" in e
        ]
        local_ready = int(len(entries_with_neg) >= 1)
        flag = torch.tensor(local_ready, dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        if flag.item() == 0:
            return None

        sampled_cpu = self._sample_contrastive_minibatch(self._contrastive_batch_size)
        if sampled_cpu is None:
            return None

        actual_B = len(sampled_cpu["relabeled_instruction"])

        kwargs = {}
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ]:
            kwargs = {
                "temperature": self.cfg.algorithm.sampling_params.temperature_train,
                "top_k": self.cfg.algorithm.sampling_params.top_k,
            }

        scores_pos, scores_neg = self._compute_pairwise_scores(
            sampled_cpu, device, kwargs
        )
        loss, c_metrics = pairwise_contrastive_loss(
            scores_pos,
            scores_neg,
            temperature=self._contrastive_temperature,
        )

        loss_val = loss.detach().item()
        weighted_val = self._contrastive_loss_weight * loss_val
        if loss_val > 10.0:
            self.log_warning(
                f"[HER-Contrastive] loss={loss_val:.4f} "
                f"(weighted: {weighted_val:.4f}) is unusually large."
            )
        else:
            self.log_info(
                f"[HER-Contrastive] loss={loss_val:.4f} "
                f"acc={c_metrics['contrastive/accuracy']:.3f} "
                f"gap={c_metrics['contrastive/score_gap']:.4f} "
                f"B={actual_B}"
            )

        # Print to stdout for job-log visibility.
        print(
            f"[HER-Contrastive] contrastive loss={loss_val:.4f} "
            f"weighted={weighted_val:.4f} "
            f"acc={c_metrics['contrastive/accuracy']:.3f} "
            f"B={actual_B}"
        )

        metrics_data["contrastive/loss"] = loss_val
        metrics_data["contrastive/buffer_size"] = float(len(entries_with_neg))
        metrics_data["contrastive/num_trajectories"] = float(actual_B)
        metrics_data.update(c_metrics)

        scaled_loss = self._contrastive_loss_weight * loss / self.gradient_accumulation
        backward_ctx = self.before_micro_batch(self.model, is_last_micro_batch)
        with backward_ctx:
            self.grad_scaler.scale(scaled_loss).backward()
        return scaled_loss.detach().item()
