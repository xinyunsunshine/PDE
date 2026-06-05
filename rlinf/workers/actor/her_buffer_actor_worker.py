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

"""EmbodiedFSDPActorHERBuffer: shared CPU replay buffer for HER auxiliary losses.

This is the common base class for ``EmbodiedFSDPActorHERSFT`` and
``EmbodiedFSDPActorHERContrastive``.  It provides:

  1. A bounded CPU replay buffer (``deque``) that stores relabeled
     (forward_inputs, action_tokens) entries.
  2. Rollout batch preprocessing: runs HERProcessor in compute-only mode
     (``apply_to_batch=False``), retokenizes the subsampled trajectories,
     builds buffer entries, and appends them via an augmentation hook.
  3. A generic flat random sampler for buffer entries.

Subclasses add their own auxiliary loss (SFT or contrastive) on top of the
GRPO RL loss.
"""

import random
from abc import abstractmethod
from collections import deque
from typing import Optional

import torch

from rlinf.workers.actor.her_embodied_actor_worker import EmbodiedFSDPActorHER


class EmbodiedFSDPActorHERBuffer(EmbodiedFSDPActorHER):
    """EmbodiedFSDPActorHER extended with a CPU replay buffer for auxiliary losses.

    Subclasses must override ``_get_buffer_config`` to return the config dict
    containing ``replay_buffer_size`` and ``relabel_per_group``.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        buf_cfg = self._get_buffer_config(cfg)
        buffer_size = int(buf_cfg.get("replay_buffer_size", 8))
        self._her_relabel_per_group = int(buf_cfg.get("relabel_per_group", 1))
        self._her_replay_buffer: deque[dict] = deque(maxlen=buffer_size)

    @abstractmethod
    def _get_buffer_config(self, cfg) -> dict:
        """Return the config section with buffer settings.

        Subclasses override to point at their own namespace, e.g.
        ``cfg.actor.get("her_sft", {})``.
        """

    # ------------------------------------------------------------------
    # Rollout batch preprocessing — runs HERProcessor (compute-only),
    # retokenizes the subsampled set, and stores buffer entries.
    # ------------------------------------------------------------------

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        rollout_batch = self._preprocess_rollout_batch(rollout_batch)

        # Compute HER relabeling without modifying the batch.
        her_result = self._her_processor(rollout_batch, apply_to_batch=False)
        self._her_metrics = her_result["metrics"]

        traj_instructions = her_result["traj_instructions"]
        fallback_trajs = her_result["fallback_trajs"]

        # Only non-fallback trajectories with genuine relabeled instructions.
        relabeled_bs = sorted(b for b in traj_instructions if b not in fallback_trajs)

        # Subsample to relabel_per_group per GRPO group to limit buffer memory.
        if self._her_relabel_per_group > 0 and relabeled_bs:
            group_size = self.cfg.algorithm.get("group_size", 8)
            groups: dict[int, list[int]] = {}
            for b in relabeled_bs:
                groups.setdefault(b // group_size, []).append(b)
            selected: list[int] = []
            for g_bs in groups.values():
                selected.extend(
                    random.sample(g_bs, min(self._her_relabel_per_group, len(g_bs)))
                )
            relabeled_bs = sorted(selected)

        n_relabeled = len(relabeled_bs)
        if n_relabeled > 0:
            setup = her_result["setup"]

            # Retokenize only the subsampled trajectories.
            selected_instructions = {b: traj_instructions[b] for b in relabeled_bs}
            retokenized = self._her_processor.retokenize_prompts(
                rollout_batch, selected_instructions
            )
            buf = self._build_buffer_entry(
                rollout_batch,
                relabeled_bs,
                retokenized,
                traj_instructions,
                setup["is_openpi"],
            )

            # Build VLM inputs cache so augmentation hooks (e.g. contrastive
            # negative generation) have access to trajectory frames.
            vlm_inputs_cache: dict = {}
            for b in relabeled_bs:
                frames = self._her_processor._get_traj_frames(
                    setup["pixel_values"], b
                )
                vlm_inputs_cache[b] = (frames, setup["all_tasks"][b])

            buf = self._augment_relabeled_buf(
                buf, relabeled_bs, vlm_inputs_cache, traj_instructions
            )
            self._her_replay_buffer.append(buf)

        # Merge buffer stats into _her_metrics so they flow through
        # compute_advantages_and_returns → EmbodiedRunner metric logging.
        T = her_result["setup"]["T"]
        self._her_metrics["her_buffer_trajs_relabeled"] = n_relabeled
        self._her_metrics["her_buffer_trajs_stored"] = T * n_relabeled
        self._her_metrics["her_buffer_size"] = len(self._her_replay_buffer)

        if self._rank == 0 and n_relabeled > 0:
            self.log_info(
                f"[HER-Buffer] {n_relabeled} trajs → buffer, "
                f"{T * n_relabeled} samples stored. "
                f"Buffer size: {len(self._her_replay_buffer)}"
            )

        return self._postprocess_rollout_batch(rollout_batch)

    # ------------------------------------------------------------------
    # Buffer entry construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_buffer_entry(
        rollout_batch: dict,
        relabeled_bs: list[int],
        retokenized: dict,
        traj_instructions: dict[int, str],
        is_openpi: bool,
    ) -> dict:
        """Build a buffer entry from retokenized prompts and batch data.

        Retokenized tokens/masks are expanded across T timesteps and
        flattened together with observations/actions sliced from the
        original batch.
        """
        fi = rollout_batch["forward_inputs"]
        n_rel = len(relabeled_bs)
        tokens = retokenized["tokens"]  # [n_rel, seq_len]
        masks = retokenized["masks"]  # [n_rel, seq_len]

        # Determine T and the prompt key names based on model type.
        if is_openpi:
            T = fi["tokenized_prompt"].shape[0]
            token_key, mask_key = "tokenized_prompt", "tokenized_prompt_mask"
            extra_keys = ["chains", "denoise_inds"]
            extra_keys += [k for k in fi if k.startswith("observation/")]
        else:
            T = fi["input_ids"].shape[0]
            token_key, mask_key = "input_ids", "attention_mask"
            extra_keys = ["pixel_values", "action_tokens"]

        total = T * n_rel
        buf: dict = {
            token_key: tokens.unsqueeze(0)
            .expand(T, -1, -1)
            .reshape(total, -1)
            .cpu()
            .contiguous(),
            mask_key: masks.unsqueeze(0)
            .expand(T, -1, -1)
            .reshape(total, -1)
            .cpu()
            .contiguous(),
            "relabeled_instruction": [traj_instructions[b] for b in relabeled_bs] * T,
        }
        for key in extra_keys:
            v = fi.get(key)
            if v is None:
                continue
            v = v[:, relabeled_bs]
            buf[key] = v.reshape(total, *v.shape[2:]).cpu()

        return buf

    # ------------------------------------------------------------------
    # Buffer augmentation hook (overridden by subclasses, e.g. contrastive)
    # ------------------------------------------------------------------

    def _augment_relabeled_buf(
        self,
        buf: dict,
        relabeled_bs: list,
        vlm_inputs_cache: dict,
        traj_instructions: dict,
    ) -> dict:
        """Hook called just before a buffer entry is appended.

        Subclasses can override to add extra keys (e.g. negative instructions
        for contrastive training).  Base class is a no-op.
        """
        return buf

    # ------------------------------------------------------------------
    # Replay buffer sampling
    # ------------------------------------------------------------------

    def _sample_her_buffer_microbatch(self, micro_batch_size: int) -> Optional[dict]:
        """Randomly sample ``micro_batch_size`` items from the replay buffer.

        Returns a dict of CPU tensors ready to be moved to the GPU, or None if
        the buffer is empty.
        """
        if not self._her_replay_buffer:
            return None

        entries = list(self._her_replay_buffer)
        keys = list(entries[0].keys())

        # Concatenate all buffer entries per key.
        merged: dict = {}
        for k in keys:
            vals = [e[k] for e in entries]
            if isinstance(vals[0], torch.Tensor):
                merged[k] = torch.cat(vals, dim=0)
            else:
                merged[k] = sum(vals, [])

        size_key = "input_ids" if "input_ids" in merged else "tokenized_prompt"
        N = merged[size_key].shape[0]
        if N == 0:
            return None

        n_sample = min(micro_batch_size, N)
        indices = torch.randperm(N)[:n_sample]

        result: dict = {}
        for k in keys:
            if isinstance(merged[k], torch.Tensor):
                result[k] = merged[k][indices]
            else:
                idx_list = indices.tolist()
                result[k] = [merged[k][i] for i in idx_list]
        return result
