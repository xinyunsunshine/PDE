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

"""HERProcessor: applies HER relabeling to a rollout batch."""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from rlinf.workers.actor.her_utils import (
    HER_VLM_MAX_FRAMES,
    _save_trajectory_video,
    _to_vlm_frame_uint8,
    select_frame_indices_for_vlm,
    select_frames_for_vlm,
)
from rlinf.workers.actor.her_vlm_client import HERVLMClient


class HERProcessor:
    """Applies Hindsight Experience Replay to a rollout batch.

    Orchestrates the full HER pipeline:
      1. ``_setup_her_batch``             — extract per-trajectory data from the batch
      2. ``_select_groups``               — stochastically choose groups to relabel
      3. ``_generate_group_instructions`` — query VLM for hindsight instruction per group
      4. ``_resolve_nothing_fallbacks``   — handle groups where VLM said "Nothing"
      5. ``_expand_to_traj_instructions`` — broadcast group instruction to each trajectory
      6. ``_run_parallel_reward_eval``    — query VLM for binary reward per trajectory
      7. ``_patch_rewards``               — zero original rewards, write VLM reward
      8. ``_patch_subtraj_terminations``  — shorten dones/loss_mask for subtraj cutoffs
      9. ``_patch_prompts``               — re-encode input_ids with new instructions

    Usage::

        processor = HERProcessor(cfg, rank, get_model_fn, log_info, log_warning)
        rollout_batch, metrics = processor(rollout_batch)
        processor.version = global_step
    """

    def __init__(
        self,
        cfg: DictConfig,
        rank: int,
        get_model_fn: Callable[[], Any],
        log_warning_fn: Callable[[str], None],
    ) -> None:
        self.cfg = cfg
        self._rank = rank
        self._get_model_fn = get_model_fn
        self.version = 0

        alg = cfg.algorithm
        her_endpoint = alg.get("her_endpoint", "") or alg.get(
            "instruction_relabel_endpoint", ""
        )
        her_endpoint_file = alg.get("her_endpoint_file", "") or None
        her_model = alg.get("her_model", "") or alg.get("instruction_relabel_model", "")
        her_video_mode = alg.get("her_video_mode", "") or alg.get(
            "instruction_relabel_backend", "mp4"
        )
        her_api_key = alg.get("her_api_key", "") or (
            os.environ.get("LITELLM_API_KEY", "")
            if her_video_mode == "anthropic"
            else os.environ.get("OPENAI_API_KEY", "EMPTY")
        )

        self.her_endpoint = her_endpoint
        self.her_endpoint_file = her_endpoint_file
        self._vlm = HERVLMClient(
            endpoint=her_endpoint,
            endpoint_file=her_endpoint_file,
            model=her_model,
            video_mode=her_video_mode,
            api_key=her_api_key,
            max_frames=alg.get("her_vlm_max_frames", HER_VLM_MAX_FRAMES),
            flip_horizontal=alg.get("flip_video_horizontal", False),
            reward_eval_prompt_version=alg.get("her_reward_eval_prompt_version", 1),
            log_warning_fn=log_warning_fn,
        )
        self._flip_video_horizontal = alg.get("flip_video_horizontal", False)
        self._reward_eval_num_samples = alg.get("her_reward_eval_num_samples", 1)
        self._reward_eval_vote_temperature = alg.get(
            "her_reward_eval_vote_temperature", 0.6
        )
        self._her_prompt_version = alg.get("her_prompt_version", "v1")
        self._her_log_relabeled_trajs = alg.get("her_log_relabeled_trajs", False)
        self._her_dump_rollout_batch = alg.get("her_dump_rollout_batch", False)
        self._her_dump_relabeled_text = alg.get("her_dump_relabeled_text", False)
        self._her_dump_traj_frames = alg.get("her_dump_traj_frames", False)
        self._her_dump_traj_frames_n = alg.get("her_dump_traj_frames_n", 16)
        self._her_prompt_debug_log_n = alg.get("her_prompt_debug_log_n", 0)
        self._log_max_groups = alg.get("log_max_groups", 2)
        _base_log_path = cfg.runner.logger.get("log_path", "./logs")
        _exp_name = cfg.runner.logger.get("experiment_name", "experiment")
        self._her_prompt_debug_log_dir = os.path.join(
            _base_log_path, _exp_name, "her_prompt_debug"
        )
        _dump_base = alg.get("her_dump_base_path", "/PROJECT_ROOT/rollout_dumps")
        self._rollout_dump_dir = os.path.join(_dump_base, _exp_name)
        _text_dump_base = alg.get(
            "her_relabeled_text_base_path",
            "/PROJECT_ROOT/relabeled_text_dumps",
        )
        self._relabeled_text_dump_dir = os.path.join(_text_dump_base, _exp_name)
        self._traj_frames_dump_dir = os.path.join(_text_dump_base, f"{_exp_name}_frames")

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _is_openpi_forward_inputs(rollout_batch: dict) -> bool:
        return "tokenized_prompt" in rollout_batch.get("forward_inputs", {})

    @staticmethod
    def _is_groot_forward_inputs(rollout_batch: dict) -> bool:
        return "eagle_input_ids" in rollout_batch.get("forward_inputs", {})

    @staticmethod
    def _extract_task_from_prompt(decoded_prompt: str) -> str:
        m = re.search(r"What action should the robot take to (.+?)\?", decoded_prompt)
        return m.group(1).strip() if m else decoded_prompt

    # ── Tokenizer / model helpers ─────────────────────────────────────────────

    def _get_tokenizer(self) -> Optional[Any]:
        model = self._get_model_fn()
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
        inner = model.module if hasattr(model, "module") else model
        if hasattr(inner, "input_processor") and hasattr(
            inner.input_processor, "tokenizer"
        ):
            return inner.input_processor.tokenizer
        return None

    # ── Frame helpers ─────────────────────────────────────────────────────────

    def _get_traj_frames(
        self, pixel_values: torch.Tensor, b: int
    ) -> list[torch.Tensor]:
        """Extract all frames for trajectory b from pixel_values.

        pixel_values: [T, B, ...]  — time-major, batch second
        Returns a list of T frame tensors for trajectory b.
        """
        return [pixel_values[t, b] for t in range(pixel_values.shape[0])]

    def _get_traj_frames_with_cutoff(
        self, pixel_values: torch.Tensor, b: int, cutoff: Optional[float]
    ) -> list[torch.Tensor]:
        """Like _get_traj_frames but truncate to the first ``cutoff`` fraction."""
        frames = self._get_traj_frames(pixel_values, b)
        if cutoff is not None and cutoff < 1.0:
            n_keep = max(1, int(round(len(frames) * cutoff)))
            frames = frames[:n_keep]
        return frames

    # ── Batch setup ───────────────────────────────────────────────────────────

    def _setup_her_batch(self, rollout_batch: dict[str, Any]) -> dict[str, Any]:
        """Extract per-trajectory metadata needed by all HER steps.

        Tensor shapes:
          pixel_values: [T, B, ...]      (camera observations; required)
        """
        is_openpi = self._is_openpi_forward_inputs(rollout_batch)
        is_groot = not is_openpi and self._is_groot_forward_inputs(rollout_batch)
        tokenizer = None
        if not is_openpi and not is_groot:
            tokenizer = self._get_tokenizer()
            if tokenizer is None:
                raise RuntimeError("[her] No tokenizer available")

        forward_inputs = rollout_batch["forward_inputs"]
        pixel_values = forward_inputs.get("observation/image")
        if pixel_values is None:
            pixel_values = forward_inputs.get("pixel_values")
        if pixel_values is None:
            pixel_values = forward_inputs.get("her_raw_frames")
        if pixel_values is None:
            raise RuntimeError(
                "[her] No pixel_values found in forward_inputs. "
                "Expected key 'observation/image' (OpenPI), 'pixel_values' (OpenVLA/OFT), "
                "or 'her_raw_frames' (GR00T)."
            )

        input_ids = None
        seq_len = None
        pad_id = None

        if is_openpi:
            # tokenized_prompt: [T, B, seq_len]
            T, B = forward_inputs["tokenized_prompt"].shape[:2]
            task_descriptions = rollout_batch.get("task_descriptions")
            if task_descriptions is None or len(task_descriptions) < B:
                raise RuntimeError(
                    f"[her] Missing task_descriptions for OpenPI rollout "
                    f"(got {len(task_descriptions) if task_descriptions else 0}, "
                    f"expected {B})"
                )
            all_tasks = [str(task_descriptions[b]).strip() for b in range(B)]
        elif is_groot:
            # eagle_input_ids: [T, B, seq_len]
            T, B, seq_len = forward_inputs["eagle_input_ids"].shape
            task_descriptions = rollout_batch.get("task_descriptions")
            if task_descriptions is None or len(task_descriptions) < B:
                raise RuntimeError(
                    f"[her] Missing task_descriptions for GR00T rollout "
                    f"(got {len(task_descriptions) if task_descriptions else 0}, "
                    f"expected {B})"
                )
            all_tasks = [str(task_descriptions[b]).strip() for b in range(B)]
        else:
            input_ids = forward_inputs["input_ids"]  # [T, B, seq_len]
            T, B, seq_len = input_ids.shape
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            all_tasks = [
                self._extract_task_from_prompt(
                    tokenizer.decode(input_ids[0, b], skip_special_tokens=True).strip()
                )
                for b in range(B)
            ]

        group_size = self.cfg.algorithm.get("group_size", 8)
        assert B % group_size == 0, (
            f"batch {B} not divisible by group_size {group_size}"
        )
        n_groups = B // group_size

        return {
            "is_openpi": is_openpi,
            "is_groot": is_groot,
            "tokenizer": tokenizer,
            "pixel_values": pixel_values,
            "T": T,
            "B": B,
            "seq_len": seq_len,
            "pad_id": pad_id,
            "all_tasks": all_tasks,
            "all_scene_objects": rollout_batch.get("scene_objects"),
            "group_size": group_size,
            "n_groups": n_groups,
        }

    # ── Group selection ───────────────────────────────────────────────────────

    def _filter_groups(
        self,
        per_traj_reward: torch.Tensor,  # [B]
        group_size: int,
        n_groups: int,
    ) -> list[int]:
        """Stochastically select group indices for HER processing.

        Pure selection gate — no anchor picking.

        Returns:
            selected_groups: list of group indices chosen for processing
        """
        import random as _rng

        alg = self.cfg.algorithm
        selection_mode = alg.get("her_selection", "asymmetric")
        std_epsilon = alg.get("her_std_epsilon", 0.1)
        low_threshold = alg.get("her_low_threshold", 0.5)
        high_threshold = alg.get("her_high_threshold", 0.5)

        selected_groups: list[int] = []

        for g in range(n_groups):
            group_start = g * group_size
            group_rewards = per_traj_reward[group_start : group_start + group_size]

            if selection_mode == "symmetric":
                p = std_epsilon / (group_rewards.std(correction=0).item() + std_epsilon)
                if _rng.random() >= p:
                    continue
            elif selection_mode == "threshold":
                if low_threshold <= group_rewards.mean().item() <= high_threshold:
                    continue
            else:  # asymmetric (default)
                n_failed = int((group_rewards == 0).sum().item())
                if _rng.random() >= n_failed / group_size:
                    continue

            selected_groups.append(g)

        if selection_mode == "threshold":
            detail = (
                f"selection={selection_mode}(low={low_threshold},high={high_threshold})"
            )
        else:
            detail = f"selection={selection_mode}"
        print(
            f"[her] {len(selected_groups)}/{n_groups} groups selected ({detail})",
            flush=True,
        )
        return selected_groups

    def _pick_anchors(
        self,
        selected_groups: list[int],
        per_traj_reward: torch.Tensor,  # [B]
        group_size: int,
    ) -> tuple[list[int], list[int]]:
        """Pick one anchor trajectory per selected group for VLM instruction generation.

        Default behavior: prefer a failed trajectory; fall back to any trajectory
        if ``her_anchor_from_success`` is set and the group has no failures.
        Groups with no valid anchor are dropped.

        If ``her_anchor_random`` is set, ignore failure status entirely and pick
        an anchor uniformly at random from all trajectories in the group. Every
        selected group keeps an anchor.

        Returns:
            anchored_groups: subset of selected_groups that have an anchor
            group_anchors:   parallel list of trajectory indices
        """
        import random as _rng

        anchor_from_success = self.cfg.algorithm.get("her_anchor_from_success", False)
        anchor_random = self.cfg.algorithm.get("her_anchor_random", False)

        anchored_groups: list[int] = []
        group_anchors: list[int] = []

        for g in selected_groups:
            group_start = g * group_size
            if anchor_random:
                anchored_groups.append(g)
                group_anchors.append(
                    _rng.choice(range(group_start, group_start + group_size))
                )
                continue
            n_failed = int(
                (
                    per_traj_reward[group_start : group_start + group_size] == 0
                )
                .sum()
                .item()
            )
            if n_failed > 0:
                failed_idxs = [
                    b
                    for b in range(group_start, group_start + group_size)
                    if per_traj_reward[b].item() == 0
                ]
                anchored_groups.append(g)
                group_anchors.append(_rng.choice(failed_idxs))
            elif anchor_from_success:
                anchored_groups.append(g)
                group_anchors.append(
                    _rng.choice(range(group_start, group_start + group_size))
                )

        print(
            f"[her] {len(anchored_groups)}/{len(selected_groups)} groups anchored",
            flush=True,
        )
        return anchored_groups, group_anchors

    def _select_groups(
        self,
        per_traj_reward: torch.Tensor,  # [B]
        group_size: int,
        n_groups: int,
    ) -> tuple[list[int], list[int]]:
        """Stochastically select groups and pick anchors for HER relabeling.

        Returns:
            selected_groups: list of group indices chosen for relabeling
            group_anchors:   list of trajectory indices (one per selected group)
                             used to generate the hindsight instruction
        """
        selected = self._filter_groups(per_traj_reward, group_size, n_groups)
        return self._pick_anchors(selected, per_traj_reward, group_size)

    # ── Instruction generation ────────────────────────────────────────────────

    def _generate_group_instructions(
        self,
        selected_groups: list[int],
        group_anchors: list[int],
        setup: dict[str, Any],
        debug_groups: set[int],
    ) -> tuple[dict[int, str], dict[int, float], int]:
        """Query the VLM in parallel for a hindsight instruction per selected group.

        Returns:
            group_instructions: {g -> instruction_str}
            group_cutoffs:      {g -> cutoff_fraction}
                                 present only for subtraj prompt responses (Form B)
            n_failures:         always 0 (kept for interface compatibility)
        """
        pixel_values = setup["pixel_values"]
        all_tasks = setup["all_tasks"]
        all_scene_objects = setup.get("all_scene_objects")

        # Guard: v4/v5 prompts embed object names from the scene; warn if missing.
        if self._her_prompt_version in ("v4", "v5", "v7", "v7_no_uninteresting", "v10", "v11", "v11_strict", "v12") and all_scene_objects is None:
            print(
                f"[her] WARNING: her_prompt_version={self._her_prompt_version} requires "
                "scene_objects from env, but got None. "
                "Ensure env populates 'episode_scene_objects'.",
                flush=True,
            )

        # Per-group VLM query closure.
        # Extracts the anchor trajectory's frames, sends them to the VLM with the
        # original instruction as context, and optionally saves a debug log.
        # API-level retries are handled by request_hindsight_instruction via @backoff.
        # Returns (group_idx, instruction, cutoff_fraction).
        def _query_one(g: int, anchor_b: int) -> tuple[int, str, Optional[float]]:
            frames = self._get_traj_frames(pixel_values, anchor_b)
            _raw_out: Optional[dict] = {} if g in debug_groups else None
            instr, cutoff = self._vlm.request_hindsight_instruction(
                frames,
                original_instruction=all_tasks[anchor_b],
                scene_objects=(
                    all_scene_objects[anchor_b]
                    if all_scene_objects is not None
                    else None
                ),
                prompt_version=self._her_prompt_version,
                _raw_out=_raw_out,
            )
            if _raw_out is not None:
                self._save_her_prompt_debug(
                    frames,
                    _raw_out["prompt_text"],
                    _raw_out["response"],
                    all_tasks[anchor_b],
                    g,
                    cutoff_fraction=cutoff,
                )
            return g, instr.rstrip(".,;:!?"), cutoff

        # Dispatch all groups in parallel and collect results.
        # group_cutoffs[g] is only populated for subtraj (Form B) responses.
        group_instructions: dict[int, str] = {}
        group_cutoffs: dict[int, float] = {}

        if not selected_groups:
            return group_instructions, group_cutoffs, 0

        with ThreadPoolExecutor(max_workers=min(16, len(selected_groups))) as pool:
            futures = {
                pool.submit(_query_one, g, group_anchors[i]): g
                for i, g in enumerate(selected_groups)
            }
            for fut in as_completed(futures):
                g, instr, cutoff = fut.result()
                group_instructions[g] = instr
                if cutoff is not None:
                    group_cutoffs[g] = cutoff

        return group_instructions, group_cutoffs, 0

    # ── "Nothing" fallback resolution ─────────────────────────────────────────

    def _resolve_nothing_fallbacks(
        self,
        selected_groups: list[int],
        group_anchors: list[int],
        group_instructions: dict[int, str],
        all_tasks: list[str],
    ) -> tuple[set[int], int, dict[str, float]]:
        """Replace "Nothing" instructions with the original task instruction.

        When the VLM concludes the robot did nothing interesting, we fall back
        to the original instruction so the trajectory still contributes to
        training (but does not get a new reward signal).

        Returns:
            fallback_groups:        set of group indices where fallback was applied
            n_filtered_uninteresting: count of such groups
            per_instruction_uninteresting_rate: {instruction: rate} for each task
        """
        fallback_groups: set[int] = set()
        n_filtered = 0
        uninteresting_originals: list[str] = []
        per_instruction_uninteresting: dict[str, int] = {}
        per_instruction_total: dict[str, int] = {}

        for i, g in enumerate(selected_groups):
            orig_task = all_tasks[group_anchors[i]]
            per_instruction_total[orig_task] = (
                per_instruction_total.get(orig_task, 0) + 1
            )
            instr = group_instructions.get(g)
            if instr is not None and instr.strip().lower() == "nothing":
                group_instructions[g] = orig_task
                fallback_groups.add(g)
                uninteresting_originals.append(orig_task)
                per_instruction_uninteresting[orig_task] = (
                    per_instruction_uninteresting.get(orig_task, 0) + 1
                )
                n_filtered += 1

        if uninteresting_originals:
            print(
                f"[her] {n_filtered}/{len(selected_groups)} groups filtered as "
                f"uninteresting (VLM said 'Nothing'). "
                f"Sample originals: {uninteresting_originals[:5]}",
                flush=True,
            )

        per_instruction_uninteresting_rate: dict[str, float] = {
            task: per_instruction_uninteresting.get(task, 0) / total
            for task, total in per_instruction_total.items()
        }

        return fallback_groups, n_filtered, per_instruction_uninteresting_rate

    # ── Group → trajectory expansion ─────────────────────────────────────────

    def _expand_to_traj_instructions(
        self,
        selected_groups: list[int],
        group_instructions: dict[int, str],
        group_cutoffs: dict[int, float],
        group_size: int,
        fallback_groups: set[int],
    ) -> tuple[dict[int, str], dict[int, float], set[int]]:
        """Broadcast each group's instruction to all trajectories in that group.

        Returns:
            traj_instructions: {b -> instruction_str}   — for all B in selected groups
            traj_cutoffs_map:  {b -> cutoff_fraction}   — only for subtraj cutoff trajs
            fallback_trajs:    set of trajectory indices whose group fell back to original
        """
        traj_instructions: dict[int, str] = {}
        traj_cutoffs_map: dict[int, float] = {}
        fallback_trajs: set[int] = set()

        # One VLM call was made per group (using the anchor trajectory).
        # Broadcast the result to every trajectory b in that group.
        # Groups with a subtraj cutoff get the same cutoff applied to all members.
        for g in selected_groups:
            instr = group_instructions[g]
            cutoff = group_cutoffs.get(g)
            for b in range(g * group_size, (g + 1) * group_size):
                traj_instructions[b] = instr
                if cutoff is not None:
                    traj_cutoffs_map[b] = cutoff
                if g in fallback_groups:
                    fallback_trajs.add(b)

        return traj_instructions, traj_cutoffs_map, fallback_trajs

    # ── Perturbed instruction detection ───────────────────────────────────────

    # ── Reward evaluation ─────────────────────────────────────────────────────

    def _run_parallel_reward_eval(
        self,
        trajs: list[int],
        pixel_values: torch.Tensor,
        instructions: dict[int, str],
        log_prefix: str,
        traj_cutoffs: Optional[dict[int, float]] = None,
    ) -> tuple[dict[int, float], dict[int, float], int]:
        """Query the VLM for a binary reward for each trajectory in parallel.

        pixel_values: [T, B, ...]  — only traj b's frames are extracted per call

        Returns:
            vlm_rewards:      {b -> float}   1.0 = success, 0.0 = failure / error
            eval_cutoffs:     {b -> float}   cutoffs from reward eval (v3 only, failed trajs)
                              fractions are relative to the raw trajectory; composed with
                              any prior instruction-gen cutoff already in traj_cutoffs
            n_failures:       number of trajectories where the VLM call raised an exception
        """
        vlm_rewards: dict[int, float] = {}
        eval_cutoffs: dict[int, float] = {}
        n_failures = 0
        if not trajs:
            return vlm_rewards, eval_cutoffs, n_failures

        # Per-trajectory VLM query closure.
        # Truncates frames to the subtraj cutoff fraction (if any) before sending.
        # Returns (traj_idx, reward, eval_cutoff_fraction).
        # eval_cutoff is relative to the frames passed in (already truncated by
        # the instruction-gen cutoff); we compose it back to raw trajectory fraction.
        def _eval_one(b: int) -> tuple[int, float, Optional[float]]:
            instr_cutoff = traj_cutoffs.get(b) if traj_cutoffs else None
            frames = self._get_traj_frames_with_cutoff(pixel_values, b, instr_cutoff)
            reward, eval_cutoff = self._vlm.request_reward_eval(
                frames,
                instructions[b],
                num_samples=self._reward_eval_num_samples,
                vote_temperature=self._reward_eval_vote_temperature,
            )
            if eval_cutoff is not None:
                # eval_cutoff is a fraction of frames (already truncated to instr_cutoff).
                # Compose: raw_cutoff = instr_cutoff * eval_cutoff (or just eval_cutoff).
                eval_cutoff = (instr_cutoff or 1.0) * eval_cutoff
            return b, reward, eval_cutoff

        with ThreadPoolExecutor(max_workers=min(32, len(trajs))) as pool:
            futures = {pool.submit(_eval_one, b): b for b in trajs}
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"[{log_prefix}] VLM reward eval",
                leave=False,
            ):
                b, r, eval_cutoff = fut.result()
                vlm_rewards[b] = r
                if eval_cutoff is not None:
                    eval_cutoffs[b] = eval_cutoff

        return vlm_rewards, eval_cutoffs, n_failures

    # ── Batch patching ────────────────────────────────────────────────────────

    def _cutoff_to_t_cut(self, cutoff: float, T: int) -> int:
        """Map a cutoff fraction to the last included chunk index.

        cutoff = n_keep / T (fraction of original trajectory to keep), so
        t_cut = n_keep - 1 = int(round(T * cutoff)) - 1.
        """
        return max(0, min(T - 1, int(round(T * cutoff)) - 1))

    def _patch_rewards(
        self,
        rewards: torch.Tensor,  # [T, B, A]
        trajs: list[int],
        vlm_rewards: dict[int, float],
        reward_coef: float,
        traj_cutoffs: Optional[dict[int, float]] = None,
        dones: Optional[torch.Tensor] = None,  # [T+1, B, A]
        keep_success_reward: bool = False,
    ) -> None:
        """Zero original rewards for relabeled trajectories; write VLM reward.

        rewards: [T, B, A]
          T = n_chunk_step, B = batch, A = num_action_chunks

        For each trajectory b in trajs:
          - rewards[:, b, :] is zeroed
          - VLM reward is written at rewards[t, b, -1]: the last action of the
            terminal chunk (or the cutoff chunk for subtraj)

        When dones is provided ([T+1, B, A]), trajectories with early
        termination (dones before the final step) get their VLM reward
        placed at the last step before the done signal, so that
        calculate_scores picks it up correctly.

        When keep_success_reward is True, trajectories that originally
        succeeded (had early termination) keep their original env rewards
        instead of being zeroed and re-patched.
        """
        T = rewards.shape[0]  # n_chunk_step
        for b in trajs:
            if traj_cutoffs and b in traj_cutoffs:
                rewards[:, b, :].fill_(0.0)
                # subtraj: place reward at the last action of the cutoff chunk
                t_cut = self._cutoff_to_t_cut(traj_cutoffs[b], T)
                rewards[t_cut, b, -1] = vlm_rewards[b] * reward_coef
            elif dones is not None:
                # Check for early termination: any done in dones[1:T, b, :]
                # (excluding boundary at dones[T] which is truncation)
                early_done = dones[1:T, b, :].any()
                if early_done:
                    if keep_success_reward:
                        # Keep original env rewards for succeeded trajectories
                        continue
                    # Place VLM reward at last step before the done signal
                    # Find first t where dones[t+1, b, :] is True (0-indexed in rewards)
                    done_steps = dones[1:T, b, :].any(dim=-1).nonzero(as_tuple=True)[0]
                    t_reward = int(done_steps[0].item())
                    rewards[:, b, :].fill_(0.0)
                    rewards[t_reward, b, -1] = vlm_rewards[b] * reward_coef
                else:
                    rewards[:, b, :].fill_(0.0)
                    rewards[-1, b, -1] = vlm_rewards[b] * reward_coef
            else:
                rewards[:, b, :].fill_(0.0)
                rewards[-1, b, -1] = vlm_rewards[b] * reward_coef

    def _patch_subtraj_terminations(
        self,
        rollout_batch: dict[str, Any],
        eval_trajs: list[int],
        traj_cutoffs_map: dict[int, float],
    ) -> None:
        """Shorten dones and loss_mask to reflect sub-trajectory cutoff points.

        dones:     [T, B, A]  — set to 0 everywhere, then 1 at t_cut+1
        loss_mask: [T, B, 1]  — zeroed from t_cut+1 onwards
        """
        subtraj_trajs = [b for b in eval_trajs if b in traj_cutoffs_map]
        if not subtraj_trajs:
            return

        T = rollout_batch["rewards"].shape[0]  # n_chunk_step
        dones = rollout_batch.get("dones")  # [T, B, A] or None
        loss_mask = rollout_batch.get("loss_mask")  # [T, B, 1] or None

        for b in subtraj_trajs:
            t_cut = self._cutoff_to_t_cut(traj_cutoffs_map[b], T)
            if dones is not None:
                dones[:, b, :].fill_(0.0)
                if t_cut + 1 < T:
                    dones[t_cut + 1, b, :].fill_(1.0)
            if loss_mask is not None:
                loss_mask[t_cut + 1 :, b, :].fill_(0.0)

        if loss_mask is not None:
            # Recompute loss_mask_sum after modifying loss_mask.
            # loss_mask: [T, B, 1]  → sum over T and A → broadcast back to [T, B, 1]
            new_sum = loss_mask.sum(dim=(0, 2), keepdim=True).expand_as(loss_mask)
            rollout_batch["loss_mask_sum"] = new_sum.clone()

    def retokenize_prompts(self, rollout_batch: dict, traj_instructions: dict) -> dict:
        """Tokenize new instructions for the given trajectories.

        Returns a dict with the re-tokenized tensors (NOT written in-place)::

            {
                "tokens": Tensor[n, seq_len],
                "masks": Tensor[n, seq_len],
                "her_bs": list[int],
            }

        Uses ``self.last_setup`` (populated by ``_compute_her_relabeling`` /
        ``__call__``) for tokenizer metadata.
        """
        setup = self.last_setup
        is_openpi = setup["is_openpi"]
        is_groot = setup.get("is_groot", False)
        forward_inputs = rollout_batch["forward_inputs"]
        her_bs = sorted(traj_instructions.keys())

        if is_openpi:
            model = self._get_model_fn()
            inner = model.module if hasattr(model, "module") else model
            new_prompts = [traj_instructions[b] for b in her_bs]
            ref_forward_inputs = {
                "tokenized_prompt": forward_inputs["tokenized_prompt"][:, her_bs, :]
            }
            for key in (
                "observation/image",
                "observation/state",
                "observation/wrist_image",
            ):
                if key in forward_inputs:
                    ref_forward_inputs[key] = forward_inputs[key][:, her_bs]
            tokens, masks = inner.retokenize_prompts_for_second_pass(
                new_prompts, ref_forward_inputs
            )
        elif is_groot:
            model = self._get_model_fn()
            inner = model.module if hasattr(model, "module") else model
            new_prompts = [traj_instructions[b] for b in her_bs]
            ref_forward_inputs = {
                "eagle_input_ids": forward_inputs["eagle_input_ids"][:, her_bs, :],
                "her_raw_frames": forward_inputs["her_raw_frames"][0, her_bs],
            }
            if "her_raw_wrist_frames" in forward_inputs:
                ref_forward_inputs["her_raw_wrist_frames"] = forward_inputs[
                    "her_raw_wrist_frames"
                ][0, her_bs]
            tokens, masks = inner.retokenize_prompts_for_second_pass(
                new_prompts, ref_forward_inputs
            )
        else:
            tokenizer = setup["tokenizer"]
            seq_len = setup["seq_len"]
            pad_id = setup["pad_id"]
            input_ids = forward_inputs["input_ids"]

            prompts = [
                f"In: What action should the robot take to {traj_instructions[b].lower()}?\nOut: "
                for b in her_bs
            ]
            orig_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "left"
            encoded = tokenizer(
                prompts,
                add_special_tokens=True,
                truncation=True,
                max_length=seq_len,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
            )
            tokenizer.padding_side = orig_padding_side

            tokens = encoded["input_ids"].to(
                device=input_ids.device, dtype=input_ids.dtype
            )
            masks = encoded["attention_mask"].to(
                device=input_ids.device, dtype=input_ids.dtype
            )

            bos_id = tokenizer.bos_token_id
            first_nonpad = masks.to(dtype=torch.int64).argmax(dim=1, keepdim=True)
            tokens.scatter_(1, first_nonpad, pad_id)
            masks.scatter_(1, first_nonpad, 0)
            tokens[:, 0] = bos_id
            masks[:, 0] = 1

        return {"tokens": tokens, "masks": masks, "her_bs": her_bs}

    def _patch_prompts(
        self,
        rollout_batch: dict[str, Any],
        setup: dict[str, Any],
        traj_instructions: dict[int, str],
    ) -> None:
        """Re-encode model inputs with the new hindsight instructions.

        For OpenVLA/OFT:
          input_ids:      [T, B, seq_len]  — updated for each relabeled b
          attention_mask: [T, B, seq_len]  — updated for each relabeled b

        For OpenPI:
          tokenized_prompt:      [T, B, seq_len]  — updated for each relabeled b
          tokenized_prompt_mask: [T, B, seq_len]  — updated for each relabeled b

        For GR00T:
          eagle_input_ids:      [T, B, seq_len]  — updated for each relabeled b
          eagle_attention_mask: [T, B, seq_len]  — updated for each relabeled b
        """
        if not traj_instructions:
            return

        is_openpi = setup["is_openpi"]
        is_groot = setup.get("is_groot", False)
        T = setup["T"]
        forward_inputs = rollout_batch["forward_inputs"]
        her_bs = sorted(traj_instructions.keys())

        if is_openpi:
            model = self._get_model_fn()
            inner = model.module if hasattr(model, "module") else model
            new_prompts = [traj_instructions[b] for b in her_bs]
            ref_forward_inputs = {
                "tokenized_prompt": forward_inputs["tokenized_prompt"][:, her_bs, :]
                # [T, len(her_bs), seq_len]
            }
            for key in (
                "observation/image",
                "observation/state",
                "observation/wrist_image",
            ):
                if key in forward_inputs:
                    ref_forward_inputs[key] = forward_inputs[key][:, her_bs]
            new_tp, new_tpm = inner.retokenize_prompts_for_second_pass(
                new_prompts, ref_forward_inputs
            )
            # new_tp, new_tpm: [len(her_bs), seq_len]
            for idx, b in enumerate(her_bs):
                forward_inputs["tokenized_prompt"][:, b, :] = (
                    new_tp[idx].unsqueeze(0).expand(T, -1)
                )
                forward_inputs["tokenized_prompt_mask"][:, b, :] = (
                    new_tpm[idx].unsqueeze(0).expand(T, -1)
                )
                rollout_batch["task_descriptions"][b] = traj_instructions[b]
        elif is_groot:
            model = self._get_model_fn()
            inner = model.module if hasattr(model, "module") else model
            new_prompts = [traj_instructions[b] for b in her_bs]
            ref_forward_inputs = {
                "eagle_input_ids": forward_inputs["eagle_input_ids"][:, her_bs, :],
                "her_raw_frames": forward_inputs["her_raw_frames"][0, her_bs],
            }
            if "her_raw_wrist_frames" in forward_inputs:
                ref_forward_inputs["her_raw_wrist_frames"] = forward_inputs[
                    "her_raw_wrist_frames"
                ][0, her_bs]
            new_ids, new_mask = inner.retokenize_prompts_for_second_pass(
                new_prompts, ref_forward_inputs
            )
            # new_ids, new_mask: [len(her_bs), seq_len]
            for idx, b in enumerate(her_bs):
                forward_inputs["eagle_input_ids"][:, b, :] = (
                    new_ids[idx].unsqueeze(0).expand(T, -1)
                )
                forward_inputs["eagle_attention_mask"][:, b, :] = (
                    new_mask[idx].unsqueeze(0).expand(T, -1)
                )
            if rollout_batch.get("task_descriptions") is not None:
                for b in her_bs:
                    rollout_batch["task_descriptions"][b] = traj_instructions[b]
        else:
            tokenizer = setup["tokenizer"]
            seq_len = setup["seq_len"]
            pad_id = setup["pad_id"]
            input_ids = forward_inputs["input_ids"]  # [T, B, seq_len]

            prompts = [
                f"In: What action should the robot take to {traj_instructions[b].lower()}?\nOut: "
                for b in her_bs
            ]
            orig_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "left"
            encoded = tokenizer(
                prompts,
                add_special_tokens=True,
                truncation=True,
                max_length=seq_len,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
            )
            tokenizer.padding_side = orig_padding_side

            ids_t = encoded["input_ids"].to(
                device=input_ids.device, dtype=input_ids.dtype
            )  # [len(her_bs), seq_len]
            mask_t = encoded["attention_mask"].to(
                device=input_ids.device, dtype=input_ids.dtype
            )  # [len(her_bs), seq_len]

            # Fix BOS: move first non-pad token back so BOS is always at position 0.
            bos_id = tokenizer.bos_token_id
            first_nonpad = mask_t.to(dtype=torch.int64).argmax(dim=1, keepdim=True)
            # first_nonpad: [len(her_bs), 1]
            ids_t.scatter_(1, first_nonpad, pad_id)
            mask_t.scatter_(1, first_nonpad, 0)
            ids_t[:, 0] = bos_id
            mask_t[:, 0] = 1

            for idx, b in enumerate(her_bs):
                forward_inputs["input_ids"][:, b, :] = (
                    ids_t[idx].unsqueeze(0).expand(T, -1)
                )
                forward_inputs["attention_mask"][:, b, :] = (
                    mask_t[idx].unsqueeze(0).expand(T, -1)
                )
                if rollout_batch.get("task_descriptions") is not None:
                    rollout_batch["task_descriptions"][b] = traj_instructions[b]

    # ── Logging helpers ───────────────────────────────────────────────────────

    def _save_her_prompt_debug(
        self,
        frames: list[torch.Tensor],
        prompt_text: str,
        vlm_response: str,
        original_instruction: str,
        group_idx: int,
        cutoff_fraction: Optional[float] = None,
    ) -> None:
        step = self.version
        os.makedirs(self._her_prompt_debug_log_dir, exist_ok=True)
        stem = f"step{step:06d}_g{group_idx:04d}"
        txt_path = os.path.join(self._her_prompt_debug_log_dir, f"{stem}.txt")
        full_video_path = os.path.join(self._her_prompt_debug_log_dir, f"{stem}_full.mp4")
        cutoff_video_path = os.path.join(self._her_prompt_debug_log_dir, f"{stem}_cutoff.mp4")
        subsampled = select_frames_for_vlm(frames, self._vlm.max_frames)
        try:
            _save_trajectory_video(
                subsampled,
                flip_horizontal=self._flip_video_horizontal,
                out_path=full_video_path,
            )
        except Exception as e:
            print(f"[her_debug] WARNING: full video save failed for group {group_idx}: {e}", flush=True)
            full_video_path = f"(save failed: {e})"
        if cutoff_fraction is not None and cutoff_fraction < 1.0:
            n_keep_raw = max(1, int(round(len(frames) * cutoff_fraction)))
            n_keep_sub = max(1, int(round(len(subsampled) * cutoff_fraction)))
            print(
                f"[her_debug] group {group_idx}: total_raw={len(frames)} "
                f"total_sub={len(subsampled)} cutoff={cutoff_fraction:.3f} "
                f"n_keep_raw={n_keep_raw} n_keep_sub={n_keep_sub}",
                flush=True,
            )
            try:
                _save_trajectory_video(
                    subsampled[:n_keep_sub],
                    flip_horizontal=self._flip_video_horizontal,
                    out_path=cutoff_video_path,
                )
            except Exception as e:
                print(f"[her_debug] WARNING: cutoff video save failed for group {group_idx}: {e}", flush=True)
                cutoff_video_path = f"(save failed: {e})"
        else:
            cutoff_video_path = full_video_path  # no cutoff — same as full
        cutoff_str = f"{cutoff_fraction:.3f}" if cutoff_fraction is not None else "None"
        with open(txt_path, "w") as f:
            f.write(f"step: {step}\ngroup: {group_idx}\ncutoff: {cutoff_str}\n")
            f.write(f"original_instruction: {original_instruction}\n")
            f.write(f"full_video:   {full_video_path}\n")
            f.write(f"cutoff_video: {cutoff_video_path}\n\n--- PROMPT ---\n")
            f.write(prompt_text)
            f.write("\n\n--- VLM RESPONSE ---\n")
            f.write(vlm_response)
        print(
            f"[her_debug] saved prompt debug for step {step} group {group_idx}: {txt_path}",
            flush=True,
        )

    def _build_trajectory_log(
        self,
        setup: dict[str, Any],
        pool: list[int],
        traj_instructions: dict[int, str],
        original_rewards: dict[int, float],
        patched_rewards: torch.Tensor,  # [T, B, A] — rollout_batch["rewards"] after patching
        traj_cutoffs_map: dict[int, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Build log entries for a random sample of groups (rank 0 only)."""
        import random as _rng

        if not pool or self._rank != 0:
            return []

        group_size = setup["group_size"]
        unique_groups = list({b // group_size for b in pool})
        sampled_groups = set(
            _rng.sample(unique_groups, min(self._log_max_groups, len(unique_groups)))
        )
        sampled_pool = [b for b in pool if b // group_size in sampled_groups]

        pixel_values = setup["pixel_values"]
        all_tasks = setup["all_tasks"]

        trajectory_log = []
        for b in sampled_pool:
            entry = {
                "trajectory_idx": b,
                "before_prompt": all_tasks[b],
                "after_prompt": traj_instructions[b],
                "before_reward": original_rewards[b],
                "after_reward": patched_rewards[:, b, :].sum().item(),
            }
            try:
                frames = self._get_traj_frames_with_cutoff(
                    pixel_values,
                    b,
                    traj_cutoffs_map.get(b) if traj_cutoffs_map else None,
                )
                entry["video_path"] = _save_trajectory_video(
                    frames, flip_horizontal=self._flip_video_horizontal
                )
            except Exception as e:
                print(
                    f"[trajectory_log] WARNING: Video save failed for traj {b}: {e}",
                    flush=True,
                )
            trajectory_log.append(entry)
        return trajectory_log

    def _log_relabeled_trajs(
        self,
        setup: dict[str, Any],
        her_trajs: list[int],
        traj_instructions: dict[int, str],
        fallback_trajs: set[int],
        original_rewards: dict[int, float],
        patched_rewards: torch.Tensor,  # [T, B, A] — rollout_batch["rewards"] after patching
        traj_cutoffs_map: dict[int, float] | None = None,
        loss_mask: Optional[torch.Tensor] = None,  # [T, B, 1]
    ) -> None:
        """Write a text + video log for all relabeled trajectories (rank 0 only)."""
        if not her_trajs or self._rank != 0:
            return

        step = self.version
        all_tasks = setup["all_tasks"]
        pixel_values = setup["pixel_values"]

        log_dir = os.path.join(self._her_prompt_debug_log_dir, "relabeled_trajs")
        os.makedirs(log_dir, exist_ok=True)
        txt_path = os.path.join(log_dir, f"step{step:06d}.txt")

        lines = [f"step: {step}  total_trajs: {len(her_trajs)}\n"]
        for b in sorted(her_trajs):
            before_reward = original_rewards[b]
            after_reward = patched_rewards[:, b, :].sum().item()
            cutoff = traj_cutoffs_map.get(b) if traj_cutoffs_map else None

            full_frames = self._get_traj_frames(pixel_values, b)
            subsampled = select_frames_for_vlm(full_frames, self._vlm.max_frames)

            full_video_path = os.path.join(log_dir, f"step{step:06d}_b{b:04d}_full.mp4")
            cutoff_video_path = os.path.join(log_dir, f"step{step:06d}_b{b:04d}_cutoff.mp4")
            masked_video_path = os.path.join(log_dir, f"step{step:06d}_b{b:04d}_masked.mp4")
            try:
                _save_trajectory_video(
                    subsampled,
                    flip_horizontal=self._flip_video_horizontal,
                    out_path=full_video_path,
                )
            except Exception as e:
                full_video_path = f"(save failed: {e})"

            if cutoff is not None and cutoff < 1.0:
                n_keep_sub = max(1, int(round(len(subsampled) * cutoff)))
                try:
                    _save_trajectory_video(
                        subsampled[:n_keep_sub],
                        flip_horizontal=self._flip_video_horizontal,
                        out_path=cutoff_video_path,
                    )
                except Exception as e:
                    cutoff_video_path = f"(save failed: {e})"
            else:
                cutoff_video_path = full_video_path

            # Save a video of only the frames not masked out by loss_mask.
            # loss_mask[t, b, 0] > 0 means chunk step t is included in training.
            if loss_mask is not None:
                T = pixel_values.shape[0]
                unmasked_t = [t for t in range(T) if loss_mask[t, b, 0].item() > 0]
                n_unmasked = len(unmasked_t)
                if unmasked_t:
                    unmasked_frames = [pixel_values[t, b] for t in unmasked_t]
                    try:
                        _save_trajectory_video(
                            select_frames_for_vlm(unmasked_frames, self._vlm.max_frames),
                            flip_horizontal=self._flip_video_horizontal,
                            out_path=masked_video_path,
                        )
                    except Exception as e:
                        masked_video_path = f"(save failed: {e})"
                else:
                    masked_video_path = "(all frames masked)"
            else:
                n_unmasked = None
                masked_video_path = "(no loss_mask)"

            cutoff_str = f"{cutoff:.3f}" if cutoff is not None else "None"
            n_unmasked_str = str(n_unmasked) if n_unmasked is not None else "N/A"
            lines.append(
                f"  traj {b:4d} | fallback={b in fallback_trajs} | "
                f"cutoff={cutoff_str} | n_unmasked={n_unmasked_str} | "
                f"before_reward={before_reward:.3f} after_reward={after_reward:.3f} | "
                f"before_prompt: {all_tasks[b]!r}\n"
                f"           after_prompt:  {traj_instructions[b]!r}\n"
                f"           full_video:    {full_video_path}\n"
                f"           cutoff_video:  {cutoff_video_path}\n"
                f"           masked_video:  {masked_video_path}\n"
            )
            print(
                f"[her_relabeled] traj {b} fallback={b in fallback_trajs} "
                f"cutoff={cutoff_str} n_unmasked={n_unmasked_str} "
                f"before_reward={before_reward:.3f} after_reward={after_reward:.3f} "
                f"before_prompt={all_tasks[b]!r} "
                f"after_prompt={traj_instructions[b]!r}",
                flush=True,
            )

        with open(txt_path, "w") as f:
            f.writelines(lines)
        print(
            f"[her_relabeled] saved {len(her_trajs)} traj log to {txt_path}", flush=True
        )

    def _dump_rollout_batch(
        self,
        rollout_batch: dict[str, Any],
        setup: dict[str, Any],
        her_result: dict[str, Any],
    ) -> None:
        """Save the full rollout batch and HER metadata to a pickle file (all ranks)."""
        import pickle

        step = self.version
        dump_dir = self._rollout_dump_dir
        os.makedirs(dump_dir, exist_ok=True)
        out_path = os.path.join(dump_dir, f"step{step:06d}_rank{self._rank}.pkl")

        dump = {
            "step": step,
            "rank": self._rank,
            "group_size": setup["group_size"],
            "n_groups": setup["n_groups"],
            "selected_groups": her_result.get("all_relabeled_traj_indices", []),
            "group_anchors": her_result.get("group_anchors", []),
            "all_tasks": setup["all_tasks"],
            "traj_instructions": her_result.get("traj_instructions", {}),
            "fallback_trajs": her_result.get("fallback_trajs", set()),
            "fallback_groups": her_result.get("fallback_groups", set()),
            "vlm_rewards": her_result.get("vlm_rewards", {}),
            "original_rewards": her_result.get("original_rewards", {}),
            "traj_cutoffs_map": her_result.get("traj_cutoffs_map", {}),
            "rewards": rollout_batch["rewards"].detach().cpu(),
            "actions": rollout_batch["actions"].detach().cpu(),
            "prev_logprobs": rollout_batch["prev_logprobs"].detach().cpu(),
            "dones": rollout_batch["dones"].detach().cpu()
            if "dones" in rollout_batch
            else None,
            "loss_mask": rollout_batch["loss_mask"].detach().cpu()
            if "loss_mask" in rollout_batch
            else None,
            "pixel_values": setup["pixel_values"].detach().cpu(),
        }

        with open(out_path, "wb") as f:
            pickle.dump(dump, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(
            f"[HERProcessor rank={self._rank}] dumped rollout batch to {out_path}",
            flush=True,
        )

    def _dump_relabeled_text(
        self,
        setup: dict[str, Any],
        her_result: dict[str, Any],
    ) -> None:
        """Save only the relabeled instruction text (per-rank JSON) for diversity analysis.

        Much lighter than _dump_rollout_batch: no pixels, rewards, or model tensors.
        Each rank writes one JSON per step containing original tasks, relabeled
        instructions, fallback flags, and group structure.
        """
        import json

        step = self.version
        dump_dir = self._relabeled_text_dump_dir
        os.makedirs(dump_dir, exist_ok=True)
        out_path = os.path.join(dump_dir, f"step{step:06d}_rank{self._rank}.json")

        traj_instructions = her_result.get("traj_instructions", {}) or {}
        fallback_trajs = her_result.get("fallback_trajs", set()) or set()

        dump = {
            "step": step,
            "rank": self._rank,
            "group_size": setup["group_size"],
            "n_groups": setup["n_groups"],
            "all_tasks": setup["all_tasks"],
            "traj_instructions": {str(k): v for k, v in traj_instructions.items()},
            "fallback_trajs": sorted(int(b) for b in fallback_trajs),
            "fallback_groups": sorted(
                int(g) for g in (her_result.get("fallback_groups", set()) or set())
            ),
            "selected_groups": list(her_result.get("all_relabeled_traj_indices", [])),
            "group_anchors": list(her_result.get("group_anchors", [])),
            "traj_cutoffs_map": {
                str(k): v
                for k, v in (her_result.get("traj_cutoffs_map", {}) or {}).items()
            },
        }

        with open(out_path, "w") as f:
            json.dump(dump, f, ensure_ascii=False, indent=2)
        print(
            f"[HERProcessor rank={self._rank}] dumped relabeled text to {out_path}",
            flush=True,
        )

    def _dump_traj_frames(self, setup: dict[str, Any]) -> None:
        """Dump per-trajectory subsampled frames for offline diversity analysis.

        Writes one NPZ per (step, rank) with arrays:
          frames        — uint8 [N_traj, K, H, W, 3] (K = her_dump_traj_frames_n)
          tasks         — object array [N_traj] of original task strings
          traj_indices  — int [N_traj] of in-rank batch indices

        Mirrors `_dump_relabeled_text` so the offline aggregator can glob
        step*_rank*.{json,npz} the same way.
        """
        import numpy as np

        pixel_values = setup["pixel_values"]
        all_tasks = setup["all_tasks"]
        T = pixel_values.shape[0]
        B = pixel_values.shape[1]
        K = int(self._her_dump_traj_frames_n)

        idx = select_frame_indices_for_vlm(T, K)
        # Pad to exactly K by repeating the last frame if T < K, so all
        # trajectories have the same array shape (NPZ requires uniform shape).
        while len(idx) < K:
            idx.append(idx[-1])

        per_traj = []
        for b in range(B):
            frames_b = [_to_vlm_frame_uint8(pixel_values[t, b]) for t in idx]
            per_traj.append(np.stack(frames_b, axis=0))  # [K, H, W, 3]
        frames_arr = np.stack(per_traj, axis=0)  # [B, K, H, W, 3] uint8

        step = self.version
        os.makedirs(self._traj_frames_dump_dir, exist_ok=True)
        out_path = os.path.join(
            self._traj_frames_dump_dir,
            f"step{step:06d}_rank{self._rank}.npz",
        )
        np.savez_compressed(
            out_path,
            frames=frames_arr,
            tasks=np.array(all_tasks, dtype=object),
            traj_indices=np.arange(B, dtype=np.int32),
        )
        print(
            f"[HERProcessor rank={self._rank}] dumped {B} traj frames "
            f"({frames_arr.shape}) to {out_path}",
            flush=True,
        )

    # ── Core HER pipeline ─────────────────────────────────────────────────────

    def _compute_her_relabeling(
        self,
        rollout_batch: dict[str, Any],
        setup: dict[str, Any],
        selected_groups: list[int],
        group_anchors: list[int],
    ) -> dict[str, Any]:
        """Compute HER relabeling for the selected groups (no batch mutation).

        Runs instruction generation, fallback resolution, trajectory expansion,
        and VLM reward evaluation.  Returns a result dict with all computed data
        and count-based metrics.  Does NOT modify *rollout_batch*.
        """
        import random as _rng

        # Pick a random subset of groups whose prompt+response will be saved to disk
        # for debugging. Only rank 0 writes files to avoid duplicates.
        debug_groups: set = set()
        if self._her_prompt_debug_log_n > 0 and self._rank == 0 and selected_groups:
            debug_groups = set(
                _rng.sample(
                    selected_groups,
                    min(self._her_prompt_debug_log_n, len(selected_groups)),
                )
            )

        # If no groups were selected there is nothing to relabel — return early
        # with zeroed metrics so the caller can still log them.
        if not selected_groups:
            print(
                f"[HERProcessor rank={self._rank}] no groups selected, skipping relabeling.",
                flush=True,
            )
            return {
                "traj_instructions": {},
                "fallback_trajs": set(),
                "vlm_rewards": {},
                "eval_traj_indices": [],
                "all_relabeled_traj_indices": [],
                "traj_cutoffs_map": {},
                "group_anchors": group_anchors,
                "group_cutoffs": {},
                "fallback_groups": set(),
                "setup": setup,
                "metrics": {
                    "her_groups_with_anchor": 0,
                    "her_groups_skipped": setup["n_groups"],
                    "her_eval_failures": 0,
                    "her_positive_rate": 0.0,
                    "her_relabeled_all_zero": 0,
                    "her_relabeled_all_success": 0,
                    "her_subtraj_groups": 0,
                    "her_filtered_uninteresting": 0,
                },
            }

        # Step 1: query the VLM for a hindsight instruction for each selected group.
        # One call per group (using the anchor trajectory); parallelized across groups.
        print(
            f"[HERProcessor rank={self._rank}] _apply_her step 1/8: generating group instructions ({len(selected_groups)} groups).",
            flush=True,
        )
        group_instructions, group_cutoffs, n_instr_failures = (
            self._generate_group_instructions(
                selected_groups, group_anchors, setup, debug_groups
            )
        )

        # Step 2: handle groups where the VLM said "Nothing".
        # Replace with the original instruction and mark as fallback so they skip
        # reward eval (preserving original reward, not getting a new VLM reward).
        print(
            f"[HERProcessor rank={self._rank}] _apply_her step 2/8: resolving Nothing fallbacks.",
            flush=True,
        )
        fallback_groups, n_filtered_uninteresting, per_instr_uninteresting = (
            self._resolve_nothing_fallbacks(
                selected_groups,
                group_anchors,
                group_instructions,
                setup["all_tasks"],
            )
        )

        # Step 3: broadcast each group's instruction to all trajectories b in that group.
        # traj_instructions[b] = instruction string for trajectory b
        # traj_cutoffs_map[b]   = cutoff fraction for subtraj trajectories
        # fallback_trajs        = set of b values whose group fell back to original
        print(
            f"[HERProcessor rank={self._rank}] _apply_her step 3/8: expanding to traj instructions (fallback_groups={len(fallback_groups)}).",
            flush=True,
        )
        traj_instructions, traj_cutoffs_map, fallback_trajs = (
            self._expand_to_traj_instructions(
                selected_groups,
                group_instructions,
                group_cutoffs,
                setup["group_size"],
                fallback_groups,
            )
        )

        # Step 3.5: if her_skip_same_instruction is enabled, treat trajectories
        # whose relabeled instruction matches the original (ignoring case and
        # punctuation) as fallbacks — they keep their original reward.
        n_same_instruction = 0
        if self.cfg.algorithm.get("her_skip_same_instruction", False):
            import re as _re

            def _normalize(s: str) -> str:
                return _re.sub(r"[^\w\s]", "", s).strip().lower()

            for b, instr in list(traj_instructions.items()):
                if b in fallback_trajs:
                    continue
                if _normalize(instr) == _normalize(setup["all_tasks"][b]):
                    fallback_trajs.add(b)
                    n_same_instruction += 1
            if n_same_instruction > 0:
                print(
                    f"[HERProcessor rank={self._rank}] skipped {n_same_instruction} trajs "
                    f"with same instruction (her_skip_same_instruction).",
                    flush=True,
                )

        # Step 4: query the VLM for a binary (0/1) reward for each relabeled trajectory.
        # Only non-fallback trajectories are evaluated — fallback trajectories already
        # have the original instruction restored, so there is nothing new to evaluate.
        # vlm_rewards: {traj_idx -> 1.0 (success) or 0.0 (failure)}
        all_relabeled_traj_indices = list(traj_instructions.keys())
        eval_traj_indices = [
            b for b in all_relabeled_traj_indices if b not in fallback_trajs
        ]
        reward_coef = self.cfg.env.train.get("reward_coef", 1.0)

        # Snapshot original env rewards — used in logging to show before/after reward.
        # {traj_idx -> scalar reward sum from the environment}
        rewards = rollout_batch["rewards"]  # [T, B, A]
        original_rewards = {
            b: rewards[:, b, :].sum().item() for b in all_relabeled_traj_indices
        }

        print(
            f"[HERProcessor rank={self._rank}] _apply_her step 4/8: parallel reward eval ({len(eval_traj_indices)} trajs).",
            flush=True,
        )
        vlm_rewards, eval_cutoffs, n_eval_failures = self._run_parallel_reward_eval(
            eval_traj_indices,
            setup["pixel_values"],
            traj_instructions,
            "her",
            traj_cutoffs=traj_cutoffs_map or None,
        )
        # Merge reward-eval cutoffs into traj_cutoffs_map.
        # For failed trajs, the reward eval may return a tighter cutoff
        # (where the robot made its best partial attempt).
        # Only update if the eval cutoff is tighter than the existing one.
        for b, ec in eval_cutoffs.items():
            existing = traj_cutoffs_map.get(b)
            if existing is None or ec < existing:
                traj_cutoffs_map[b] = ec

        # Compute per-group outcome statistics for metrics and the summary print.
        n_relabeled = len(eval_traj_indices)
        n_positive = sum(1 for b in eval_traj_indices if vlm_rewards[b] > 0)
        group_size = setup["group_size"]
        n_all_zero = 0
        n_all_success = 0
        for g in selected_groups:
            if g in fallback_groups:
                continue
            group_traj_indices = range(g * group_size, (g + 1) * group_size)
            evaluated = [b for b in group_traj_indices if b in vlm_rewards]
            if not evaluated:
                continue
            rewards_in_group = [vlm_rewards[b] for b in evaluated]
            if all(r == 0.0 for r in rewards_in_group):
                n_all_zero += 1
            if all(r > 0.0 for r in rewards_in_group):
                n_all_success += 1

        metrics: dict = {
            "her_groups_with_anchor": len(selected_groups),
            "her_groups_skipped": setup["n_groups"] - len(selected_groups),
            "her_eval_failures": n_eval_failures,
            "her_positive_rate": n_positive / max(n_relabeled, 1),
            "her_relabeled_all_zero": n_all_zero,
            "her_relabeled_all_success": n_all_success,
            "her_subtraj_groups": len(group_cutoffs),
            "her_eval_cutoffs": len(eval_cutoffs),
            "her_filtered_uninteresting": n_filtered_uninteresting,
            "her_skipped_same_instruction": n_same_instruction,
        }
        for task_desc, rate in per_instr_uninteresting.items():
            metrics[f"her_uninteresting_per_instr/{task_desc}"] = rate

        # Stash setup so retokenize_prompts can access tokenizer metadata.
        self.last_setup = setup

        print(
            f"[her] {len(selected_groups)}/{setup['n_groups']} groups with anchor, "
            f"fallback_groups={len(fallback_groups)}, "
            f"subtraj_groups={len(group_cutoffs)}, eval_cutoffs={len(eval_cutoffs)}, "
            f"{n_relabeled} trajs relabeled, {n_positive} positive, "
            f"all_zero={n_all_zero}, all_success={n_all_success}, "
            f"instr_failures={n_instr_failures}, eval_failures={n_eval_failures}",
            flush=True,
        )
        return {
            "traj_instructions": traj_instructions,
            "fallback_trajs": fallback_trajs,
            "vlm_rewards": vlm_rewards,
            "eval_traj_indices": eval_traj_indices,
            "all_relabeled_traj_indices": all_relabeled_traj_indices,
            "traj_cutoffs_map": traj_cutoffs_map,
            "group_anchors": group_anchors,
            "group_cutoffs": group_cutoffs,
            "fallback_groups": fallback_groups,
            "reward_coef": reward_coef,
            "original_rewards": original_rewards,
            "setup": setup,
            "metrics": metrics,
        }

    def _apply_relabeling_to_batch(
        self,
        rollout_batch: dict[str, Any],
        setup: dict[str, Any],
        her_result: dict[str, Any],
    ) -> None:
        """Apply precomputed HER relabeling results to the batch in-place.

        Patches rewards, subtraj terminations, prompts, and adds
        ``trajectory_log`` to ``her_result["metrics"]``.
        """
        # Nothing to apply if no groups were selected.
        if not her_result["all_relabeled_traj_indices"]:
            return

        rewards = rollout_batch["rewards"]
        eval_traj_indices = her_result["eval_traj_indices"]
        vlm_rewards = her_result["vlm_rewards"]
        reward_coef = her_result["reward_coef"]
        traj_cutoffs_map = her_result["traj_cutoffs_map"]
        traj_instructions = her_result["traj_instructions"]
        fallback_trajs = her_result["fallback_trajs"]
        original_rewards = her_result["original_rewards"]
        all_relabeled_traj_indices = her_result["all_relabeled_traj_indices"]

        # Write VLM rewards back into the rollout batch.
        print(
            f"[HERProcessor rank={self._rank}] _apply_her step 5/8: patching rewards.",
            flush=True,
        )
        keep_success_reward = self.cfg.algorithm.get(
            "her_keep_success_reward", False
        )
        self._patch_rewards(
            rewards,
            eval_traj_indices,
            vlm_rewards,
            reward_coef,
            traj_cutoffs=traj_cutoffs_map or None,
            dones=rollout_batch.get("dones"),
            keep_success_reward=keep_success_reward,
        )
        if traj_cutoffs_map:
            self._patch_subtraj_terminations(
                rollout_batch, eval_traj_indices, traj_cutoffs_map
            )

        # Optional: override anchor trajectory rewards to 1 regardless of VLM.
        print(
            f"[HERProcessor rank={self._rank}] _apply_her step 6/8: anchor reward override.",
            flush=True,
        )
        if self.cfg.algorithm.get("her_anchor_reward_one", False):
            T = rewards.shape[0]
            dones = rollout_batch.get("dones")
            for anchor_b in her_result["group_anchors"]:
                if anchor_b in fallback_trajs:
                    continue
                rewards[:, anchor_b, :].fill_(0.0)
                if traj_cutoffs_map and anchor_b in traj_cutoffs_map:
                    t_cut = self._cutoff_to_t_cut(traj_cutoffs_map[anchor_b], T)
                    rewards[t_cut, anchor_b, -1] = reward_coef
                elif dones is not None and dones[1:T, anchor_b, :].any():
                    done_steps = (
                        dones[1:T, anchor_b, :].any(dim=-1).nonzero(as_tuple=True)[0]
                    )
                    t_reward = int(done_steps[0].item())
                    rewards[t_reward, anchor_b, -1] = reward_coef
                else:
                    rewards[-1, anchor_b, -1] = reward_coef

        # Optional: write per-trajectory text + video log to disk (rank 0 only).
        print(
            f"[HERProcessor rank={self._rank}] _apply_her step 7/8: logging relabeled trajs.",
            flush=True,
        )
        if self._her_log_relabeled_trajs:
            self._log_relabeled_trajs(
                setup,
                all_relabeled_traj_indices,
                traj_instructions,
                fallback_trajs,
                original_rewards,
                rewards,
                traj_cutoffs_map=traj_cutoffs_map or None,
                loss_mask=rollout_batch.get("loss_mask"),
            )

        if self._her_dump_rollout_batch:
            self._dump_rollout_batch(rollout_batch, setup, her_result)

        if self._her_dump_relabeled_text:
            self._dump_relabeled_text(setup, her_result)

        if self._her_dump_traj_frames:
            self._dump_traj_frames(setup)

        # Re-tokenize model inputs with the new hindsight instructions.
        non_fallback_instructions = {
            b: instr
            for b, instr in traj_instructions.items()
            if b not in fallback_trajs
        }
        print(
            f"[HERProcessor rank={self._rank}] _apply_her step 8/8: patching prompts ({len(non_fallback_instructions)} trajs).",
            flush=True,
        )
        self._patch_prompts(rollout_batch, setup, non_fallback_instructions)

        # Build trajectory log after the batch has been patched.
        her_result["metrics"]["trajectory_log"] = self._build_trajectory_log(
            setup,
            all_relabeled_traj_indices,
            traj_instructions,
            original_rewards,
            rewards,
            traj_cutoffs_map=traj_cutoffs_map or None,
        )

    # ── Shuffled-instruction HER entry point ─────────────────────────────────

    def process_shuffled(
        self,
        rollout_batch: dict,
        instruction_pool: list[str],
    ) -> dict[str, Any]:
        """Apply HER with instructions randomly sampled from an external pool.

        For each group, one instruction is drawn uniformly from *instruction_pool*
        after excluding the group's own original task.  The VLM reward-eval
        pipeline then runs as normal on the sampled instruction.

        Unlike ``__call__``, this method processes ALL groups (no stochastic
        group-selection filter) and never calls the instruction-generation VLM.

        Args:
            rollout_batch:    The rollout data dict (mutated in-place).
            instruction_pool: Candidate instructions to sample from.  Must
                              contain at least one entry different from each
                              group's original task.

        Returns:
            Result dict with ``traj_instructions``, ``vlm_rewards``, ``setup``,
            and ``metrics`` (keys prefixed with ``her_shuffled_``).
        """
        import random as _rng

        print(
            f"[HERProcessor rank={self._rank}] process_shuffled: setting up batch.",
            flush=True,
        )
        setup = self._setup_her_batch(rollout_batch)
        B = setup["B"]
        group_size = setup["group_size"]
        n_groups = setup["n_groups"]
        all_tasks = setup["all_tasks"]

        # Sample one instruction per group from the pool, excluding the group's
        # own original task to ensure the instruction is always different.
        # Compare case-insensitively because OpenVLA's all_tasks comes from a
        # lower-cased prompt while the YAML pool may use any case.
        # pi05 was SFT'd on instructions starting with an uppercase letter, and
        # the rest of this codebase enforces the same convention on every
        # VLM-generated instruction (her_vlm_client / her_utils / her_contrastive).
        # Match that here so the sampled instruction is in the same format the
        # downstream tokenizer/model expects regardless of the YAML pool's case.
        def _to_canonical(s: str) -> str:
            s = s.strip()
            return s[0].upper() + s[1:] if s else s

        group_instructions: dict[int, str] = {}
        for g in range(n_groups):
            group_task = all_tasks[g * group_size]
            group_task_norm = group_task.strip().lower()
            # Compare case-insensitively because OpenVLA's all_tasks comes from
            # a lower-cased prompt while OpenPI's preserves original case.
            pool_filtered = [
                t for t in instruction_pool if t.strip().lower() != group_task_norm
            ]
            if not pool_filtered:
                raise ValueError(
                    f"[her_shuffled] instruction pool for group {g} is empty "
                    f"after excluding '{group_task}'. "
                    "Add more entries to her_shuffled_instruction_pool."
                )
            group_instructions[g] = _to_canonical(
                _rng.choice(pool_filtered).rstrip(".,;:!?")
            )

        print(
            f"[HERProcessor rank={self._rank}] process_shuffled: "
            f"sampled instructions for {n_groups} groups.",
            flush=True,
        )

        # Broadcast each group's instruction to its trajectories (no fallbacks).
        traj_instructions, _, _ = self._expand_to_traj_instructions(
            list(range(n_groups)),
            group_instructions,
            {},
            group_size,
            set(),
        )

        rewards = rollout_batch["rewards"]  # [T, B, A]
        reward_coef = self.cfg.env.train.get("reward_coef", 1.0)
        original_rewards = {b: rewards[:, b, :].sum().item() for b in range(B)}

        print(
            f"[HERProcessor rank={self._rank}] process_shuffled: "
            f"running reward eval for {B} trajectories.",
            flush=True,
        )
        vlm_rewards, eval_cutoffs, n_failures = self._run_parallel_reward_eval(
            list(range(B)),
            setup["pixel_values"],
            traj_instructions,
            "her_shuffled",
        )

        keep_success_reward = self.cfg.algorithm.get(
            "her_keep_success_reward", False
        )
        self._patch_rewards(
            rewards,
            list(range(B)),
            vlm_rewards,
            reward_coef,
            traj_cutoffs=eval_cutoffs or None,
            dones=rollout_batch.get("dones"),
            keep_success_reward=keep_success_reward,
        )
        if eval_cutoffs:
            self._patch_subtraj_terminations(rollout_batch, list(range(B)), eval_cutoffs)

        # Store setup so _patch_prompts can access tokenizer metadata.
        self.last_setup = setup
        self._patch_prompts(rollout_batch, setup, traj_instructions)

        # Compute per-group outcome statistics.
        n_positive = sum(1 for b in range(B) if vlm_rewards.get(b, 0.0) > 0)
        n_all_zero = 0
        n_all_success = 0
        for g in range(n_groups):
            group_bs = list(range(g * group_size, (g + 1) * group_size))
            g_rewards = [vlm_rewards[b] for b in group_bs if b in vlm_rewards]
            if not g_rewards:
                continue
            if all(r == 0.0 for r in g_rewards):
                n_all_zero += 1
            if all(r > 0.0 for r in g_rewards):
                n_all_success += 1

        metrics: dict = {
            "her_shuffled_groups": n_groups,
            "her_shuffled_eval_failures": n_failures,
            "her_shuffled_positive_rate": n_positive / max(B, 1),
            "her_shuffled_all_zero": n_all_zero,
            "her_shuffled_all_success": n_all_success,
        }

        print(
            f"[her_shuffled] {n_groups} groups, {B} trajs, "
            f"{n_positive} positive, all_zero={n_all_zero}, "
            f"all_success={n_all_success}, eval_failures={n_failures}",
            flush=True,
        )

        metrics["trajectory_log"] = self._build_trajectory_log(
            setup,
            list(traj_instructions.keys()),
            traj_instructions,
            original_rewards,
            rewards,
        )

        return {
            "traj_instructions": traj_instructions,
            "vlm_rewards": vlm_rewards,
            "setup": setup,
            "metrics": metrics,
        }

    # ── Public entry point ────────────────────────────────────────────────────

    def __call__(
        self,
        rollout_batch: dict,
        apply_to_batch: bool = True,
        force_all_groups: bool = False,
        random_select_prob: float | None = None,
    ) -> dict[str, Any]:
        """Run HER relabeling pipeline.

        Args:
            rollout_batch: The rollout data dict.
            apply_to_batch: If True (default), patch *rollout_batch* in-place
                with relabeled rewards, terminations, and prompts.  If False,
                only compute relabeling without modifying the batch.
            force_all_groups: If True, bypass ``_filter_groups`` and relabel
                every group. Anchors are still picked via ``_pick_anchors``.
            random_select_prob: If set (in (0, 1]), bypass ``_filter_groups``
                and instead include each group independently with this
                probability (uniform over groups, no reward conditioning).
                Mutually exclusive with ``force_all_groups``.

        Returns:
            *her_result* dict containing all relabeling outputs
            (``traj_instructions``, ``vlm_rewards``, ``setup``, etc.)
            and ``her_result["metrics"]`` for logging.
        """
        assert self.her_endpoint, "HER endpoint not set"
        if force_all_groups and random_select_prob is not None:
            raise ValueError(
                "force_all_groups and random_select_prob are mutually exclusive"
            )
        if random_select_prob is not None and not (0.0 < random_select_prob <= 1.0):
            raise ValueError(
                f"random_select_prob must be in (0, 1], got {random_select_prob}"
            )

        print(
            f"[HERProcessor rank={self._rank}] Stage 1/4: setting up HER batch.",
            flush=True,
        )
        setup = self._setup_her_batch(rollout_batch)
        B = setup["B"]
        # Sum rewards over T and A to get a scalar per trajectory: [B]
        per_traj_reward = (
            rollout_batch["rewards"].transpose(0, 1).reshape(B, -1).sum(dim=-1)
        )  # [B]
        print(
            f"[HERProcessor rank={self._rank}] Stage 2/4: selecting groups (B={B}, n_groups={setup['n_groups']}).",
            flush=True,
        )
        if force_all_groups:
            selected_groups, group_anchors = self._pick_anchors(
                list(range(setup["n_groups"])), per_traj_reward, setup["group_size"]
            )
        elif random_select_prob is not None:
            import random as _rng

            n_groups = setup["n_groups"]
            random_selected = [
                g for g in range(n_groups) if _rng.random() < random_select_prob
            ]
            print(
                f"[her] {len(random_selected)}/{n_groups} groups selected "
                f"(selection=random(p={random_select_prob}))",
                flush=True,
            )
            selected_groups, group_anchors = self._pick_anchors(
                random_selected, per_traj_reward, setup["group_size"]
            )
        else:
            selected_groups, group_anchors = self._select_groups(
                per_traj_reward, setup["group_size"], setup["n_groups"]
            )

        her_result = self._compute_her_relabeling(
            rollout_batch, setup, selected_groups, group_anchors
        )

        if apply_to_batch:
            self._apply_relabeling_to_batch(rollout_batch, setup, her_result)

        return her_result
