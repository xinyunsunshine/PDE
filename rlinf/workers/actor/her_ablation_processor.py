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

"""HER ablation processors: Rephrase-only, Reward-eval-only, and Random-reward.

RephraseOnlyProcessor:
    Rephrases the original instruction via VLM (text-only, no trajectory video)
    and re-tokenizes prompts with the rephrased instruction. Keeps original env
    rewards — no VLM reward evaluation.

RewardEvalOnlyProcessor:
    Skips VLM instruction generation; uses original task instructions.
    Runs VLM reward eval against the original instruction and patches rewards,
    but does not re-tokenize prompts.

RandomRewardProcessor:
    Skips VLM entirely. Assigns a random binary reward (0.0 or 1.0) to each
    trajectory, placed at the correct timestep (matching the reward placement
    logic of the HER dual and reward-eval-only variants).
"""

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from rlinf.workers.actor.her_processor import HERProcessor


class RephraseOnlyProcessor(HERProcessor):
    """HER ablation: rephrase original instructions, keep original env rewards.

    Asks the VLM to rephrase each group's original instruction (text-only, no
    trajectory video). Re-tokenizes prompts with the rephrased instruction.
    Skips VLM reward eval — original env rewards are preserved unchanged.
    """

    def _compute_her_relabeling(
        self,
        rollout_batch: dict[str, Any],
        setup: dict[str, Any],
        selected_groups: list[int],
        group_anchors: list[int],
    ) -> dict[str, Any]:
        if not selected_groups:
            print(
                f"[RephraseOnlyProcessor rank={self._rank}] no groups selected.",
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
                "reward_coef": self.cfg.env.train.get("reward_coef", 1.0),
                "original_rewards": {},
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

        all_tasks = setup["all_tasks"]
        group_size = setup["group_size"]
        temperature = self.cfg.algorithm.get("her_temperature", 0.8)

        print(
            f"[RephraseOnlyProcessor rank={self._rank}] "
            f"rephrasing instructions for {len(selected_groups)} groups "
            f"(text-only, no hindsight relabeling).",
            flush=True,
        )

        # Rephrase each group's original instruction via VLM (text-only).
        group_instructions: dict[int, str] = {}
        n_rephrase_failures = 0

        def _rephrase_one(g: int, anchor_b: int) -> tuple[int, str]:
            original = all_tasks[anchor_b]
            try:
                rephrased = self._vlm.request_rephrase_instruction(
                    original, temperature=temperature
                )
            except RuntimeError:
                return g, original
            return g, rephrased.rstrip(".,;:!?")

        with ThreadPoolExecutor(
            max_workers=min(16, len(selected_groups))
        ) as pool:
            futures = {
                pool.submit(_rephrase_one, g, group_anchors[i]): g
                for i, g in enumerate(selected_groups)
            }
            for fut in as_completed(futures):
                g, instr = fut.result()
                if instr == all_tasks[group_anchors[selected_groups.index(g)]]:
                    n_rephrase_failures += 1
                group_instructions[g] = instr

        # Expand group instructions to per-trajectory.
        traj_instructions: dict[int, str] = {}
        fallback_trajs: set[int] = set()
        for g in selected_groups:
            instr = group_instructions.get(g)
            if instr is None:
                continue
            for i in range(group_size):
                b = g * group_size + i
                traj_instructions[b] = instr

        # Skip-same-instruction filter.
        n_same_instruction = 0
        if self.cfg.algorithm.get("her_skip_same_instruction", False):
            import re as _re

            def _normalize(s: str) -> str:
                return _re.sub(r"[^\w\s]", "", s).strip().lower()

            for b, instr in list(traj_instructions.items()):
                if _normalize(instr) == _normalize(all_tasks[b]):
                    fallback_trajs.add(b)
                    n_same_instruction += 1

        all_relabeled_traj_indices = list(traj_instructions.keys())
        reward_coef = self.cfg.env.train.get("reward_coef", 1.0)
        rewards = rollout_batch["rewards"]
        original_rewards = {
            b: rewards[:, b, :].sum().item() for b in all_relabeled_traj_indices
        }

        print(
            f"[RephraseOnlyProcessor rank={self._rank}] "
            f"{len(all_relabeled_traj_indices)} trajs rephrased, "
            f"{len(fallback_trajs)} same-as-original, "
            f"rephrase_failures={n_rephrase_failures}. "
            f"Keeping original env rewards.",
            flush=True,
        )

        self.last_setup = setup

        return {
            "traj_instructions": traj_instructions,
            "fallback_trajs": fallback_trajs,
            "vlm_rewards": {},
            "eval_traj_indices": [],
            "all_relabeled_traj_indices": all_relabeled_traj_indices,
            "traj_cutoffs_map": {},
            "group_anchors": group_anchors,
            "group_cutoffs": {},
            "fallback_groups": set(),
            "reward_coef": reward_coef,
            "original_rewards": original_rewards,
            "setup": setup,
            "metrics": {
                "her_groups_with_anchor": len(selected_groups),
                "her_groups_skipped": setup["n_groups"] - len(selected_groups),
                "her_eval_failures": 0,
                "her_positive_rate": 0.0,
                "her_relabeled_all_zero": 0,
                "her_relabeled_all_success": 0,
                "her_subtraj_groups": 0,
                "her_filtered_uninteresting": 0,
                "her_skipped_same_instruction": n_same_instruction,
                "her_rephrase_failures": n_rephrase_failures,
            },
        }

    def _apply_relabeling_to_batch(
        self,
        rollout_batch: dict[str, Any],
        setup: dict[str, Any],
        her_result: dict[str, Any],
    ) -> None:
        if not her_result["all_relabeled_traj_indices"]:
            return

        traj_instructions = her_result["traj_instructions"]
        fallback_trajs = her_result["fallback_trajs"]

        # Skip reward patching — keep original env rewards.

        # Re-tokenize prompts with new hindsight instructions.
        non_fallback_instructions = {
            b: instr
            for b, instr in traj_instructions.items()
            if b not in fallback_trajs
        }
        print(
            f"[RephraseOnlyProcessor rank={self._rank}] patching prompts "
            f"({len(non_fallback_instructions)} trajs), keeping env rewards.",
            flush=True,
        )
        self._patch_prompts(rollout_batch, setup, non_fallback_instructions)

        her_result["metrics"]["trajectory_log"] = self._build_trajectory_log(
            setup,
            her_result["all_relabeled_traj_indices"],
            traj_instructions,
            her_result["original_rewards"],
            rollout_batch["rewards"],
            traj_cutoffs_map=None,
        )


class RewardEvalOnlyProcessor(HERProcessor):
    """HER ablation: VLM reward eval on original instructions, no rephrase.

    Skips VLM instruction generation entirely. Uses original task instructions
    and runs VLM reward eval against them to replace env rewards with VLM
    rewards. Does not re-tokenize prompts.
    """

    def _compute_her_relabeling(
        self,
        rollout_batch: dict[str, Any],
        setup: dict[str, Any],
        selected_groups: list[int],
        group_anchors: list[int],
    ) -> dict[str, Any]:
        if not selected_groups:
            print(
                f"[RewardEvalOnlyProcessor rank={self._rank}] no groups selected.",
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
                "reward_coef": self.cfg.env.train.get("reward_coef", 1.0),
                "original_rewards": {},
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

        # Skip steps 1-3: use original task instructions directly.
        group_size = setup["group_size"]
        traj_instructions: dict[int, str] = {}
        for g in selected_groups:
            for i in range(group_size):
                b = g * group_size + i
                traj_instructions[b] = setup["all_tasks"][b]

        all_relabeled_traj_indices = list(traj_instructions.keys())
        eval_traj_indices = all_relabeled_traj_indices
        reward_coef = self.cfg.env.train.get("reward_coef", 1.0)

        rewards = rollout_batch["rewards"]
        original_rewards = {
            b: rewards[:, b, :].sum().item() for b in all_relabeled_traj_indices
        }

        # Step 4: run VLM reward eval against original instructions.
        print(
            f"[RewardEvalOnlyProcessor rank={self._rank}] "
            f"running VLM reward eval on {len(eval_traj_indices)} trajs "
            f"with original instructions (no rephrase).",
            flush=True,
        )
        vlm_rewards, eval_cutoffs, n_eval_failures = self._run_parallel_reward_eval(
            eval_traj_indices,
            setup["pixel_values"],
            traj_instructions,
            "her_reward_only",
            traj_cutoffs=None,
        )

        traj_cutoffs_map: dict[int, float] = {}
        for b, ec in eval_cutoffs.items():
            traj_cutoffs_map[b] = ec

        n_relabeled = len(eval_traj_indices)
        n_positive = sum(1 for b in eval_traj_indices if vlm_rewards.get(b, 0) > 0)
        n_all_zero = 0
        n_all_success = 0
        for g in selected_groups:
            group_traj_indices = range(g * group_size, (g + 1) * group_size)
            evaluated = [b for b in group_traj_indices if b in vlm_rewards]
            if not evaluated:
                continue
            rewards_in_group = [vlm_rewards[b] for b in evaluated]
            if all(r == 0.0 for r in rewards_in_group):
                n_all_zero += 1
            if all(r > 0.0 for r in rewards_in_group):
                n_all_success += 1

        print(
            f"[RewardEvalOnlyProcessor rank={self._rank}] "
            f"{n_relabeled} trajs evaluated, {n_positive} positive, "
            f"all_zero={n_all_zero}, all_success={n_all_success}, "
            f"eval_failures={n_eval_failures}",
            flush=True,
        )

        self.last_setup = setup

        return {
            "traj_instructions": traj_instructions,
            "fallback_trajs": set(),
            "vlm_rewards": vlm_rewards,
            "eval_traj_indices": eval_traj_indices,
            "all_relabeled_traj_indices": all_relabeled_traj_indices,
            "traj_cutoffs_map": traj_cutoffs_map,
            "group_anchors": group_anchors,
            "group_cutoffs": {},
            "fallback_groups": set(),
            "reward_coef": reward_coef,
            "original_rewards": original_rewards,
            "setup": setup,
            "metrics": {
                "her_groups_with_anchor": len(selected_groups),
                "her_groups_skipped": setup["n_groups"] - len(selected_groups),
                "her_eval_failures": n_eval_failures,
                "her_positive_rate": n_positive / max(n_relabeled, 1),
                "her_relabeled_all_zero": n_all_zero,
                "her_relabeled_all_success": n_all_success,
                "her_subtraj_groups": 0,
                "her_eval_cutoffs": len(eval_cutoffs),
                "her_filtered_uninteresting": 0,
                "her_skipped_same_instruction": 0,
            },
        }

    def _apply_relabeling_to_batch(
        self,
        rollout_batch: dict[str, Any],
        setup: dict[str, Any],
        her_result: dict[str, Any],
    ) -> None:
        if not her_result["all_relabeled_traj_indices"]:
            return

        rewards = rollout_batch["rewards"]
        eval_traj_indices = her_result["eval_traj_indices"]
        vlm_rewards = her_result["vlm_rewards"]
        reward_coef = her_result["reward_coef"]
        traj_cutoffs_map = her_result["traj_cutoffs_map"]
        original_rewards = her_result["original_rewards"]
        all_relabeled_traj_indices = her_result["all_relabeled_traj_indices"]

        # Patch rewards with VLM scores.
        print(
            f"[RewardEvalOnlyProcessor rank={self._rank}] "
            f"patching rewards ({len(eval_traj_indices)} trajs), skipping prompt patch.",
            flush=True,
        )
        keep_success_reward = self.cfg.algorithm.get("her_keep_success_reward", False)
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

        # Skip prompt patching — instructions are unchanged.

        her_result["metrics"]["trajectory_log"] = self._build_trajectory_log(
            setup,
            all_relabeled_traj_indices,
            her_result["traj_instructions"],
            original_rewards,
            rewards,
            traj_cutoffs_map=traj_cutoffs_map or None,
        )


class RandomRewardProcessor(HERProcessor):
    """HER ablation: random binary reward, no VLM calls.

    Assigns a random reward (0.0 or 1.0) to each trajectory. Rewards are
    placed at the correct terminal timestep using the same _patch_rewards
    logic as the reward-eval-only variant. No VLM calls are made, and
    prompts are not re-tokenized.
    """

    def _compute_her_relabeling(
        self,
        rollout_batch: dict[str, Any],
        setup: dict[str, Any],
        selected_groups: list[int],
        group_anchors: list[int],
    ) -> dict[str, Any]:
        if not selected_groups:
            print(
                f"[RandomRewardProcessor rank={self._rank}] no groups selected.",
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
                "reward_coef": self.cfg.env.train.get("reward_coef", 1.0),
                "original_rewards": {},
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

        group_size = setup["group_size"]
        traj_instructions: dict[int, str] = {}
        for g in selected_groups:
            for i in range(group_size):
                b = g * group_size + i
                traj_instructions[b] = setup["all_tasks"][b]

        all_relabeled_traj_indices = list(traj_instructions.keys())
        eval_traj_indices = all_relabeled_traj_indices
        reward_coef = self.cfg.env.train.get("reward_coef", 1.0)

        rewards = rollout_batch["rewards"]
        original_rewards = {
            b: rewards[:, b, :].sum().item() for b in all_relabeled_traj_indices
        }

        # Assign random binary reward to each trajectory.
        vlm_rewards: dict[int, float] = {}
        for b in eval_traj_indices:
            vlm_rewards[b] = float(random.randint(0, 1))

        n_positive = sum(1 for r in vlm_rewards.values() if r > 0)
        n_all_zero = 0
        n_all_success = 0
        for g in selected_groups:
            group_traj_indices = range(g * group_size, (g + 1) * group_size)
            rewards_in_group = [
                vlm_rewards[b] for b in group_traj_indices if b in vlm_rewards
            ]
            if not rewards_in_group:
                continue
            if all(r == 0.0 for r in rewards_in_group):
                n_all_zero += 1
            if all(r > 0.0 for r in rewards_in_group):
                n_all_success += 1

        print(
            f"[RandomRewardProcessor rank={self._rank}] "
            f"{len(eval_traj_indices)} trajs assigned random rewards, "
            f"{n_positive} positive, "
            f"all_zero={n_all_zero}, all_success={n_all_success}",
            flush=True,
        )

        self.last_setup = setup

        return {
            "traj_instructions": traj_instructions,
            "fallback_trajs": set(),
            "vlm_rewards": vlm_rewards,
            "eval_traj_indices": eval_traj_indices,
            "all_relabeled_traj_indices": all_relabeled_traj_indices,
            "traj_cutoffs_map": {},
            "group_anchors": group_anchors,
            "group_cutoffs": {},
            "fallback_groups": set(),
            "reward_coef": reward_coef,
            "original_rewards": original_rewards,
            "setup": setup,
            "metrics": {
                "her_groups_with_anchor": len(selected_groups),
                "her_groups_skipped": setup["n_groups"] - len(selected_groups),
                "her_eval_failures": 0,
                "her_positive_rate": n_positive / max(len(eval_traj_indices), 1),
                "her_relabeled_all_zero": n_all_zero,
                "her_relabeled_all_success": n_all_success,
                "her_subtraj_groups": 0,
                "her_filtered_uninteresting": 0,
                "her_skipped_same_instruction": 0,
            },
        }

    def _apply_relabeling_to_batch(
        self,
        rollout_batch: dict[str, Any],
        setup: dict[str, Any],
        her_result: dict[str, Any],
    ) -> None:
        if not her_result["all_relabeled_traj_indices"]:
            return

        rewards = rollout_batch["rewards"]
        eval_traj_indices = her_result["eval_traj_indices"]
        vlm_rewards = her_result["vlm_rewards"]
        reward_coef = her_result["reward_coef"]
        original_rewards = her_result["original_rewards"]
        all_relabeled_traj_indices = her_result["all_relabeled_traj_indices"]

        print(
            f"[RandomRewardProcessor rank={self._rank}] "
            f"patching rewards ({len(eval_traj_indices)} trajs), "
            f"skipping prompt patch.",
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
            traj_cutoffs=None,
            dones=rollout_batch.get("dones"),
            keep_success_reward=keep_success_reward,
        )

        her_result["metrics"]["trajectory_log"] = self._build_trajectory_log(
            setup,
            all_relabeled_traj_indices,
            her_result["traj_instructions"],
            original_rewards,
            rewards,
            traj_cutoffs_map=None,
        )
