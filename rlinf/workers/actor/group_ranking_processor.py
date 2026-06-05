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

"""GroupRankingProcessor: VPR-style HER that ranks trajectories instead of relabeling instructions."""

import re
from typing import Any, Callable

import backoff
import openai
from omegaconf import DictConfig

from rlinf.workers.actor.her_processor import HERProcessor
from rlinf.workers.actor.her_utils import (
    _pixel_tensors_to_frame_base64s,
    select_frames_for_vlm,
)

# Frames sent to the VLM per trajectory in a group ranking call.
# Kept low so N trajectories together stay within context limits.
_RANKING_MAX_FRAMES_PER_TRAJ = 8

RANKING_PROMPT = (
    "You are comparing {n} robot manipulation trajectories. "
    'All trajectories attempt the same task: "{task}"\n\n'
    "Each trajectory is shown as a sequence of frames labeled "
    "[Trajectory 1], [Trajectory 2], etc.\n\n"
    "Rank these trajectories from MOST to LEAST progress toward completing the task.\n"
    "Progress stages (earlier = less progress):\n"
    "  1. No movement toward the target\n"
    "  2. Approaching the target object\n"
    "  3. Making contact with the target object\n"
    "  4. Grasping the target object\n"
    "  5. Transporting the object toward the goal\n"
    "  6. Placing the object at the goal (task complete)\n\n"
    "Ties are allowed: if two trajectories reached the same stage, "
    "assign them the same rank.\n\n"
    "Output format (strict):\n"
    "<thought>For each trajectory, state which stage it reached and why.</thought>\n"
    "<ranking>comma-separated trajectory numbers from best to worst; "
    "use = for ties, e.g. 2,1=3,4</ranking>"
)


def _parse_ranking(raw: str, n: int) -> list[list[int]] | None:
    """Parse a <ranking> tag into grouped 0-based indices (ties via ``=``).

    Returns None if the tag is missing or values don't form a valid
    partition of [1..n].  Example: ``"2,1=3,4"`` with n=4 →
    ``[[1], [0, 2], [3]]`` (0-based).
    """
    matched = re.search(
        r"<ranking>\s*([\d,=\s]+)\s*</ranking>", raw, flags=re.DOTALL
    )
    if not matched:
        return None
    try:
        groups: list[list[int]] = []
        for element in matched.group(1).split(","):
            element = element.strip()
            if not element:
                continue
            group = [int(x.strip()) for x in element.split("=") if x.strip()]
            groups.append([v - 1 for v in group])
    except ValueError:
        return None
    seen: set[int] = set()
    for group in groups:
        for idx in group:
            if idx < 0 or idx >= n or idx in seen:
                return None
            seen.add(idx)
    if len(seen) != n:
        return None
    return groups


def _ranking_to_rewards(groups: list[list[int]]) -> dict[int, float]:
    """Assign linearly spaced rewards in [0, 1] based on rank position.

    groups: list of tied-index groups ordered best→worst, e.g.
        ``[[1], [0, 2], [3]]``.  Tied trajectories share the average
        reward of their occupied positions.
    Returns: {traj_local_idx -> normalized_reward}.
    """
    n = sum(len(g) for g in groups)
    if n == 1:
        return {groups[0][0]: 1.0}
    rewards: dict[int, float] = {}
    pos = 0
    for group in groups:
        avg_reward = sum(1.0 - (pos + k) / (n - 1) for k in range(len(group))) / len(
            group
        )
        for idx in group:
            rewards[idx] = avg_reward
        pos += len(group)
    return rewards


class GroupRankingProcessor(HERProcessor):
    """HER variant that ranks trajectories by visual progress instead of relabeling instructions.

    For each all-fail group, sends all trajectory clips to the VLM in a single
    call and asks it to rank them by progress toward the original task. Rewards
    are assigned proportionally to rank (best → reward_coef, worst → 0).

    No instruction relabeling. No prompt retokenization. Drop-in replacement
    for HERProcessor: same group selection logic, same _patch_rewards call.
    """

    def __init__(
        self,
        cfg: DictConfig,
        rank: int,
        get_model_fn: Callable[[], Any],
        log_warning_fn: Callable[[str], None],
    ) -> None:
        super().__init__(cfg, rank, get_model_fn, log_warning_fn)
        self._ranking_max_frames = cfg.algorithm.get(
            "ranking_max_frames_per_traj", _RANKING_MAX_FRAMES_PER_TRAJ
        )
        from openai import OpenAI

        self._ranking_client = OpenAI(
            api_key=self._vlm.api_key,
            base_url=self._vlm.endpoint or None,
            timeout=600,
        )

    # ── Batch setup (tokenizer-free override) ────────────────────────────────

    def _setup_her_batch(self, rollout_batch: dict[str, Any]) -> dict[str, Any]:
        """Minimal setup that skips the tokenizer — GroupRankingProcessor never patches prompts."""
        forward_inputs = rollout_batch["forward_inputs"]
        pixel_values = forward_inputs.get("observation/image")
        if pixel_values is None:
            pixel_values = forward_inputs.get("pixel_values")
        if pixel_values is None:
            raise RuntimeError(
                "[ranking] No pixel_values found in forward_inputs. "
                "Expected 'observation/image' or 'pixel_values'."
            )

        # Derive T, B from pixel_values shape (time-major: [T, B, ...]).
        T, B = pixel_values.shape[:2]

        task_descriptions = rollout_batch.get("task_descriptions")
        if not task_descriptions or len(task_descriptions) < B:
            raise RuntimeError(
                f"[ranking] Missing task_descriptions (got "
                f"{len(task_descriptions) if task_descriptions else 0}, expected {B})"
            )
        all_tasks = [str(task_descriptions[b]).strip() for b in range(B)]

        group_size = self.cfg.algorithm.get("group_size", 8)
        assert B % group_size == 0, f"batch {B} not divisible by group_size {group_size}"
        n_groups = B // group_size

        return {
            "is_openpi": False,
            "tokenizer": None,
            "pixel_values": pixel_values,
            "T": T,
            "B": B,
            "seq_len": None,
            "pad_id": None,
            "all_tasks": all_tasks,
            "all_scene_objects": rollout_batch.get("scene_objects"),
            "group_size": group_size,
            "n_groups": n_groups,
        }

    # ── VLM ranking call ──────────────────────────────────────────────────────

    def _build_ranking_content(
        self, all_traj_frames: list[list], task: str
    ) -> list[dict]:
        """Build interleaved multimodal content for ranking N trajectories.

        Sends frames in 'frames' mode (JPEG). Each trajectory is preceded by a
        text label [Trajectory i] so the VLM can reference them by number.
        The ranking prompt is appended at the end.
        """
        content: list[dict] = []
        for i, frames in enumerate(all_traj_frames):
            selected = select_frames_for_vlm(frames, self._ranking_max_frames)
            content.append({"type": "text", "text": f"[Trajectory {i + 1}]"})
            for fb64 in _pixel_tensors_to_frame_base64s(
                selected, flip_horizontal=self._vlm.flip_horizontal
            ):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{fb64}"},
                    }
                )
        content.append(
            {
                "type": "text",
                "text": RANKING_PROMPT.format(n=len(all_traj_frames), task=task),
            }
        )
        return content

    @backoff.on_exception(
        backoff.expo,
        (openai.APIConnectionError, openai.InternalServerError, openai.RateLimitError),
        max_time=300,
        on_backoff=lambda d: d["args"][0]._vlm.log_warning(
            f"[ranking] API error on attempt {d['tries']}, retrying: {d['exception']}"
        ),
    )
    def request_group_ranking(
        self, all_traj_frames: list[list], task: str
    ) -> list[list[int]]:
        """Send all trajectory frames to the VLM and return a ranking.

        Returns:
            ranking: grouped 0-based indices ordered best→worst.
                     e.g. [[1], [0, 2], [3]] means traj 1 best, 0 & 2 tied, 3 worst.

        Retries until the VLM returns a valid <ranking> tag.
        """
        n = len(all_traj_frames)
        content = self._build_ranking_content(all_traj_frames, task)
        # Ranking always sends JPEG frames with a <thought>-tagged prompt.
        extra_body = {"chat_template_kwargs": {"enable_thinking": True}}

        attempt = 0
        while True:
            attempt += 1
            try:
                response = self._ranking_client.chat.completions.create(
                    model=self._vlm.model,
                    messages=[{"role": "user", "content": content}],  # type: ignore[arg-type]
                    max_tokens=8192,
                    extra_body=extra_body,
                )
                raw = response.choices[0].message.content
                if isinstance(raw, list):
                    raw = "\n".join(
                        str(item.get("text", ""))
                        for item in raw
                        if isinstance(item, dict) and item.get("type") == "text"
                    )
                raw = str(raw).strip()
            except RuntimeError as e:
                self._vlm.log_warning(
                    f"[ranking] attempt {attempt}: VLM call failed ({e}), retrying"
                )
                continue
            except openai.APIStatusError as e:
                if isinstance(e, (openai.RateLimitError, openai.InternalServerError)):
                    raise
                raise RuntimeError(
                    f"Ranking VLM request failed: "
                    f"{type(e).__name__}(status={e.status_code}): {e.message}"
                ) from None

            ranking = _parse_ranking(raw, n)
            if ranking is None:
                self._vlm.log_warning(
                    f"[ranking] attempt {attempt}: invalid <ranking> tag, retrying. "
                    f"Raw (first 300 chars): {raw[:300]!r}"
                )
                continue

            flat = [idx + 1 for group in ranking for idx in group]
            print(
                f"[ranking] parsed ranking (1-based best→worst): {flat}",
                flush=True,
            )
            return ranking

    # ── Core ranking pipeline ─────────────────────────────────────────────────

    def _compute_group_ranking(
        self,
        rollout_batch: dict[str, Any],
        setup: dict[str, Any],
        selected_groups: list[int],
    ) -> dict[str, Any]:
        """Rank trajectories within each selected group and compute rewards.

        Returns a result dict compatible with __call__'s apply step.
        traj_instructions is always empty — no instruction relabeling happens.
        """
        pixel_values = setup["pixel_values"]  # [T, B, ...]
        all_tasks = setup["all_tasks"]
        group_size = setup["group_size"]
        n_groups = setup["n_groups"]
        reward_coef = self.cfg.env.train.get("reward_coef", 1.0)

        vlm_rewards: dict[int, float] = {}
        n_ranked = 0
        n_failed = 0

        for g in selected_groups:
            group_start = g * group_size
            local_indices = list(range(group_size))
            global_indices = [group_start + i for i in local_indices]
            task = all_tasks[group_start]

            all_traj_frames = [
                self._get_traj_frames(pixel_values, b) for b in global_indices
            ]

            try:
                # ranking: 0-based local indices within the group, best to worst
                ranking = self.request_group_ranking(all_traj_frames, task)
            except Exception as e:
                self._vlm.log_warning(
                    f"[ranking] group {g}: ranking call failed ({e}), skipping"
                )
                n_failed += 1
                continue

            # normalized rewards in [0, 1]; _patch_rewards scales by reward_coef
            local_rewards = _ranking_to_rewards(ranking)
            for local_idx, r in local_rewards.items():
                vlm_rewards[group_start + local_idx] = r

            n_ranked += 1
            flat_ranking = [idx + 1 for group in ranking for idx in group]
            print(
                f"[ranking] group {g} ('{task}'): "
                f"ranking={flat_ranking} "
                f"rewards={[round(local_rewards.get(i, 0.0), 2) for i in local_indices]}",
                flush=True,
            )

        patched = list(vlm_rewards.keys())
        print(
            f"[ranking] {n_ranked}/{len(selected_groups)} groups ranked "
            f"({n_failed} failed), {len(patched)} trajs patched",
            flush=True,
        )

        return {
            # reward fields consumed by __call__
            "vlm_rewards": vlm_rewards,
            "patched_traj_indices": patched,
            "reward_coef": reward_coef,
            # empty: no instruction relabeling
            "traj_instructions": {},
            "fallback_trajs": set(),
            "eval_traj_indices": patched,
            "all_relabeled_traj_indices": patched,
            "traj_cutoffs_map": {},
            "group_cutoffs": {},
            "fallback_groups": set(),
            "setup": setup,
            "metrics": {
                "ranking_groups_selected": len(selected_groups),
                "ranking_groups_skipped": n_groups - len(selected_groups),
                "ranking_groups_ranked": n_ranked,
                "ranking_groups_failed": n_failed,
                "ranking_trajs_patched": len(patched),
            },
        }

    # ── Public entry point ────────────────────────────────────────────────────

    def __call__(
        self, rollout_batch: dict, apply_to_batch: bool = True
    ) -> dict[str, Any]:
        """Run group-ranking pipeline.

        Reuses parent's _setup_her_batch and _filter_groups (same selection
        gate), then ranks all trajectories in each group via a single VLM
        call. No anchor picking, no prompt patching.
        """
        assert self.her_endpoint, "HER endpoint not set"

        print(
            f"[GroupRankingProcessor rank={self._rank}] Stage 1/3: setup batch.",
            flush=True,
        )
        setup = self._setup_her_batch(rollout_batch)
        B = setup["B"]
        per_traj_reward = (
            rollout_batch["rewards"].transpose(0, 1).reshape(B, -1).sum(dim=-1)
        )

        print(
            f"[GroupRankingProcessor rank={self._rank}] "
            f"Stage 2/3: selecting groups (B={B}, n_groups={setup['n_groups']}).",
            flush=True,
        )
        selected_groups = self._filter_groups(
            per_traj_reward, setup["group_size"], setup["n_groups"]
        )

        print(
            f"[GroupRankingProcessor rank={self._rank}] "
            f"Stage 3/3: ranking {len(selected_groups)} groups.",
            flush=True,
        )
        result = self._compute_group_ranking(
            rollout_batch, setup, selected_groups
        )

        if apply_to_batch and result["patched_traj_indices"]:
            self._patch_rewards(
                rollout_batch["rewards"],
                result["patched_traj_indices"],
                result["vlm_rewards"],
                result["reward_coef"],
            )

        return result
