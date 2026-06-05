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

"""Unit tests for MIReward deduplication and InfoNCE logic (no VLM calls)."""

import numpy as np
import pytest

from rlinf.algorithms.rewards.mi import MIReward, infonce_rewards


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mi(**kwargs) -> MIReward:
    return MIReward(vllm_url="http://localhost:1", **kwargs)


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------


def test_dedup_grpo_group_size_4():
    """GRPO batch: 2 groups × 4 rollouts, 2 unique instructions."""
    instrs = [
        "pick up the cup",   # group 0, rollouts 0-3
        "pick up the cup",
        "pick up the cup",
        "pick up the cup",
        "open the drawer",   # group 1, rollouts 4-7
        "open the drawer",
        "open the drawer",
        "open the drawer",
    ]
    mi = _make_mi()

    seen: dict[str, int] = {}
    unique: list[str] = []
    for instr in instrs:
        if instr not in seen:
            seen[instr] = len(unique)
            unique.append(instr)
    intended = [seen[i] for i in instrs]

    assert unique == ["pick up the cup", "open the drawer"]
    assert intended == [0, 0, 0, 0, 1, 1, 1, 1]
    assert len(unique) == 2


def test_dedup_preserves_first_seen_order():
    """Unique instructions keep the order of first appearance."""
    instrs = ["c", "a", "c", "b", "a", "b"]
    seen: dict[str, int] = {}
    unique: list[str] = []
    for instr in instrs:
        if instr not in seen:
            seen[instr] = len(unique)
            unique.append(instr)
    assert unique == ["c", "a", "b"]


def test_dedup_all_unique():
    """No duplicates → unique list equals input."""
    instrs = ["a", "b", "c", "d"]
    seen: dict[str, int] = {}
    unique: list[str] = []
    for instr in instrs:
        if instr not in seen:
            seen[instr] = len(unique)
            unique.append(instr)
    assert unique == instrs
    assert [seen[i] for i in instrs] == [0, 1, 2, 3]


def test_dedup_all_same():
    """All rollouts share one instruction → M=1."""
    instrs = ["do the thing"] * 8
    seen: dict[str, int] = {}
    unique: list[str] = []
    for instr in instrs:
        if instr not in seen:
            seen[instr] = len(unique)
            unique.append(instr)
    assert unique == ["do the thing"]
    assert [seen[i] for i in instrs] == [0] * 8


# ---------------------------------------------------------------------------
# InfoNCE tests
# ---------------------------------------------------------------------------


def test_infonce_correct_rank1_gets_highest_reward():
    """Rollout whose intended instruction scores highest gets the best (least negative) reward."""
    # InfoNCE is always <= 0; reward is maximised (closest to 0) when intended ranks #1.
    score_matrix = np.array([[5.0, 1.0, 0.0]], dtype=np.float32)
    r_rank1 = infonce_rewards(score_matrix, intended=[0])[0]  # intended scores 5
    r_rank2 = infonce_rewards(score_matrix, intended=[1])[0]  # intended scores 1
    r_rank3 = infonce_rewards(score_matrix, intended=[2])[0]  # intended scores 0
    assert r_rank1 > r_rank2 > r_rank3, "reward should decrease as intended rank worsens"
    assert r_rank1 <= 0, "InfoNCE reward is always <= 0"


def test_infonce_correct_rank_last_gets_worst_reward():
    """Rollout whose intended instruction scores lowest gets the worst reward."""
    score_matrix = np.array([[0.0, 4.0, 5.0]], dtype=np.float32)
    rewards_last = infonce_rewards(score_matrix, intended=[0])
    rewards_first = infonce_rewards(score_matrix, intended=[2])
    assert rewards_first[0] > rewards_last[0]


def test_infonce_grpo_group_differentiates_rollouts():
    """Within a GRPO group (same instruction), better-matching rollouts get higher reward."""
    # 4 rollouts, same instruction (intended=0), 2 unique instructions
    # rollout 0 scores 5 on instr 0 (good match), rollout 3 scores 1 (bad match)
    score_matrix = np.array(
        [
            [5.0, 0.0],  # rollout 0: strong match
            [3.0, 1.0],  # rollout 1: moderate match
            [2.0, 2.0],  # rollout 2: ambiguous
            [1.0, 4.0],  # rollout 3: bad match (wrong instruction scores higher)
        ],
        dtype=np.float32,
    )
    intended = [0, 0, 0, 0]
    rewards = infonce_rewards(score_matrix, intended=intended)

    assert rewards[0] > rewards[1] > rewards[2] > rewards[3], (
        "rewards should decrease as match quality decreases"
    )


def test_infonce_temperature_scaling():
    """Higher temperature flattens reward differences."""
    score_matrix = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
    r_low_temp = infonce_rewards(score_matrix, intended=[0], temperature=0.1)
    r_high_temp = infonce_rewards(score_matrix, intended=[0], temperature=10.0)
    # With low temp, the gap between intended and others is amplified → reward closer to 0
    # With high temp, scores are squashed → reward closer to -log(M)
    assert r_low_temp[0] > r_high_temp[0]
