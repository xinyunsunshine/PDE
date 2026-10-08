import random
from pathlib import Path

import pytest

from pde.discovery import DiscoveryConfig, Evaluation, discover_prompt_pool
from pde.prompt_pool import PromptPool, load_prompt_pool
from pde.training import (
    canonical_sampling_probability,
    mixed_log_probability,
    sample_rollout_prompt,
)


def _empty_pool() -> PromptPool:
    return PromptPool(
        task_id="task-0",
        canonical_prompt="close the microwave",
        policy_checkpoint="checkpoint",
        environment="LIBERO-90",
    )


def test_discovery_builds_admitted_pool(tmp_path):
    proposals = iter([["push the door", "ignore"], ["shut the appliance", "ignore"]])

    def supervisor(_pool, candidates):
        assert candidates == 2
        return next(proposals)

    def evaluator(_task_id, prompt, episodes):
        successes = 1 if prompt == "push the door" else 0
        return Evaluation([1.0] * successes + [0.0] * (episodes - successes), prompt)

    pool = discover_prompt_pool(
        _empty_pool(),
        evaluator,
        supervisor,
        DiscoveryConfig(
            iterations=2, candidates_per_iteration=2, rollouts_per_candidate=3
        ),
    )
    assert [candidate.prompt for candidate in pool.admitted] == ["push the door"]

    artifact = tmp_path / "pool.json"
    pool.save(artifact)
    assert load_prompt_pool(artifact) == pool


def test_prompt_sampling_and_schedule():
    assert canonical_sampling_probability(0.0) == pytest.approx(0.05)
    assert canonical_sampling_probability(0.25) == pytest.approx(0.5)
    assert canonical_sampling_probability(0.8) == pytest.approx(1.0)

    pool = _empty_pool()
    assert sample_rollout_prompt(pool, 0.0, random.Random(0)) == pool.canonical_prompt


def test_mixed_log_probability():
    assert mixed_log_probability(2.0, 4.0) == pytest.approx(3.0)


def test_rlinf_is_pinned_as_submodule():
    root = Path(__file__).resolve().parents[2]
    gitmodules = (root / ".gitmodules").read_text()
    assert "path = RLinf" in gitmodules
    assert not (root / "rlinf").exists()
