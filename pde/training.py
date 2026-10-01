"""Prompt sampling and canonical-prompt anchoring for PDE training."""

from __future__ import annotations

import random
from typing import TypeVar

from pde.prompt_pool import PromptPool

TensorLike = TypeVar("TensorLike")


def canonical_sampling_probability(
    canonical_success_ema: float,
    consolidation_target: float = 0.5,
    minimum: float = 0.05,
) -> float:
    """Compute the paper's adaptive canonical-prompt mixture coefficient."""
    if consolidation_target <= 0:
        raise ValueError("consolidation_target must be positive")
    if not 0 <= minimum <= 1:
        raise ValueError("minimum must be in [0, 1]")
    return max(minimum, min(1.0, canonical_success_ema / consolidation_target))


def sample_rollout_prompt(
    pool: PromptPool,
    canonical_probability: float,
    rng: random.Random | None = None,
) -> str:
    """Sample the canonical prompt or a uniformly chosen admitted prompt."""
    if not 0 <= canonical_probability <= 1:
        raise ValueError("canonical_probability must be in [0, 1]")
    admitted = pool.admitted
    if not admitted:
        return pool.canonical_prompt
    rng = rng or random
    if rng.random() < canonical_probability:
        return pool.canonical_prompt
    return rng.choice(admitted).prompt


def mixed_log_probability(
    exploratory_log_probability: TensorLike,
    canonical_log_probability: TensorLike,
) -> TensorLike:
    """Return the geometric-mean log likelihood used in mixed backpropagation.

    Gradients flow through both arguments. The caller forms the PPO ratio by
    subtracting the stored old log probability under the rollout prompt.
    """
    return (exploratory_log_probability + canonical_log_probability) * 0.5
