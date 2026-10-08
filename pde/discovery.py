"""Framework-independent prompt-posterior update loop from PDE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from pde.prompt_pool import PromptCandidate, PromptPool


@dataclass(frozen=True)
class DiscoveryConfig:
    """Prompt discovery settings used by the VLA experiments."""

    iterations: int = 10
    candidates_per_iteration: int = 5
    rollouts_per_candidate: int = 10
    admission_threshold: float = 0.0

    def __post_init__(self) -> None:
        if (
            min(
                self.iterations,
                self.candidates_per_iteration,
                self.rollouts_per_candidate,
            )
            <= 0
        ):
            raise ValueError("discovery counts must be positive")
        if not 0 <= self.admission_threshold <= 1:
            raise ValueError("admission_threshold must be in [0, 1]")


@dataclass(frozen=True)
class Evaluation:
    """Aggregate result of executing one prompt for several episodes."""

    rewards: Sequence[float]
    summary: str


class RolloutEvaluator(Protocol):
    """Adapter implemented by a simulator or robot rollout stack."""

    def __call__(self, task_id: str, prompt: str, episodes: int) -> Evaluation: ...


class PromptSupervisor(Protocol):
    """Adapter implemented by a VLM prompt sampler."""

    def __call__(
        self,
        pool: PromptPool,
        candidates: int,
    ) -> Sequence[str]: ...


def discover_prompt_pool(
    pool: PromptPool,
    evaluator: RolloutEvaluator,
    supervisor: PromptSupervisor,
    config: DiscoveryConfig | None = None,
) -> PromptPool:
    """Refine prompts using rollout feedback while policy weights stay frozen.

    The evaluator owns environment execution and video capture. The supervisor
    owns VLM inference and can inspect ``pool.candidates`` for both positive and
    negative history. This function owns budgeting, de-duplication, aggregation,
    and the public artifact contract.
    """
    config = config or DiscoveryConfig()
    if pool.candidates:
        raise ValueError("discovery expects an empty pool; load/resume externally")
    pool.admission_threshold = config.admission_threshold

    seen = {pool.canonical_prompt}
    for iteration in range(config.iterations):
        proposals = list(supervisor(pool, config.candidates_per_iteration))
        unique = []
        for proposal in proposals:
            normalized = proposal.strip()
            if normalized and normalized not in seen:
                unique.append(normalized)
                seen.add(normalized)
            if len(unique) == config.candidates_per_iteration:
                break
        if not unique:
            break

        for prompt in unique:
            result = evaluator(
                pool.task_id,
                prompt,
                config.rollouts_per_candidate,
            )
            if len(result.rewards) != config.rollouts_per_candidate:
                raise ValueError(
                    "evaluator returned an unexpected number of rewards: "
                    f"{len(result.rewards)}"
                )
            successes = sum(float(reward) > 0 for reward in result.rewards)
            pool.add(
                PromptCandidate(
                    prompt=prompt,
                    successes=successes,
                    rollouts=config.rollouts_per_candidate,
                    summary=result.summary.strip(),
                    iteration=iteration,
                )
            )
    return pool
