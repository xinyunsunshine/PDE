"""Versioned prompt-pool artifacts shared by discovery and RL training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PromptCandidate:
    """One prompt evaluated against a fixed VLA checkpoint."""

    prompt: str
    successes: int
    rollouts: int
    summary: str
    iteration: int

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("candidate prompt cannot be empty")
        if self.rollouts <= 0:
            raise ValueError("candidate rollouts must be positive")
        if not 0 <= self.successes <= self.rollouts:
            raise ValueError("candidate successes must be in [0, rollouts]")
        if self.iteration < 0:
            raise ValueError("candidate iteration cannot be negative")

    @property
    def success_rate(self) -> float:
        """Return empirical episode success rate."""
        return self.successes / self.rollouts


@dataclass
class PromptPool:
    """Prompt-search result for one task and one frozen policy checkpoint."""

    task_id: str
    canonical_prompt: str
    policy_checkpoint: str
    environment: str
    admission_threshold: float = 0.0
    candidates: list[PromptCandidate] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported prompt-pool schema {self.schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        if not self.task_id.strip():
            raise ValueError("task_id cannot be empty")
        if not self.canonical_prompt.strip():
            raise ValueError("canonical_prompt cannot be empty")
        if not 0 <= self.admission_threshold <= 1:
            raise ValueError("admission_threshold must be in [0, 1]")
        prompts = [candidate.prompt for candidate in self.candidates]
        if len(prompts) != len(set(prompts)):
            raise ValueError("candidate prompts must be unique")

    @property
    def admitted(self) -> list[PromptCandidate]:
        """Return candidates whose measured success exceeds the threshold."""
        return [
            candidate
            for candidate in self.candidates
            if candidate.success_rate > self.admission_threshold
        ]

    def add(self, candidate: PromptCandidate) -> None:
        """Append a newly evaluated candidate, rejecting duplicates."""
        if candidate.prompt in {item.prompt for item in self.candidates}:
            raise ValueError(f"duplicate candidate prompt: {candidate.prompt!r}")
        self.candidates.append(candidate)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the artifact with derived success rates for readability."""
        payload = asdict(self)
        payload["candidates"] = [
            {**asdict(candidate), "success_rate": candidate.success_rate}
            for candidate in self.candidates
        ]
        return payload

    def save(self, path: str | Path) -> None:
        """Write the pool as stable, human-readable JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def load_prompt_pool(path: str | Path) -> PromptPool:
    """Load and validate a prompt-pool artifact."""
    payload = json.loads(Path(path).read_text())
    raw_candidates = payload.pop("candidates", [])
    candidates = []
    for raw in raw_candidates:
        raw = dict(raw)
        raw.pop("success_rate", None)
        candidates.append(PromptCandidate(**raw))
    return PromptPool(candidates=candidates, **payload)
