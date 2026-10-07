"""PDE trajectory types layered on RLinf's embodied data model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rlinf.data.embodied_io_struct import (
    EmbodiedRolloutResult,
    Trajectory,
    convert_trajectories_to_batch as rlinf_convert_trajectories_to_batch,
)


@dataclass(kw_only=True)
class PDETrajectory(Trajectory):
    """An RLinf trajectory carrying the language context used by PDE."""

    task_descriptions: list[str] | None = None
    scene_objects: list[list[str]] | None = None


@dataclass(kw_only=True)
class PDERolloutResult(EmbodiedRolloutResult):
    """An RLinf rollout result that preserves per-trajectory language metadata."""

    task_descriptions: list[str] | None = None
    scene_objects: list[list[str]] | None = None

    def to_trajectory(self) -> PDETrajectory:
        base = super().to_trajectory()
        values = {
            name: getattr(base, name) for name in Trajectory.__dataclass_fields__
        }
        return PDETrajectory(
            **values,
            task_descriptions=self.task_descriptions,
            scene_objects=self.scene_objects,
        )

    def to_splited_trajectories(self, split_size: int) -> list[PDETrajectory]:
        trajectory = EmbodiedRolloutResult.to_trajectory(self)
        original_to_trajectory = self.to_trajectory
        self.to_trajectory = lambda: trajectory  # type: ignore[method-assign]
        try:
            chunks = super().to_splited_trajectories(split_size)
        finally:
            self.to_trajectory = original_to_trajectory  # type: ignore[method-assign]

        output: list[PDETrajectory] = []
        for index, chunk in enumerate(chunks):
            values = {
                name: getattr(chunk, name) for name in Trajectory.__dataclass_fields__
            }
            output.append(
                PDETrajectory(
                    **values,
                    task_descriptions=_slice(self.task_descriptions, index, split_size),
                    scene_objects=_slice(self.scene_objects, index, split_size),
                )
            )
        return output


def _slice(values: list[Any] | None, index: int, parts: int) -> list[Any] | None:
    if values is None:
        return None
    start = index * len(values) // parts
    end = (index + 1) * len(values) // parts
    return values[start:end]


def convert_trajectories_to_batch(
    trajectories: list[PDETrajectory],
) -> dict[str, Any]:
    """Use RLinf's tensor collation and append PDE's language fields."""
    batch = rlinf_convert_trajectories_to_batch(trajectories)
    for field in ("task_descriptions", "scene_objects"):
        values = [
            item
            for trajectory in trajectories
            for item in (getattr(trajectory, field, None) or [])
        ]
        if values:
            batch[field] = values
    return batch
