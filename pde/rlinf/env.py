"""PDE environment extensions for RLinf."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rlinf.envs.libero.libero_env import LiberoEnv
from rlinf.workers.env.env_worker import EnvWorker

from pde.rlinf.data import PDERolloutResult


_BDDL_SECTION = re.compile(r"\(:(?:fixtures|objects)(.*?)\)", re.DOTALL)


def _objects_in_bddl(path: str) -> list[str]:
    """Read fixture and object identifiers from a LIBERO BDDL task."""
    text = Path(path).read_text()
    return [
        parts[0]
        for section in _BDDL_SECTION.findall(text)
        for line in section.splitlines()
        if len(parts := line.strip().split()) >= 3 and parts[1] == "-"
    ]


class PDELiberoEnv(LiberoEnv):
    """RLinf's LIBERO environment with scene metadata for VLM prompting."""

    def get_env_fn_params(self, env_idx=None):
        params = super().get_env_fn_params(env_idx)
        parsed = [_objects_in_bddl(item["bddl_file_name"]) for item in params]
        if env_idx is None or not hasattr(self, "scene_objects"):
            self.scene_objects = parsed
        else:
            for index, env_id in enumerate(env_idx):
                self.scene_objects[env_id] = parsed[index]
        return params

    def _wrap_obs(self, obs_list):
        observations = super()._wrap_obs(obs_list)
        observations["scene_objects"] = self.scene_objects
        return observations


class PDEEnvWorker(EnvWorker):
    """RLinf environment worker that transports PDE language metadata."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._pde_epoch_metadata: list[list[dict[str, Any]]] = []

    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage: int):
        if issubclass(env_cls, LiberoEnv):
            env_cls = PDELiberoEnv
        return super()._setup_env_and_wrappers(env_cls, env_cfg, num_envs_per_stage)

    def bootstrap_step(self, *args, **kwargs):
        outputs = super().bootstrap_step(*args, **kwargs)
        if not self._pde_epoch_metadata:
            self._pde_epoch_metadata = [[] for _ in outputs]
        for stage, output in enumerate(outputs):
            self._pde_epoch_metadata[stage].append(
                {
                    "task_descriptions": list(
                        output.obs.get("task_descriptions") or []
                    ),
                    "scene_objects": list(output.obs.get("scene_objects") or []),
                }
            )
        return outputs

    async def _run_interact_once(self, *args, **kwargs):
        """Use the PDE rollout subclass while retaining RLinf's interaction loop."""
        import rlinf.workers.env.env_worker as env_worker_module

        self._pde_epoch_metadata = []
        original = env_worker_module.EmbodiedRolloutResult
        env_worker_module.EmbodiedRolloutResult = PDERolloutResult
        try:
            return await super()._run_interact_once(*args, **kwargs)
        finally:
            env_worker_module.EmbodiedRolloutResult = original

    async def send_rollout_trajectories(self, rollout_result, channel):
        """Attach env-major metadata before RLinf splits and sends trajectories."""
        stage = next(
            index
            for index, candidate in enumerate(self.rollout_results)
            if candidate is rollout_result
        )
        epochs = self._pde_epoch_metadata[stage]
        rollout_result.task_descriptions = _env_major(epochs, "task_descriptions")
        rollout_result.scene_objects = _env_major(epochs, "scene_objects")
        await super().send_rollout_trajectories(rollout_result, channel)


def _env_major(epochs: list[dict[str, Any]], key: str) -> list[Any] | None:
    values = [epoch[key] for epoch in epochs if epoch[key]]
    if not values:
        return None
    return [value[env] for env in range(len(values[0])) for value in values]
