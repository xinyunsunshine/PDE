# Copyright 2026 The RLinf Authors.
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

"""Entry script for the bifranka real-world PPO loop (pi0.5 / OpenPI).

Mirrors `train_embodied_agent.py` but dispatches the rollout worker class
and runner based on `cfg.runner.runner_class` / `cfg.rollout.generation_backend`.

When `cfg.runner.runner_class == "realworld"`:
- Actor is `RealworldFSDPActor` (adds one-shot prev_logprobs recompute).
- Rollout is `RealworldRolloutWorker` (rsyncs HDF5 from bifranka, recomputes
  logprobs, pushes Trajectories into the output_channel which the runner
  aliases to actor_channel).
- Env is `_NoOpEnv` — the trainer never touches FR3 hardware. Episodes are
  produced by Ankile's pi0.5 inference server on bifranka.
- Runner is `RealworldEmbodiedRunner` (post-save scp of full_weights.pt
  + ssh heartbeat).

Sim training is unaffected — use `train_embodied_agent.py` for sim.
"""

import json

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker

mp.set_start_method("spawn", force=True)


class _NoOpHandle:
    """Quacks like a Ray Handle but does nothing."""

    def wait(self):
        return []

    def consume_durations(self, *args, **kwargs):
        # The runner may call this with `return_per_rank=True` (and/or other
        # kwargs) and expect a (durations, per_rank) tuple. We have no env
        # process, so return empty dicts of matching arity.
        if kwargs.get("return_per_rank") or (args and args[0] is True):
            return {}, {}
        return {}


class _NoOpEnv:
    """Drop-in for the EnvWorker group when running realworld.

    EmbodiedRunner calls these methods each iteration. All return _NoOpHandle
    so the runner's `.wait()` / `.consume_durations()` calls succeed without
    spawning a robot connection.
    """

    def __init__(self, group_name: str = "EnvGroup"):
        # Used by EmbodiedRunner._log_ranked_metrics to namespace per-rank
        # metric keys. The metrics list itself is empty (no env worker), so
        # the value is only consumed as a string prefix.
        self.worker_group_name = group_name

    def init_worker(self):                       return _NoOpHandle()
    def interact(self, *args, **kwargs):         return _NoOpHandle()
    def evaluate(self, *args, **kwargs):         return _NoOpHandle()
    def flush_videos(self):                      return _NoOpHandle()
    def set_video_step(self, *args, **kwargs):   return _NoOpHandle()
    def set_task_env_counts(self, *args, **kwargs): return _NoOpHandle()
    def set_global_step(self, *args, **kwargs):  return None


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="realworld_ppo_openpi_pi05",
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    runner_class = cfg.runner.get("runner_class", "embodied")
    backend = cfg.rollout.get("generation_backend", "huggingface")

    # ---- Actor worker -----------------------------------------------------
    actor_placement = component_placement.get_strategy("actor")
    if runner_class == "realworld":
        her_dual_enabled = (
            cfg.actor.get("her_dual", None) is not None
            or cfg.algorithm.get("modification_type", None) == "her_dual"
        )
        if her_dual_enabled:
            from rlinf.workers.actor.realworld_her_dual_actor_worker import (
                RealworldFSDPActorHERDual,
            )

            actor_worker_cls = RealworldFSDPActorHERDual
        else:
            from rlinf.workers.actor.fsdp_realworld_actor_worker import (
                RealworldFSDPActor,
            )

            actor_worker_cls = RealworldFSDPActor
    else:
        from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

        actor_worker_cls = EmbodiedFSDPActor
    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    # ---- Rollout worker — backend dispatcher ------------------------------
    rollout_placement = component_placement.get_strategy("rollout")
    if backend == "realworld":
        from rlinf.workers.rollout.realworld.realworld_worker import (
            RealworldRolloutWorker,
        )

        rollout_cls = RealworldRolloutWorker
    else:
        from rlinf.workers.rollout.hf.huggingface_worker import (
            MultiStepRolloutWorker,
        )

        rollout_cls = MultiStepRolloutWorker
    rollout_group = rollout_cls.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    # ---- Env worker -------------------------------------------------------
    # In realworld, the trainer does NOT drive any env (data comes from
    # bifranka HDF5). Spawning the real EnvWorker would try to connect FR3.
    if runner_class == "realworld":
        env_group = _NoOpEnv(group_name=cfg.env.group_name)
    else:
        env_placement = component_placement.get_strategy("env")
        env_group = EnvWorker.create_group(cfg).launch(
            cluster, name=cfg.env.group_name, placement_strategy=env_placement
        )

    # ---- Runner -----------------------------------------------------------
    if runner_class == "realworld":
        from rlinf.runners.realworld_embodied_runner import RealworldEmbodiedRunner

        runner_cls = RealworldEmbodiedRunner
    else:
        from rlinf.runners.embodied_runner import EmbodiedRunner

        runner_cls = EmbodiedRunner

    runner = runner_cls(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
    )

    runner.init_workers()
    runner.run()
    cluster.shutdown()


if __name__ == "__main__":
    main()
