"""Launch PDE reinforcement learning on the pinned RLinf runtime."""

import json

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

from pde.rlinf.actor import PDEActor
from pde.rlinf.dual_actor import PDEDualActor
from pde.rlinf.env import PDEEnvWorker
from pde.rlinf.runner import PDERunner

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1",
    config_path="../../RLinf/examples/embodiment/config",
    config_name="libero_object_grpo_openpi_pi05",
)
def main(cfg) -> None:
    """Create RLinf worker groups with PDE-owned actor and environment subclasses."""
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster,
        distributed_log_dir=cfg.runner.per_worker_log_path,
    )
    placement = HybridComponentPlacement(cfg, cluster)
    actor_cls = (
        PDEDualActor
        if cfg.algorithm.get("pde_dual", True)
        else PDEActor
    )
    actor = actor_cls.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy("actor"),
    )
    rollout = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy("rollout"),
    )
    env = PDEEnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=placement.get_strategy("env"),
    )

    reward = None
    if cfg.get("reward", {}).get("use_reward_model", False):
        reward = EmbodiedRewardWorker.create_group(cfg).launch(
            cluster,
            name=cfg.reward.group_name,
            placement_strategy=placement.get_strategy("reward"),
        )

    runner = PDERunner(cfg=cfg, actor=actor, rollout=rollout, env=env, reward=reward)
    runner.init_workers()
    runner.run()
    cluster.shutdown()


if __name__ == "__main__":
    main()
