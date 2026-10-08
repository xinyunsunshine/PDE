"""Launch paper-style PDE or its PPO baseline on RLinf."""

import json

import hydra
from omegaconf import OmegaConf

from pde.configuration import configure_rlinf, validate_pde

RLINF_ROOT = configure_rlinf()


@hydra.main(version_base="1.1", config_path="configs", config_name="libero")
def main(cfg):
    from pde.provenance import verify_rlinf, write_manifest

    verify_rlinf(RLINF_ROOT)
    import ray
    import torch.multiprocessing as mp
    from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
    from rlinf.config import validate_cfg
    from rlinf.runners.embodied_runner import EmbodiedRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    from pde.actor import PDEActor
    from pde.env import PDEEnvWorker
    from pde.rollout import PDERolloutWorker
    from pde.runner import PDERunner

    mp.set_start_method("spawn", force=True)
    validate_pde(cfg)
    if cfg.pde.enabled:
        from pde.pools import load_pools

        load_pools(cfg.pde.pool_dir, cfg.actor.model.model_path)
    cfg = validate_cfg(cfg)
    write_manifest(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))
    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    try:
        placement = HybridComponentPlacement(cfg, cluster)
        actor_cls = PDEActor if cfg.pde.enabled else EmbodiedFSDPActor
        rollout_cls = PDERolloutWorker if cfg.pde.enabled else MultiStepRolloutWorker
        runner_cls = PDERunner if cfg.pde.enabled else EmbodiedRunner

        def launch(cls, component):
            return cls.create_group(cfg).launch(
                cluster,
                name=cfg[component].group_name,
                placement_strategy=placement.get_strategy(component),
            )

        if cfg.runner.only_eval:
            if cfg.runner.get("resume_dir"):
                raise ValueError(
                    "Evaluation uses runner.ckpt_path, not a training resume_dir"
                )
            runner = EmbodiedEvalRunner(
                cfg=cfg,
                rollout=launch(MultiStepRolloutWorker, "rollout"),
                env=launch(PDEEnvWorker, "env"),
            )
        else:
            runner = runner_cls(
                cfg=cfg,
                actor=launch(actor_cls, "actor"),
                rollout=launch(rollout_cls, "rollout"),
                env=launch(PDEEnvWorker, "env"),
                reward=None,
            )
        runner.init_workers()
        runner.run()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
