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

import json

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create actor worker group
    actor_placement = component_placement.get_strategy("actor")

    runner_cls = EmbodiedRunner

    if cfg.algorithm.loss_type == "embodied_sac":
        from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

        actor_worker_cls = EmbodiedSACFSDPPolicy
    elif cfg.algorithm.loss_type == "embodied_dagger":
        from rlinf.workers.actor.fsdp_dagger_policy_worker import (
            EmbodiedDAGGERFSDPPolicy,
        )

        actor_worker_cls = EmbodiedDAGGERFSDPPolicy
    elif cfg.algorithm.loss_type == "embodied_nft":
        from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy

        actor_worker_cls = EmbodiedNFTFSDPPolicy
    elif cfg.actor.get("her_contrastive", None) is not None:
        from rlinf.workers.actor.her_contrastive_actor_worker import (
            EmbodiedFSDPActorHERContrastive,
        )

        actor_worker_cls = EmbodiedFSDPActorHERContrastive
    elif cfg.actor.get("her_shuffled_dual", None) is not None:
        from rlinf.workers.actor.her_shuffled_dual_actor_worker import (
            EmbodiedFSDPActorHERShuffledDual,
        )

        actor_worker_cls = EmbodiedFSDPActorHERShuffledDual
    elif cfg.actor.get("her_dual_multi", None) is not None:
        from rlinf.workers.actor.her_dual_multi_actor_worker import (
            EmbodiedFSDPActorHERDualMulti,
        )

        actor_worker_cls = EmbodiedFSDPActorHERDualMulti
    elif cfg.actor.get("her_dual_cosine_grad", None) is not None:
        from rlinf.workers.actor.her_dual_cosine_grad_actor_worker import (
            EmbodiedFSDPActorHERDualCosineGrad,
        )

        actor_worker_cls = EmbodiedFSDPActorHERDualCosineGrad
    elif cfg.actor.get("her_dual_pcgrad", None) is not None:
        from rlinf.workers.actor.her_dual_pcgrad_actor_worker import (
            EmbodiedFSDPActorHERDualPCGrad,
        )

        actor_worker_cls = EmbodiedFSDPActorHERDualPCGrad
    elif cfg.actor.get("her_dual_rephrase_only", None) is not None:
        from rlinf.workers.actor.her_ablation_dual_actor_worker import (
            EmbodiedFSDPActorHERDualRephraseOnly,
        )

        actor_worker_cls = EmbodiedFSDPActorHERDualRephraseOnly
    elif cfg.actor.get("her_dual_reward_eval_only", None) is not None:
        from rlinf.workers.actor.her_ablation_dual_actor_worker import (
            EmbodiedFSDPActorHERDualRewardEvalOnly,
        )

        actor_worker_cls = EmbodiedFSDPActorHERDualRewardEvalOnly
    elif cfg.actor.get("her_dual_random_reward", None) is not None:
        from rlinf.workers.actor.her_ablation_dual_actor_worker import (
            EmbodiedFSDPActorHERDualRandomReward,
        )

        actor_worker_cls = EmbodiedFSDPActorHERDualRandomReward
    elif cfg.actor.get("her_dual_separate_grad", None) is not None:
        from rlinf.workers.actor.her_dual_separate_grad_actor_worker import (
            EmbodiedFSDPActorHERDualSeparateGrad,
        )

        actor_worker_cls = EmbodiedFSDPActorHERDualSeparateGrad
    elif (
        cfg.actor.get("her_dual", None) is not None
        or cfg.algorithm.get("modification_type", None) == "her_dual"
    ):
        from rlinf.workers.actor.her_dual_actor_worker import (
            EmbodiedFSDPActorHERDual,
        )

        actor_worker_cls = EmbodiedFSDPActorHERDual
    elif (
        cfg.actor.get("her", None) is not None
        or cfg.algorithm.get("modification_type", None) == "her"
        or cfg.algorithm.get("modification_type", None) == "group_ranking"
    ):
        from rlinf.workers.actor.her_embodied_actor_worker import EmbodiedFSDPActorHER

        actor_worker_cls = EmbodiedFSDPActorHER
    elif cfg.actor.get("her_sft", None) is not None:
        from rlinf.workers.actor.her_sft_actor_worker import EmbodiedFSDPActorHERSFT

        actor_worker_cls = EmbodiedFSDPActorHERSFT
    else:
        from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

        actor_worker_cls = EmbodiedFSDPActor
    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    # Create rollout worker group
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    # Create env worker group
    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    reward_group = None
    if cfg.get("reward", {}).get("use_reward_model", False) and not cfg.get(
        "reward", {}
    ).get("standalone_realworld", False):
        # Create reward worker group
        reward_placement = component_placement.get_strategy("reward")
        reward_group = EmbodiedRewardWorker.create_group(cfg).launch(
            cluster, name=cfg.reward.group_name, placement_strategy=reward_placement
        )

    runner = runner_cls(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
        reward=reward_group,
    )

    runner.init_workers()
    runner.run()
    cluster.shutdown()


if __name__ == "__main__":
    main()
