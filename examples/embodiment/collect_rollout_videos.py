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

"""Collect rollout trajectory videos for offline VLM prompt testing.

Runs env + rollout workers (no actor training) for ``video_collection.num_steps``
steps and saves trajectory videos to ``video_collection.output_dir``.

Usage::

    python examples/embodiment/collect_rollout_videos.py \\
        --config-path ../../config/libero10/openvlaoft \\
        --config-name libero10_openvlaoft_her_sft \\
        video_collection.output_dir=./collected_videos \\
        video_collection.num_steps=3 \\
        video_collection.max_videos_per_step=16

Each step produces a sub-directory ``step_NNNNN/`` containing:
  - ``traj_NNNN.mp4``  — trajectory video (preprocessed frames, same as HER-SFT VLM input)
  - ``metadata.json``  — per-trajectory original instructions and total rewards
"""

import json

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, open_dict

from rlinf.config import validate_cfg
from rlinf.runners.video_collection_runner import VideoCollectionRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.video_collector_worker import VideoCollectorWorker
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)

_VC_DEFAULTS = {
    "output_dir": "./collected_videos",
    "num_steps": 1,
    "max_videos_per_step": 0,
    "video_fps": 15,
}


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    # Inject video_collection defaults before validate_cfg locks the struct.
    with open_dict(cfg):
        if not OmegaConf.select(cfg, "video_collection"):
            cfg.video_collection = {}
        for k, v in _VC_DEFAULTS.items():
            if OmegaConf.select(cfg, f"video_collection.{k}") is None:
                cfg.video_collection[k] = v

    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # VideoCollectorWorker takes the actor placement slot (CPU-only, no model).
    actor_placement = component_placement.get_strategy("actor")
    collector_group = VideoCollectorWorker.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    runner = VideoCollectionRunner(
        cfg=cfg,
        collector=collector_group,
        rollout=rollout_group,
        env=env_group,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
