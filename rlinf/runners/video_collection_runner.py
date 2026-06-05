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

"""VideoCollectionRunner: drives env + rollout to collect trajectory videos.

No actor model or training involved — just rollout generation and video saving.
"""

import typing

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle

if typing.TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig

    from rlinf.workers.actor.video_collector_worker import VideoCollectorWorker
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class VideoCollectionRunner:
    """Runner that collects trajectory videos without any training."""

    def __init__(
        self,
        cfg: "DictConfig",
        collector: "VideoCollectorWorker",
        rollout: "MultiStepRolloutWorker",
        env: "EnvWorker",
    ):
        self.cfg = cfg
        self.collector = collector
        self.rollout = rollout
        self.env = env

        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")
        self.actor_channel = Channel.create("Actor")

        vc_cfg = cfg.get("video_collection", {})
        self.num_steps = int(vc_cfg.get("num_steps", 1))

    def init_workers(self) -> None:
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()
        self.collector.init_worker().wait()

    def run(self) -> None:
        for step in range(self.num_steps):
            self.collector.set_global_step(step)
            self.rollout.set_global_step(step)

            print(f"[VideoCollection] step {step + 1}/{self.num_steps}: running rollout", flush=True)
            env_handle: Handle = self.env.interact(
                input_channel=self.rollout_channel,
                rollout_channel=self.env_channel,
                reward_channel=None,
                actor_channel=self.actor_channel,
            )
            rollout_handle: Handle = self.rollout.generate(
                input_channel=self.env_channel,
                output_channel=self.rollout_channel,
            )
            self.collector.recv_rollout_trajectories(
                input_channel=self.actor_channel
            ).wait()
            rollout_handle.wait()
            env_handle.wait()

            print(f"[VideoCollection] step {step + 1}/{self.num_steps}: saving videos", flush=True)
            stats_list = self.collector.save_videos(step=step).wait()
            if stats_list:
                total = sum(s.get("videos_saved", 0) for s in stats_list if s)
                print(f"[VideoCollection] step {step + 1}: {total} videos saved", flush=True)

        print(f"[VideoCollection] Done. Videos written to: {self.cfg.video_collection.output_dir}", flush=True)
