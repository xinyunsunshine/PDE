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

"""VideoCollectorWorker: lightweight CPU-only worker that receives rollout
trajectories and saves them as mp4 videos for offline VLM prompt testing.

Replaces the actor in the rollout pipeline — no GPU or model required.
The pixel tensors are decoded with the same logic used by HER-SFT so the
saved videos are identical to what the relabeling VLM would see.
"""

import json
import os

from omegaconf import DictConfig

from rlinf.data.embodied_io_struct import convert_trajectories_to_batch
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.fsdp_actor_worker import process_nested_dict_for_adv
from rlinf.workers.actor.her_utils import _save_trajectory_video


class VideoCollectorWorker(Worker):
    """CPU-only actor replacement that saves rollout trajectory videos to disk.

    Config keys under ``cfg.video_collection``:
      output_dir (str): Directory to write videos and metadata.
      max_videos_per_step (int, default 0): Max trajectories to save per
        rollout step.  0 = save all.
      video_fps (int, default 15): FPS for the saved mp4 files.
    """

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        self.cfg = cfg
        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        self.stage_num = cfg.rollout.pipeline_stage_num
        self._flip_video_horizontal = cfg.algorithm.get("flip_video_horizontal", False)

        vc_cfg = cfg.get("video_collection", {})
        self.output_dir = vc_cfg.get("output_dir", "./collected_videos")
        self.max_videos_per_step = int(vc_cfg.get("max_videos_per_step", 0))
        self.video_fps = int(vc_cfg.get("video_fps", 15))

        self.rollout_batch = None

    def init_worker(self) -> None:
        if self._rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)
            self.log_info(f"[VideoCollector] output_dir={self.output_dir}")

    def set_global_step(self, step: int) -> None:
        self._global_step = step

    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """Receive rollout trajectories from the env worker (same protocol as actor)."""
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        batch = convert_trajectories_to_batch(recv_list)

        # Re-shape forward_inputs into [T, B, ...] matching HER-SFT conventions.
        rollout_epoch = self.cfg.algorithm.rollout_epoch
        task_descs = batch.pop("task_descriptions", None)
        scene_objs = batch.pop("scene_objects", None)
        batch = process_nested_dict_for_adv(batch, rollout_epoch)
        if task_descs is not None:
            bsz = len(task_descs) // rollout_epoch
            batch["task_descriptions"] = [
                task_descs[env_j * rollout_epoch + epoch_i]
                for epoch_i in range(rollout_epoch)
                for env_j in range(bsz)
            ]
        if scene_objs is not None:
            bsz = len(scene_objs) // rollout_epoch
            batch["scene_objects"] = [
                scene_objs[env_j * rollout_epoch + epoch_i]
                for epoch_i in range(rollout_epoch)
                for env_j in range(bsz)
            ]
        self.rollout_batch = batch

    def save_videos(self, step: int) -> dict:
        """Save trajectory videos to ``output_dir/step_{step}/``.

        Only rank 0 writes to disk to avoid duplicate files.
        Returns a stats dict with the number of videos saved.
        """
        if self.rollout_batch is None:
            return {"videos_saved": 0}

        if self._rank != 0:
            return {"videos_saved": 0}

        forward_inputs = self.rollout_batch.get("forward_inputs", {})
        pixel_values = forward_inputs.get("pixel_values")
        if pixel_values is None:
            pixel_values = forward_inputs.get("observation/image")

        if pixel_values is None:
            self.log_warning("[VideoCollector] No pixel_values in rollout_batch — skipping.")
            return {"videos_saved": 0}

        T, B = pixel_values.shape[:2]
        task_descs = self.rollout_batch.get("task_descriptions", None)
        scene_objs = self.rollout_batch.get("scene_objects", None)
        rewards = self.rollout_batch.get("rewards", None)  # [T, B, ...]

        step_dir = os.path.join(self.output_dir, f"step_{step:05d}")
        os.makedirs(step_dir, exist_ok=True)

        n_to_save = B if self.max_videos_per_step <= 0 else min(self.max_videos_per_step, B)
        metadata = []

        for b in range(n_to_save):
            frames = [pixel_values[t, b] for t in range(T)]
            video_path = os.path.join(step_dir, f"traj_{b:04d}.mp4")
            _save_trajectory_video(
                frames,
                fps=self.video_fps,
                flip_horizontal=self._flip_video_horizontal,
                out_path=video_path,
            )
            entry = {"traj": b, "video": os.path.relpath(video_path, self.output_dir)}
            if task_descs is not None and b < len(task_descs):
                entry["instruction"] = task_descs[b]
            if scene_objs is not None and b < len(scene_objs):
                entry["scene_objects"] = scene_objs[b]
            if rewards is not None:
                total_reward = float(rewards[:, b].reshape(-1).sum().item())
                entry["total_reward"] = total_reward
            metadata.append(entry)

        meta_path = os.path.join(step_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump({"step": step, "trajectories": metadata}, f, indent=2)

        self.log_info(f"[VideoCollector] step={step}: saved {n_to_save}/{B} videos → {step_dir}")
        return {"videos_saved": n_to_save}
