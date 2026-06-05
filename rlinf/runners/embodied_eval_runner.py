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

import random
import typing

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import (
    compute_evaluate_metrics,
    compute_per_instruction_metrics,
)

if typing.TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig

    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedEvalRunner:
    def __init__(
        self,
        cfg: "DictConfig",
        rollout: "MultiStepRolloutWorker",
        env: "EnvWorker",
        run_timer=None,
    ):
        self.cfg = cfg
        self.rollout = rollout
        self.env = env

        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

        self.logger = get_logger()

    def init_workers(self):
        rollout_handle = self.rollout.init_worker()
        env_handle = self.env.init_worker()

        rollout_handle.wait()
        env_handle.wait()

    def evaluate(self):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = []
        all_video_logs: list[dict] = []
        for result in env_results:
            if result is None:
                continue
            if isinstance(result, tuple):
                metrics, video_logs = result
                eval_metrics_list.append(metrics)
                all_video_logs.extend(video_logs)
            else:
                eval_metrics_list.append(result)
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        per_instruction_metrics = compute_per_instruction_metrics(eval_metrics_list)
        return eval_metrics, per_instruction_metrics, all_video_logs

    def _log_eval_videos(self, video_logs: list[dict], step: int) -> None:
        """Sample video_logs and log them to W&B as a table."""
        if not video_logs or "wandb" not in self.metric_logger.logger_backends:
            return
        import wandb

        num_samples = self.cfg.runner.logger.get("num_eval_videos", 5)
        sample = random.sample(video_logs, min(num_samples, len(video_logs)))

        rows = []
        for entry in sample:
            path = entry["video_path"]
            task_descs = entry.get("task_descriptions", [])
            instructions = "\n".join(sorted(set(task_descs))) if task_descs else ""
            pert_types = entry.get("perturbation_types", [])
            unique_perts = sorted(set(p for p in pert_types if p))
            perturbation = ", ".join(unique_perts) if unique_perts else ""
            rows.append([wandb.Video(path, fps=10, format="mp4"), instructions, perturbation])

        table = wandb.Table(columns=["video", "instructions", "perturbation"], data=rows)
        self.metric_logger.logger["wandb"].log(
            {"eval/videos": table, "global_step": step}, commit=False
        )

    def run(self):
        eval_metrics, per_instruction_metrics, video_logs = self.evaluate()
        prefix = self.cfg.runner.logger.get("metric_prefix", "eval")
        eval_metrics = {f"{prefix}/{k}": v for k, v in eval_metrics.items()}
        self.logger.info(eval_metrics)
        self.metric_logger.log(step=0, data=eval_metrics)
        if per_instruction_metrics:
            instr_metrics = {
                f"eval/instruction_success/{desc}": rate
                for desc, rate in per_instruction_metrics.items()
            }
            self.logger.info(instr_metrics)
            self.metric_logger.log(step=0, data=instr_metrics)

        self._log_eval_videos(video_logs, step=0)
        self.metric_logger.finish()
