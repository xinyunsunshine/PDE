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

import os

from omegaconf import DictConfig, OmegaConf


class _TensorboardLogger:
    def __init__(self, log_path):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_path)

    def log(self, data: dict[str, float], step: int) -> None:
        for key, value in data.items():
            self.writer.add_scalar(key, value, step)

    def finish(self):
        self.writer.close()


class MetricLogger:
    supported_logger = ["wandb", "swanlab", "tensorboard"]

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        logger_cfg = cfg.runner.logger

        self.log_path = logger_cfg.get("log_path", "logs")
        self.project_name = logger_cfg.get("project_name", "rlinf")
        self.experiment_name = logger_cfg.get("experiment_name", "default")
        self.per_worker_log = bool(cfg.runner.get("per_worker_log", False))
        self.per_worker_log_root = cfg.runner.get(
            "per_worker_log_path", os.path.join(self.log_path, "worker_logs")
        )

        logger_backends = logger_cfg.get("logger_backends", ["tensorboard"])
        if isinstance(logger_backends, str):
            self.logger_backends = [logger_backends]
        elif logger_backends is None:
            self.logger_backends = []
        else:
            self.logger_backends = logger_backends

        self.wandb_proxy = logger_cfg.get("wandb_proxy", None)
        self.wandb_run_id = logger_cfg.get("wandb_run_id", None)
        self.swanlab_mode = logger_cfg.get("swanlab_mode", "cloud")
        if len(self.logger_backends) > 0:
            assert all(
                backend in self.supported_logger for backend in self.logger_backends
            ), f"Unsupported logger backend: {self.logger_backends}"

        self.config = OmegaConf.to_container(cfg, resolve=True)
        self._all_loggers = []
        self._worker_loggers: dict[tuple[str, int], dict] = {}
        self._latest_step_by_logger: dict[int, int] = {}
        self._resume_dir = cfg.runner.get("resume_dir", None)
        self.logger = self._create_logger_bundle(
            log_path=self.log_path,
            experiment_name=self.experiment_name,
            log_path_suffix="all" if self.per_worker_log else "",
        )

    def _create_logger_bundle(
        self, log_path: str, experiment_name: str, log_path_suffix: str = ""
    ) -> dict:
        logger = {}
        if "wandb" in self.logger_backends:
            import wandb

            wandb_log_path = os.path.join(log_path, "wandb", log_path_suffix)
            os.makedirs(wandb_log_path, exist_ok=True)

            settings = None
            if self.wandb_proxy:
                settings = wandb.Settings(https_proxy=self.wandb_proxy)

            init_kwargs = {
                "project": self.project_name,
                "name": experiment_name,
                "config": self.config,
                "settings": settings,
                "dir": wandb_log_path,
                "reinit": True,
            }
            if self.wandb_run_id:
                init_kwargs["id"] = self.wandb_run_id
                init_kwargs["resume"] = "allow"
            wandb.init(**init_kwargs)
            wandb.define_metric("global_step")
            wandb.define_metric("*", step_metric="global_step")

            logger["wandb"] = wandb

        if "swanlab" in self.logger_backends:
            import swanlab

            swanlab_log_path = os.path.join(log_path, "swanlab", log_path_suffix)
            os.makedirs(swanlab_log_path, exist_ok=True)

            swanlab.init(
                project=self.project_name,
                experiment_name=experiment_name,
                config=self.config,
                logdir=swanlab_log_path,
                mode=self.swanlab_mode,
            )
            logger["swanlab"] = swanlab

        if "tensorboard" in self.logger_backends:
            tensorboard_log_path = os.path.join(
                log_path, "tensorboard", log_path_suffix
            )
            os.makedirs(tensorboard_log_path, exist_ok=True)

            config_yaml_path = os.path.join(tensorboard_log_path, "config.yaml")
            OmegaConf.save(self.cfg, config_yaml_path, resolve=True)

            logger["tensorboard"] = _TensorboardLogger(tensorboard_log_path)
        self._all_loggers.append(logger)
        return logger

    def _log_wandb(self, logger_bundle: dict, data: dict, step: int, commit: bool = False):
        """Log metrics to a specific wandb logger bundle with explicit global step."""
        payload = dict(data)
        payload["global_step"] = step
        # Pass step= so W&B's internal counter matches the training global step.
        # This ensures the default x-axis in the W&B UI is correct even for new
        # runs created on resume (where the first step is e.g. 26, not 0).
        logger_bundle["wandb"].log(payload, step=step, commit=commit)
        self._latest_step_by_logger[id(logger_bundle)] = step

    def _get_scoped_logger(self, worker_group_name: str, rank: int) -> dict:
        key = (worker_group_name, int(rank))
        if key in self._worker_loggers:
            return self._worker_loggers[key]

        scoped_log_path = os.path.join(
            self.per_worker_log_root,
            worker_group_name,
            f"rank_{int(rank)}",
        )
        scoped_experiment_name = (
            f"{self.experiment_name}-{worker_group_name}-rank_{int(rank)}"
        )
        scoped_logger = self._create_logger_bundle(
            log_path=scoped_log_path,
            experiment_name=scoped_experiment_name,
        )
        self._worker_loggers[key] = scoped_logger
        return scoped_logger

    def log(
        self,
        data,
        step,
        backend=None,
        worker_group_name: str | None = None,
        rank: int | None = None,
    ):
        target_logger = self.logger
        if self.per_worker_log and worker_group_name is not None and rank is not None:
            target_logger = self._get_scoped_logger(
                worker_group_name=worker_group_name,
                rank=rank,
            )
        for default_backend, logger_instance in target_logger.items():
            if backend is None or default_backend in backend:
                if default_backend == "wandb":
                    self._log_wandb(target_logger, data, step=step)
                else:
                    logger_instance.log(data=data, step=step)

    def flush(self, step: int) -> None:
        """Commit all buffered wandb metrics for the current step.

        wandb.log with an explicit ``step=`` argument defaults to
        ``commit=False``, which means data is only sent to the server when the
        *next* step is logged.  Call this once at the end of each training step
        to force an immediate commit so the run history stays up to date.
        """
        if "wandb" in self.logger:
            self._log_wandb(self.logger, {}, step=step, commit=True)

    def log_table(self, df_data, name, step):
        if "wandb" in self.logger_backends:
            table = self.logger["wandb"].Table(dataframe=df_data)
            self._log_wandb(self.logger, {name: table}, step=step)
        else:
            raise ValueError(f"Unsupported log table for {self.logger_backends}")

    def __del__(self):
        self.finish()

    def finish(self):
        for logger in self._all_loggers:
            if "wandb" in logger:
                latest_step = self._latest_step_by_logger.get(id(logger))
                if latest_step is not None and logger["wandb"].run is not None:
                    logger["wandb"].run.summary["latest_global_step"] = latest_step
            for logger_instance in logger.values():
                logger_instance.finish()
