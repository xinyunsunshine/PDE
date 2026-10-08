"""RLinf environment worker selecting PDE's LIBERO subclass."""

import os

from rlinf.workers.env.env_worker import EnvWorker


class PDEEnvWorker(EnvWorker):
    """Keep simulation and rewards in RLinf, with explicit task selection."""

    def init_worker(self):
        variant = self.cfg.env.train.get("libero_variant", "standard")
        if self.cfg.env.eval.get("libero_variant", "standard") != variant:
            raise ValueError("Train and eval must use the same LIBERO package variant")
        os.environ["LIBERO_TYPE"] = variant
        super().init_worker()

    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage):
        from pde.libero import PDELiberoEnv

        return super()._setup_env_and_wrappers(
            PDELiberoEnv, env_cfg, num_envs_per_stage
        )
