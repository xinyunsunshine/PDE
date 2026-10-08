"""Prompt-Driven Exploration actor, built on RLinf's PPO actor."""

import hashlib
import json
from pathlib import Path

import torch
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

from pde.pools import load_pools
from pde.training import canonical_sampling_probability


class PDEActor(EmbodiedFSDPActor):
    """Train from frozen prompt pools with the paper's mixed likelihood.

    RLinf owns trajectory batching, advantages, PPO, FSDP and optimization.
    PDE only specializes the model likelihood and canonical-success curriculum.
    No VLM is invoked, and rewards remain the original task rewards.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.pools = load_pools(cfg.pde.pool_dir, cfg.actor.model.model_path)
        self.success_ema = [0.0] * len(self.pools)
        payload = json.dumps([p.to_dict() for p in self.pools], sort_keys=True)
        self.pool_digest = hashlib.sha256(payload.encode()).hexdigest()

    def model_provider_func(self):
        from pde.model import extend_model

        return extend_model(super().model_provider_func())

    def compute_advantages_and_returns(self):
        inputs = self.rollout_batch["forward_inputs"]
        tasks = inputs.pop("pde_task")[0].reshape(-1).long()
        canonical = inputs.pop("pde_canonical")[0].reshape(-1).bool()
        # success_once: any positive original task reward over the episode.
        successes = self.rollout_batch["rewards"].gt(0).any(dim=(0, 2))
        stats = torch.zeros(len(self.pools), 2, device=self.device)
        for task in range(len(self.pools)):
            selected = canonical & (tasks == task)
            stats[task, 0] = successes[selected].sum()
            stats[task, 1] = selected.sum()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats)
        beta = float(self.cfg.pde.ema_beta)
        for task, (wins, count) in enumerate(stats.cpu().tolist()):
            if count:
                self.success_ema[task] = (
                    beta * wins / count + (1 - beta) * self.success_ema[task]
                )
        metrics = super().compute_advantages_and_returns()
        for index, ema in enumerate(self.success_ema):
            metrics[f"pde/task_{index}/canonical_success_ema"] = ema
            metrics[f"pde/task_{index}/canonical_probability"] = (
                canonical_sampling_probability(
                    ema, self.cfg.pde.consolidation_target, self.cfg.pde.minimum
                )
            )
        return metrics

    def get_curriculum(self):
        return {"pool_digest": self.pool_digest, "success_ema": self.success_ema}

    def save_checkpoint(self, save_path, step=0):
        super().save_checkpoint(save_path, step)
        if self._rank == 0:
            Path(save_path, "pde.json").write_text(
                json.dumps(self.get_curriculum()) + "\n"
            )

    def load_checkpoint(self, load_path):
        state = json.loads(Path(load_path, "pde.json").read_text())
        if state["pool_digest"] != self.pool_digest:
            raise ValueError("Cannot resume PDE with a different prompt pool")
        if len(state["success_ema"]) != len(self.pools):
            raise ValueError("Invalid saved PDE curriculum")
        super().load_checkpoint(load_path)
        self.success_ema = state["success_ema"]
