"""Episode-consistent prompt sampling on RLinf's rollout worker."""

import hashlib
import json
import random

import torch
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

from pde.objective import canonical_tokens
from pde.pools import load_pools
from pde.training import canonical_sampling_probability, sample_rollout_prompt


class PDERolloutWorker(MultiStepRolloutWorker):
    """Collect actions under one sampled prompt for each complete episode."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.pools = load_pools(cfg.pde.pool_dir, cfg.actor.model.model_path)
        self.task_indices = {p.canonical_prompt: i for i, p in enumerate(self.pools)}
        payload = json.dumps([p.to_dict() for p in self.pools], sort_keys=True)
        self.pool_digest = hashlib.sha256(payload.encode()).hexdigest()
        self.success_ema = [0.0] * len(self.pools)
        self._episode_prompts = {}
        self._stage = 0
        self._epoch = 0

    def set_curriculum(self, state):
        if state["pool_digest"] != self.pool_digest:
            raise ValueError("Actor and rollout workers loaded different prompt pools")
        self.success_ema = list(state["success_ema"])

    async def generate(self, *args, **kwargs):
        self._epoch = 0
        return await super().generate(*args, **kwargs)

    async def generate_one_epoch(self, *args, **kwargs):
        self._episode_prompts = {}
        self._stage = 0
        self._rng = random.Random(
            f"{self.cfg.actor.seed}:{self._rank}:{self.version}:{self._epoch}"
        )
        self._epoch += 1
        return await super().generate_one_epoch(*args, **kwargs)

    def get_bootstrap_values(self, final_obs):
        # Bootstrapping must not advance the stage counter or resample prompts.
        if final_obs is not None:
            raise ValueError("PDE requires fixed-horizon rollouts without auto-reset")
        return None

    def predict(self, env_obs, mode="train"):
        if mode == "eval":
            return super().predict(env_obs, mode=mode)
        stage = self._stage % self.num_pipeline_stages
        self._stage += 1
        originals = list(env_obs["task_descriptions"])
        missing = set(originals) - self.task_indices.keys()
        if missing:
            raise ValueError(
                f"Missing prompt pools for canonical instructions: {sorted(missing)}"
            )
        indices = [self.task_indices[prompt] for prompt in originals]
        if stage not in self._episode_prompts:
            prompts = [
                sample_rollout_prompt(
                    self.pools[i],
                    canonical_sampling_probability(
                        self.success_ema[i],
                        self.cfg.pde.consolidation_target,
                        self.cfg.pde.minimum,
                    ),
                    self._rng,
                )
                for i in indices
            ]
            self._episode_prompts[stage] = (originals, prompts)
        previous, prompts = self._episode_prompts[stage]
        if previous != originals:
            raise ValueError("Task changed inside a PDE rollout epoch")
        actions, result = super().predict(
            dict(env_obs, task_descriptions=prompts), mode=mode
        )
        inputs = result["forward_inputs"]
        tokens, masks = canonical_tokens(self.hf_model, originals, inputs)
        inputs["canonical_prompt"] = tokens
        inputs["canonical_prompt_mask"] = masks
        inputs["pde_task"] = torch.tensor(indices, dtype=torch.long)[:, None]
        inputs["pde_canonical"] = torch.tensor(
            [prompt == original for prompt, original in zip(prompts, originals)]
        )[:, None]
        return actions, result
