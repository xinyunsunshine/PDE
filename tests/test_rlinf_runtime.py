"""Run in the RLinf environment; exercises the actual pinned base classes."""

import random
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

# These require RLinf's installed training dependencies, beyond core CPU tests.
pytest.importorskip("ray")
from pde.actor import PDEActor
from pde.rollout import PDERolloutWorker
from pde.prompt_pool import PromptPool, PromptCandidate
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def test_actual_base_preprocessing_and_canonical_only_ema(monkeypatch):
    actor = object.__new__(PDEActor)
    actor.cfg = OmegaConf.create(
        {
            "algorithm": {"rollout_epoch": 2, "filter_rewards": False},
            "env": {"train": {"auto_reset": False, "ignore_terminations": True}},
            "pde": {"ema_beta": 0.3, "consolidation_target": 0.5, "minimum": 0.05},
        }
    )
    actor.device = "cpu"
    actor.pools = [None, None]
    actor.success_ema = [0.0, 0.4]
    # Two epochs, two envs. Exploratory success for task 1 must not affect EMA.
    batch = {
        "rewards": torch.tensor([[[1.0], [1.0]], [[0.0], [1.0]]]),
        "forward_inputs": {
            "pde_task": torch.tensor([[[0], [1]], [[0], [1]]]),
            "pde_canonical": torch.tensor([[[True], [False]], [[True], [False]]]),
        },
    }
    actor.rollout_batch = actor._process_received_rollout_batch(batch)
    assert actor.rollout_batch["rewards"].shape == (1, 4, 1)
    monkeypatch.setattr(
        EmbodiedFSDPActor, "compute_advantages_and_returns", lambda self: {}
    )
    metrics = actor.compute_advantages_and_returns()
    assert actor.success_ema == pytest.approx([0.15, 0.4])
    assert metrics["pde/task_0/canonical_probability"] == pytest.approx(0.3)


def test_episode_prompts_persist_and_eval_is_canonical(monkeypatch):
    worker = object.__new__(PDERolloutWorker)
    worker.cfg = OmegaConf.create(
        {"pde": {"consolidation_target": 0.5, "minimum": 0.0}}
    )
    pool = PromptPool("t", "goal", "checkpoint", "libero")
    pool.add(PromptCandidate("explore", 1, 1, "", 0))
    worker.pools = [pool]
    worker.task_indices = {"goal": 0}
    worker.success_ema = [0.0]
    worker._episode_prompts = {}
    worker._stage = 0
    worker.num_pipeline_stages = 1
    worker._rng = random.Random(3)
    worker.hf_model = SimpleNamespace(
        input_transform=lambda obs, transpose: {
            "tokenized_prompt": torch.ones(1, 4),
            "tokenized_prompt_mask": torch.ones(1, 4, dtype=torch.bool),
        }
    )
    seen = []

    def predict(self, obs, mode="train"):
        seen.append((mode, obs["task_descriptions"]))
        return torch.zeros(1, 7), {
            "forward_inputs": {
                "tokenized_prompt": torch.zeros(1, 4),
                "observation/state": torch.zeros(1, 3),
            }
        }

    monkeypatch.setattr(MultiStepRolloutWorker, "predict", predict)
    observations = {"task_descriptions": ["goal"]}
    _, output = worker.predict(observations)
    worker.success_ema = [1.0]  # must only affect the next episode
    worker.predict(observations)
    worker.predict(observations, mode="eval")
    assert seen == [("train", ["explore"]), ("train", ["explore"]), ("eval", ["goal"])]
    assert not output["forward_inputs"]["pde_canonical"].any()
    assert observations["task_descriptions"] == ["goal"]


def test_curriculum_checkpoint_and_pool_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(EmbodiedFSDPActor, "save_checkpoint", lambda *args: None)
    monkeypatch.setattr(EmbodiedFSDPActor, "load_checkpoint", lambda *args: None)
    actor = object.__new__(PDEActor)
    actor._rank = 0
    actor.pools = [None]
    actor.pool_digest = "same"
    actor.success_ema = [0.25]
    actor.save_checkpoint(str(tmp_path), 3)
    actor.success_ema = [0.0]
    actor.load_checkpoint(str(tmp_path))
    assert actor.success_ema == [0.25]
    actor.pool_digest = "changed"
    with pytest.raises(ValueError, match="different"):
        actor.load_checkpoint(str(tmp_path))
