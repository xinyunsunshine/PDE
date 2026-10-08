"""Executable frozen-pi0.5 discovery using RLinf simulation and a VLM."""

import copy
import os
import random

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from pde.configuration import configure_rlinf
from pde.discovery import DiscoveryConfig, Evaluation, discover_prompt_pool
from pde.prompt_pool import PromptPool

RLINF_ROOT = configure_rlinf()


@hydra.main(version_base="1.1", config_path="configs", config_name="search")
def main(cfg):
    from pde.vlm import VLMSupervisor
    from pde.provenance import verify_rlinf

    verify_rlinf(RLINF_ROOT)
    # Set before importing any LIBERO modules; simulator subprocesses inherit it.
    os.environ["LIBERO_TYPE"] = cfg.env.train.libero_variant
    from rlinf.envs.action_utils import prepare_actions
    from rlinf.models import get_model
    from pde.libero import PDELiberoEnv

    settings = DiscoveryConfig(**OmegaConf.to_container(cfg.search.budget))
    env_cfg = copy.deepcopy(cfg.env.train)
    with open_dict(env_cfg):
        env_cfg.group_size = 1
        env_cfg.auto_reset = False
        env_cfg.ignore_terminations = True
        env_cfg.filter_task_ids = [cfg.search.task_index]
        env_cfg.use_fixed_reset_state_ids = False
        env_cfg.video_cfg.save_video = False
    if env_cfg.max_episode_steps % cfg.actor.model.num_action_chunks:
        raise ValueError("Episode horizon must be divisible by action chunk size")
    random.seed(cfg.actor.seed)
    np.random.seed(cfg.actor.seed)
    torch.manual_seed(cfg.actor.seed)
    env = PDELiberoEnv(env_cfg, settings.rollouts_per_candidate, 0, 1, None)
    try:
        model = get_model(cfg.actor.model)
        model.eval()
        model.requires_grad_(False)
        supervisor = VLMSupervisor(
            cfg.search.vlm_model, cfg.search.base_url, cfg.search.frames_per_video
        )
        obs, _ = env.reset()
        originals = set(obs["task_descriptions"])
        if len(originals) != 1:
            raise ValueError(
                "Discovery requires a single canonical instruction; split BDDL variants into separate runs"
            )
        canonical = originals.pop()
        pool = PromptPool(
            task_id=cfg.search.task_id,
            canonical_prompt=canonical,
            policy_checkpoint=cfg.actor.model.model_path,
            environment=f"{env_cfg.libero_variant}:{env_cfg.task_suite_name}:{env_cfg.get('perturbation_suffix', '')}",
            metadata={
                "task_index": cfg.search.task_index,
                "seed": env_cfg.seed,
                "vlm_model": cfg.search.vlm_model,
                "rlinf_revision": "fce5435df9472e2c61957e4f849fb903fc70827c",
            },
        )

        def evaluate(task_id, prompt, episodes):
            observations, _ = env.reset()
            if any(p != canonical for p in observations["task_descriptions"]):
                raise ValueError(
                    "Canonical instruction changed between discovery episodes"
                )
            success = np.zeros(episodes, dtype=bool)
            frames = [[] for _ in range(episodes)]
            steps = env_cfg.max_episode_steps // cfg.actor.model.num_action_chunks
            selected = set(
                np.linspace(0, steps, cfg.search.frames_per_video, dtype=int).tolist()
            )

            def capture(step):
                if step in selected:
                    images = observations["main_images"].cpu().numpy()
                    for index, image in enumerate(images):
                        frames[index].append(image.astype(np.uint8))

            capture(0)
            for step in range(steps):
                inputs = dict(observations, task_descriptions=[prompt] * episodes)
                inputs.setdefault("extra_view_images", None)
                with torch.inference_mode():
                    actions, _ = model.predict_action_batch(
                        env_obs=inputs, mode="train"
                    )
                actions = prepare_actions(
                    raw_chunk_actions=actions,
                    env_type=env_cfg.env_type,
                    model_type=cfg.actor.model.model_type,
                    num_action_chunks=cfg.actor.model.num_action_chunks,
                    action_dim=cfg.actor.model.action_dim,
                    policy=cfg.actor.model.get("policy_setup"),
                )
                obs_list, _, _, _, infos = env.chunk_step(actions)
                observations = obs_list[-1]
                success |= (
                    infos[-1]["episode"]["success_once"].cpu().numpy().astype(bool)
                )
                capture(step + 1)
            return Evaluation(
                success.astype(float).tolist(),
                supervisor.summarize(canonical, prompt, frames),
            )

        baseline = evaluate(pool.task_id, canonical, settings.rollouts_per_candidate)
        pool.metadata["canonical_evaluation"] = {
            "prompt": canonical,
            "summary": baseline.summary,
            "success_rate": sum(baseline.rewards) / len(baseline.rewards),
            "rollouts": len(baseline.rewards),
        }
        discover_prompt_pool(pool, evaluate, supervisor, settings)
        pool.save(cfg.search.output)
    finally:
        env.env.close()


if __name__ == "__main__":
    main()
