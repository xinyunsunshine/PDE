"""Resolve local RLinf configs and reject unsupported reproduction settings."""

import os
from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.config_store import ConfigStore
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import OmegaConf


class RLinfConfigSearchPath(SearchPathPlugin):
    """Resolve RLinf's model/environment defaults without modifying the submodule."""

    def manipulate_search_path(self, search_path: ConfigSearchPath):
        search_path.append(
            provider="pde", path="file://" + os.environ["PDE_RLINF_CONFIG"]
        )


def configure_rlinf():
    root = Path(
        os.environ.get("PDE_RLINF_PATH", Path(__file__).resolve().parents[1] / "RLinf")
    )
    if not (root / "examples/embodiment/config").is_dir():
        raise RuntimeError("Initialize RLinf: git submodule update --init --recursive")
    os.environ["EMBODIED_PATH"] = str(root / "examples/embodiment")
    os.environ["PDE_RLINF_CONFIG"] = str(root / "examples/embodiment/config")
    base = OmegaConf.load(
        root / "examples/embodiment/config/libero_object_ppo_openpi_pi05.yaml"
    )
    # Upstream's searchpath is legal only when that YAML is the primary config.
    # Register its remaining settings unchanged and supply search paths as a plugin.
    del base.hydra.searchpath
    ConfigStore.instance().store(name="rlinf_ppo", node=base)
    Plugins.instance().register(RLinfConfigSearchPath)
    return root


def validate_pde(cfg):
    if cfg.actor.model.model_type != "openpi":
        raise ValueError("This release supports pi0.5/OpenPI")
    if cfg.algorithm.adv_type != "gae" or cfg.algorithm.loss_type != "actor_critic":
        raise ValueError("PDE requires PPO with GAE, as in the paper")
    if cfg.env.train.auto_reset:
        raise ValueError(
            "Use auto_reset=false to hold prompts fixed for a full episode"
        )
    if cfg.env.train.max_steps_per_rollout_epoch != cfg.env.train.max_episode_steps:
        raise ValueError("Each PDE rollout epoch must cover the full episode horizon")
    if cfg.env.train.max_episode_steps % cfg.actor.model.num_action_chunks:
        raise ValueError("Episode horizon must be divisible by action chunk size")
    if cfg.rollout.get("expert_model") or cfg.actor.get("enable_sft_co_train", False):
        raise ValueError("Expert mixing and SFT co-training are outside the PDE path")
    if cfg.algorithm.get("filter_rewards", False):
        raise ValueError("Disable reward filtering for the released PPO path")
    if cfg.get("reward", {}).get("use_reward_model", False):
        raise ValueError("PDE uses the original environment reward")
    if cfg.runner.get("weight_sync_interval", 1) != 1:
        raise ValueError(
            "PDE requires a curriculum and weight sync every training step"
        )
    if not 0 < cfg.pde.ema_beta <= 1:
        raise ValueError("ema_beta must be in (0, 1]")
