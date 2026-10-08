from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from pde.configuration import configure_rlinf, validate_pde
from pde.pools import load_pools
from pde.prompt_pool import PromptPool


@pytest.mark.parametrize("name", ["libero", "libero_pro", "search"])
def test_composes_with_pinned_rlinf(name):
    configure_rlinf()
    directory = str(Path(__file__).resolve().parents[1] / "pde/configs")
    with initialize_config_dir(config_dir=directory, version_base="1.1"):
        cfg = compose(
            config_name=name,
            overrides=["actor.model.model_path=checkpoint", "pde.pool_dir=pools"],
        )
    validate_pde(cfg)
    assert cfg.algorithm.update_epoch == 4
    assert cfg.rollout.model.model_path == cfg.actor.model.model_path
    assert cfg.pde.ema_beta == 0.3
    cfg.env.train.auto_reset = True
    with pytest.raises(ValueError, match="auto_reset"):
        validate_pde(cfg)


def test_rejects_wrong_checkpoint_and_duplicate_task_pools(tmp_path):
    pool = PromptPool("t", "goal", "checkpoint", "libero")
    pool.save(tmp_path / "a.json")
    with pytest.raises(ValueError, match="checkpoint"):
        load_pools(tmp_path, "different")
    pool.save(tmp_path / "b.json")
    with pytest.raises(ValueError, match="unique"):
        load_pools(tmp_path)
