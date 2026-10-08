"""Record the inputs needed to compare a run with a paper experiment."""

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

RLINF_REVISION = "fce5435df9472e2c61957e4f849fb903fc70827c"


def verify_rlinf(root):
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != RLINF_REVISION:
        raise ValueError(f"Expected RLinf {RLINF_REVISION}, found {revision}")
    spec = importlib.util.find_spec("rlinf")
    if (
        spec is None
        or Path(spec.origin).resolve() != (root / "rlinf/__init__.py").resolve()
    ):
        raise RuntimeError(
            "Python must import the pinned submodule; install with pip install -e RLinf"
        )
    if subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError(
            "RLinf submodule has local changes; use the unmodified pinned revision"
        )


def write_manifest(cfg):
    destination = Path(cfg.runner.logger.log_path, cfg.runner.logger.experiment_name)
    destination.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, destination / "config.yaml", resolve=True)
    pools = {}
    if cfg.pde.enabled:
        pool_dest = destination / "prompt_pools"
        pool_dest.mkdir(exist_ok=True)
        for path in sorted(Path(cfg.pde.pool_dir).glob("*.json")):
            data = path.read_bytes()
            pools[path.name] = hashlib.sha256(data).hexdigest()
            (pool_dest / path.name).write_bytes(data)
    manifest = {
        "rlinf_commit": RLINF_REVISION,
        "initial_checkpoint": cfg.actor.model.model_path,
        "prompt_pool_sha256": pools,
        "seed": cfg.actor.seed,
        "note": "Record the checkpoint content hash and benchmark revision with published results.",
    }
    (destination / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
