"""Load the immutable task pools consumed by PDE workers."""

from pathlib import Path

from pde.prompt_pool import load_prompt_pool


def load_pools(directory, checkpoint=None):
    paths = sorted(Path(directory).glob("*.json"))
    if not paths:
        raise ValueError(f"No prompt pools found in {directory}")
    pools = [load_prompt_pool(path) for path in paths]
    prompts = [pool.canonical_prompt for pool in pools]
    if len(set(prompts)) != len(prompts):
        raise ValueError("Each pool must have a unique canonical task instruction")
    if checkpoint is not None:
        for pool in pools:
            if pool.policy_checkpoint != checkpoint:
                raise ValueError(
                    f"{pool.task_id}: pool checkpoint differs from training initialization"
                )
    return pools
