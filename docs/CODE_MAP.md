# Code map

The PDE method is implemented directly in `pde/`. RLinf remains an unmodified
submodule pinned to `fce5435df9472e2c61957e4f849fb903fc70827c`.

| Method component | Code | Responsibility |
|---|---|---|
| Prompt search | `pde/discovery.py` | Budget, history and pool admission |
| Executable search | `pde/search.py` | Frozen RLinf policy, simulator rollouts and videos |
| VLM supervisor | `pde/vlm.py` | Video summaries and new prompt proposals |
| Artifacts | `pde/prompt_pool.py`, `pde/pools.py` | Schema, loading and checkpoint matching |
| Actor | `pde/actor.py` | RLinf PPO actor subclass; canonical-success EMA and resume |
| Model objective | `pde/model.py`, `pde/objective.py` | Two prompt-conditioned likelihoods within one PPO loss |
| Rollout | `pde/rollout.py` | Frozen-pool sampling per episode and canonical token transport |
| Environment | `pde/env.py`, `pde/libero.py` | RLinf subclasses for task subsets and BDDL instructions |
| Runner | `pde/runner.py` | Synchronize globally aggregated curriculum with the policy |
| Entrypoint/configs | `pde/train.py`, `pde/configs/` | PDE or matched PPO, configuration checks |
| Provenance | `pde/provenance.py` | Verify RLinf pin; save resolved configs and pool hashes |

Workers, model and environment classes inherit their RLinf counterparts. Artifact
dataclasses and the VLM supervisor are PDE utilities; they have no corresponding
RLinf class to inherit. There is no separate `pde.rlinf` integration namespace.

Canonical prompt tokens travel as tensors in RLinf's existing `forward_inputs`.
This avoids custom trajectory classes, string metadata sharding, or global
monkey patches. RLinf retains its full advantage and optimizer implementations.

The OpenPI subclass overrides the default forward method. RLinf creates and
loads the model first; PDE promotes that individual instance to a stateless
subclass, preserving transforms, parameters, LoRA settings and state-dict names.
