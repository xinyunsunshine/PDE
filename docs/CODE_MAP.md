# Code map

| Concern | Location |
|---|---|
| Prompt-pool schema and validation | `pde/prompt_pool.py` |
| Frozen-policy prompt search | `pde/discovery.py` |
| Prompt sampling and mixed likelihood | `pde/training.py` |
| VLM prompts and transport | `pde/rlinf/prompting.py`, `vlm_client.py` |
| Trajectory relabeling | `pde/rlinf/relabeler.py` |
| Single-branch PDE actor | `pde/rlinf/actor.py` |
| Canonical + relabeled dual actor | `pde/rlinf/dual_actor.py` |
| Language-aware RLinf trajectory types | `pde/rlinf/data.py` |
| LIBERO and environment-worker extensions | `pde/rlinf/env.py` |
| OpenPI prompt retokenization | `pde/rlinf/model.py` |
| Training entry point | `pde/rlinf/train.py` |
| Unmodified training framework | `RLinf/` submodule |

The JSON prompt pool is the stable boundary between discovery and training.
Raw videos, endpoint credentials, scheduler state, and machine-local paths are
not part of the artifact.
