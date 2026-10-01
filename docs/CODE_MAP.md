# PDE code and dependency map

This map separates the stable paper-facing API from the RLinf implementation
that executes large VLA experiments.

| Concern | Public entry point | RLinf integration |
|---|---|---|
| Prompt-pool schema | `pde/prompt_pool.py` | Task descriptions flow through rollout batches |
| Discovery budget and admission | `pde/discovery.py` | `rlinf/workers/actor/her_processor.py` |
| VLM requests and video encoding | Adapter protocol in `pde/discovery.py` | `her_vlm_client.py`, `her_utils.py` |
| Canonical/pool prompt sampling | `pde/training.py` | Environment and rollout workers under `rlinf/workers/` |
| Mixed backpropagation | `pde/training.py` | `her_dual_actor_worker.py` and model forward methods |
| PPO loss | RLinf registry | `rlinf/algorithms/losses.py`, `registry.py` |
| pi0.5 policy and log probabilities | RLinf model interface | `rlinf/models/embodiment/openpi/` |
| LIBERO-PRO environments | RLinf environment interface | `rlinf/envs/libero/`, `LIBERO-PRO/` |
| Training orchestration | Hydra entry point | `examples/embodiment/train_embodied_agent.py` |
| Evaluation | Canonical prompts only | `examples/embodiment/eval_embodied_agent.py` |

## Runtime dependencies

- Python 3.10 or 3.11, PyTorch, Ray, Hydra, NumPy, and the embodied optional
  dependencies declared in `pyproject.toml`;
- OpenPI/pi0.5 dependencies installed through `requirements/install.sh`;
- LIBERO and LIBERO-PRO assets, pinned to the revision recorded with a release;
- an OpenAI-compatible multimodal endpoint for the VLM supervisor;
- CUDA-capable hardware for policy rollout and training.

The `pde` artifact and scheduling modules themselves use only the Python
standard library. This keeps pool inspection and validation CPU-only.

## Data flow

```text
fixed checkpoint + task + rollout evaluator + VLM supervisor
                         |
                         v
              versioned prompt-pool JSON
                         |
                         v
canonical/pool episode sampling -> VLA rollouts -> mixed PPO update
                         |
                         v
              canonical-prompt evaluation
```

Prompt-pool JSON is the only supported boundary between discovery and RL. Raw
videos, endpoint credentials, scheduler state, and local paths must remain
outside released artifacts.

## Upstream and licenses

The training substrate is derived from RLinf and retains its Apache-2.0 license
and copyright headers. LIBERO, LIBERO-PRO, OpenPI/pi0.5, GR00T, and other model
or benchmark assets keep their own licenses and are external dependencies unless
their files are explicitly included with compatible notices.
