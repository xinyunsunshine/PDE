# Prompt-Driven Exploration

Official code release for **Prompt-Driven Exploration: Language as an
Exploration Space for VLA Reinforcement Learning**.

PDE uses language to expose behaviors already present in a vision-language-
action policy. The release has two parts:

1. `pde.discovery` searches for useful instructions with a frozen policy and a
   VLM supervisor, then writes a versioned prompt-pool JSON artifact.
2. `pde.rlinf` trains with relabeled prompts while anchoring updates to the
   original instruction. Every framework extension subclasses an RLinf type.

RLinf is pinned as the `RLinf/` Git submodule. No RLinf source is copied into
this repository.

## Installation

Clone with submodules and use Python 3.10 or 3.11:

```bash
git clone --recurse-submodules https://github.com/xinyunsunshine/PDE.git
cd PDE
python -m pip install -e RLinf
python -m pip install -e ".[integration]"
```

If the repository was cloned without submodules, run
`git submodule update --init --recursive`.

Follow the [RLinf installation guide](https://rlinf.readthedocs.io/) for the
model, simulator, CUDA, and distributed-runtime dependencies used by your
experiment.

## Prompt discovery

```python
from pde import DiscoveryConfig, PromptPool, discover_prompt_pool

pool = PromptPool(
    task_id="libero.close_the_microwave",
    canonical_prompt="close the microwave",
    policy_checkpoint="org/pi05-weak-sft",
    environment="LIBERO",
)
pool = discover_prompt_pool(
    pool,
    evaluator=rollout_evaluator,
    supervisor=vlm_supervisor,
    config=DiscoveryConfig(),
)
pool.save("prompt_pool.json")
```

The evaluator and supervisor are explicit callables so users can choose the VLA
runtime and OpenAI-compatible VLM endpoint without storing credentials in an
artifact. Validate a pool with:

```bash
pde validate-pool pde/examples/close_microwave.example.json
```

## RLinf integration

| PDE class | RLinf base |
|---|---|
| `PDEActor` | `EmbodiedFSDPActor` |
| `PDEDualActor` | `PDEActor` |
| `PDEEnvWorker` | `EnvWorker` |
| `PDELiberoEnv` | `LiberoEnv` |
| `PDEOpenPiActionModel` | `OpenPi0ForRLActionPrediction` |
| `PDETrajectory` | `Trajectory` |
| `PDERolloutResult` | `EmbodiedRolloutResult` |
| `PDERunner` | `EmbodiedRunner` |

The entry point starts from RLinf's pi0.5 LIBERO configuration and swaps in
these subclasses:

```bash
python -m pde.rlinf.train \
  +algorithm.pde_dual=true \
  +algorithm.her_endpoint=http://VLM_HOST:PORT/v1 \
  +algorithm.her_model=Qwen/Qwen3-VL-235B-A22B-Thinking-FP8 \
  +algorithm.her_video_mode=frames \
  +algorithm.her_prompt_version=v7
```

Use Hydra overrides from the pinned RLinf configuration for checkpoints,
placement, environment counts, logging, and training budgets. See the
[release guide](docs/PDE_RELEASE.md) and [code map](docs/CODE_MAP.md).

## Development checks

```bash
python -m pip install -e ".[test]"
pytest -q
ruff check pde tests
python scripts/release_audit.py .
```

Citation metadata is in [`CITATION.cff`](CITATION.cff). PDE and the pinned
RLinf dependency use Apache-2.0; external models and benchmarks retain their
own licenses.
