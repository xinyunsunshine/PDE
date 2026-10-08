# Prompt-Driven Exploration

Code for **Prompt-Driven Exploration: Language as an Exploration Space for VLA
Reinforcement Learning**.

PDE discovers prompts that elicit useful behavior from a frozen VLA, then trains
with a fixed prompt pool while transferring that behavior to the original task
instruction. Evaluation always uses the original instruction.

The method lives in `pde/`: `PDEActor` is our PPO actor, inheriting RLinf's
`EmbodiedFSDPActor`. The unmodified `RLinf/` submodule supplies distributed
execution, policy loading, simulation, PPO and optimization.

## Install

Use Python 3.10 or 3.11 and clone the pinned dependency:

```bash
git clone --recurse-submodules https://github.com/xinyunsunshine/PDE.git
cd PDE
```

Set up the pi0.5/LIBERO environment using
[RLinf's pinned installation instructions](RLinf/docs/source-en/rst_source/examples/embodied/pi0.rst).
Then, in that environment:

```bash
python -m pip install -e RLinf
python -m pip install -e ".[integration]"
```

An existing clone needs `git submodule update --init --recursive`. Launch from
the repository root. Both actor and rollout workers must see the same checkpoint
and pool directory.

## 1. Discover prompts with a VLM

Run discovery once per task, with fixed policy weights. For example, for task 0
of the configured LIBERO-object suite:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://YOUR_VLM_ENDPOINT/v1
python -m pde.search \
  actor.model.model_path=/path/to/weak-pi05 \
  search.task_index=0 search.task_id=libero_object.task_0 \
  search.output=prompt_pools/task_0.json
```

The command evaluates the canonical instruction, then performs up to ten rounds
of five proposals with ten rollouts per candidate. The VLM summarizes videos;
both successful and failed prompts enter its feedback history. Only prompts
with nonzero task success enter the admitted pool. Set `env.train.libero_variant=pro`
and `+env.train.perturbation_suffix=task` for LIBERO-PRO discovery.

The example JSON in `pde/examples/` illustrates the schema; it is not a measured
paper prompt pool.

## 2. Train with frozen prompt pools

Once the pool directory covers every selected task:

```bash
python -m pde.train --config-name libero_pro \
  actor.model.model_path=/path/to/weak-pi05 \
  pde.pool_dir=prompt_pools
```

For a single task, add `env.train.filter_task_ids=[0]` and
`env.eval.total_num_envs=250`. Use `--config-name libero` for standard LIBERO.
Run the matched PPO baseline with the same overrides plus `pde.enabled=false`.
Training makes no VLM calls.

A prompt stays fixed for each rollout episode. Canonical-prompt successes update
an EMA with smoothing 0.3; its value controls the canonical sampling probability
with a floor of 0.05 and consolidation target of 0.5. Each PPO update uses

```text
logp = 0.5 * log pi(action | observation, sampled_prompt)
     + 0.5 * log pi(action | observation, canonical_prompt)
ratio = exp(logp - old_logp_under_sampled_prompt)
```

Both current-policy terms receive gradients. Rewards and the stored old
likelihood remain those of the original rollout.

## Reproducibility status

The current pi0.5/LIBERO implementation follows the attached paper's two-stage
method and has CPU regression tests against the pinned RLinf. It replaces the
earlier HER-based release, which did not implement that method correctly.
A full GPU/simulator training run has **not** been validated.

Reproducing the reported scores also requires the exact weak SFT checkpoint,
paper prompt pools, benchmark revision/task splits, and resolved run configs.
Those artifacts are not in this repository yet. The supplied configs encode
the paper's shared PPO settings, not verified figure-specific experiment configs.
See [reproduction details](docs/REPRODUCTION.md) and the [code map](docs/CODE_MAP.md).

## Tests

```bash
python -m pip install -e ".[test]"
pytest -q
ruff check pde tests
python scripts/release_audit.py .
```

Runtime tests use the installed RLinf training dependencies. Core artifact,
objective and config tests run on CPU; no model download or VLM call is needed.

## Citation

```bibtex
@misc{jiang2026promptdriven,
  title = {Prompt-Driven Exploration: Language as an Exploration Space for VLA Reinforcement Learning},
  author = {Jiang, Sunshine and Marangola, John and Zhang, David and Kowdeed, Raghuram and Luo, Ruiyang and Dashora, Nitish and Li, Richard and Agrawal, Pulkit and Hong, Zhang-Wei},
  year = {2026},
  url = {https://xinyunsunshine.github.io/prompt-rl}
}
```

Also available as [CITATION.bib](CITATION.bib). PDE and RLinf use Apache-2.0;
models and benchmarks retain their own licenses.
