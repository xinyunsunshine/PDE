# PDE release guide

## Stage 1: discover prompts with a frozen policy

Create one `PromptPool` per task. Pass `discover_prompt_pool` a rollout
evaluator and a VLM supervisor. The default budget is ten rounds, five
candidates per round, and ten evaluations per candidate. A candidate is
admitted when it achieves nonzero empirical success. Keep the policy checkpoint
fixed throughout discovery and record its immutable identifier in the pool.

## Stage 2: train through RLinf

Initialize the `RLinf/` submodule and install RLinf using its environment guide.
`python -m pde.rlinf.train` uses RLinf's runner, scheduler, rollout worker, and
reward worker, while selecting PDE subclasses for the actor, environment,
trajectory, LIBERO environment, OpenPI model, and runner.

`PDEActor` relabels rollout instructions and rewards before advantage
calculation. `PDEDualActor` retains the canonical batch and trains on paired
canonical and relabeled batches. Set `algorithm.pde_dual=false` to use the
single branch.

Evaluation should use the canonical task instruction. Report prompt-search
rollouts separately and include them in the total environment interaction
budget.

## Reproducibility artifacts

A paper result requires the checkpoint revision and checksum, prompt pools,
benchmark revision and task IDs, resolved Hydra configuration, seeds, and
environment-step budget. Publish large or externally licensed artifacts
separately with stable URLs and checksums.
