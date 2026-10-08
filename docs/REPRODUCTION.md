# Reproducing PDE

## What is implemented and verified

The release implements the VLA method in Sections 4.1–4.2 and Appendix A:
fixed-policy prompt discovery followed by PPO with a frozen pool, canonical
success EMA, and averaged current-policy log likelihoods. It targets pi0.5 on
LIBERO and LIBERO-PRO. It does not claim to reproduce the ManiSkill, real-robot,
or LLM results.

The previous release used HER instruction/reward relabeling and two separately
weighted PPO losses. That was not the paper's method. It also depended on actor
hooks absent from the pinned RLinf. Those paths have been removed.

CPU tests cover config composition, gradient flow through both prompt
likelihoods, original rollout likelihood preservation, real RLinf batch
preprocessing, canonical-only EMA updates, episode prompt persistence, evaluation
prompts and curriculum checkpointing. No full simulator or GPU training run has
yet established numerical agreement with the paper.

## Required experiment inputs

The paper-result inputs still need to be supplied: the exact weak pi0.5 weights
and normalization assets with an immutable revision/checksum; the generated
per-task prompt-pool JSON files; the LIBERO-PRO package/assets revision and
perturbation task list; and each run's seeds, horizon, environment count,
placement, evaluation schedule and total interaction budget.

The defaults in `pde/configs/libero.yaml` encode Table 3's learning rate
5e-6, batch size 2048, PPO clipping 0.2, GAE 0.95, discount 0.99, four update
epochs, EMA smoothing 0.3, canonical floor 0.05 and consolidation target 0.5.
Other settings inherit the pinned RLinf LIBERO-object PPO config and must be
matched to the archived experiment. A successful run with these defaults alone
does not establish reproduction of a reported score.

## Discovery

Use `python -m pde.search` with the task index, task identifier, checkpoint and
output path as shown in the README. Discovery uses ten parallel episodes for
each prompt, including a canonical-prompt evaluation before candidate search.
That initial evaluation is recorded in `metadata.canonical_evaluation`.
Candidate rollout counts are recorded individually. Include both counts in
the prompt-search interaction budget.

The supervisor uses the paper's feedback structure but the release prompt text
is a reconstruction, not an archived API transcript. Deterministic reproduction
of RL should use the archived frozen pools; regenerating them through a live
VLM is not expected to produce identical prompts.

A pool is keyed by the exact canonical BDDL instruction. Discovery fails if a
single batch contains different canonical instructions. Run separate searches
for such perturbation variants. Pool files keep the checkpoint identifier used
during discovery; training rejects a mismatch. When moving a checkpoint to a
new path, preserve a consistent identifier/path across pool creation and
training.

## Training and evaluation

Run `python -m pde.train` for standard LIBERO or add
`--config-name libero_pro`. Set `env.train.filter_task_ids=[...]` to the
archived subset; eval inherits the same subset and benchmark variant. Set the
evaluation environment count to 250 times the number of selected tasks.
The example defaults use all ten tasks and 2,500 eval environments. This can
require substantial simulator memory. Match the archived evaluation schedule
and adjust parallel environment count and `algorithm.eval_rollout_epoch`
together to retain the desired number of episodes. Environment counts must be
divisible by the worker count times `rollout.pipeline_stage_num`; configure
placement explicitly when moving between GPU counts.

Prompts are sampled once per environment per rollout epoch. This release
requires `auto_reset=false` and a rollout epoch equal to the full episode
horizon. Canonical sampling is task-specific. Across actor ranks, canonical
successes and episode counts are summed before the EMA update; tasks with no
canonical sample retain their prior EMA.

The original environment supplies all rewards. The same observed actions and
diffusion noise chain are scored under sampled and canonical prompts. Their
average log likelihood goes into one RLinf PPO ratio. Value and entropy terms
use the behavior prompt. Evaluation bypasses prompt sampling entirely.

Use `pde.enabled=false` for the baseline, preserving the checkpoint, task
subset, optimizer settings and interaction budget. Use `runner.only_eval=true`
with a trained `runner.ckpt_path` for canonical-only evaluation; the
run manifest still records the matching frozen pools. Evaluation uses RLinf’s
`EmbodiedEvalRunner` and plain rollout worker; it never runs PPO updates.

Training writes the resolved `config.yaml`, pool copies, pool SHA-256 values and
RLinf revision to the experiment log directory. Checkpoints include
`actor/pde.json` with the EMA and a pool fingerprint. Resume through
`runner.resume_dir`; a changed pool fails validation. Prompt sampling uses
seed, rank, training step and rollout epoch. RLinf's simulator state is not
checkpointed, so resume does not promise bitwise trajectory equivalence.

## Environment setup

Install OpenPI and the simulator with RLinf's pinned embodied installer. The
LIBERO-PRO Python package and assets are separate dependencies. The worker sets
`LIBERO_TYPE` before importing LIBERO; train and eval must use the same variant.
PRO task selection validates that the requested perturbed BDDL actually exists,
reads its language field and avoids applying standard LIBERO initial states to
PRO/Plus tasks.

RLinf is read-only: runtime startup checks its Git revision, a clean submodule,
and Python's import location. This prevents accidentally running an older
editable checkout while validating this release.
