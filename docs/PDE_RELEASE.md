# PDE release guide

PDE has a strict two-stage contract. Prompt discovery runs against a fixed VLA
checkpoint and writes JSON artifacts. RL training consumes those artifacts
without mutating them. This separation makes the interaction budget auditable
and lets users reproduce training without repeating paid VLM inference.

## Stage 1: VLM-guided prompt discovery

Create a `PromptPool` for each task and provide two adapters to
`discover_prompt_pool`:

- a rollout evaluator that runs the frozen policy for `N` episodes and returns
  rewards plus a one-sentence behavior summary;
- a VLM supervisor that reads the canonical prompt, successful pool, and full
  positive/negative history and proposes `K` new prompts.

The paper defaults are ten iterations, five candidates per iteration, ten
rollouts per candidate, and admission for empirical success greater than zero.
The policy checkpoint must remain frozen for the entire search. The artifact
records its identifier because pools are policy-specific.

```python
from pde import DiscoveryConfig, PromptPool, discover_prompt_pool

pool = PromptPool(
    task_id="libero_90.close_the_microwave",
    canonical_prompt="close the microwave",
    policy_checkpoint="org/pi05-weak-sft",
    environment="LIBERO-90",
)
pool = discover_prompt_pool(
    pool,
    evaluator=rollout_evaluator,
    supervisor=vlm_supervisor,
    config=DiscoveryConfig(),
)
pool.save("prompt_pools/libero_90.close_the_microwave.json")
```

The production VLM client, video encoding, and rollout relabeling utilities are
in `rlinf/workers/actor/her_vlm_client.py`, `her_utils.py`, and
`her_processor.py`. These modules retain their historical `HER` names; the
paper-facing `pde` package owns the stable public vocabulary and artifact
contract.

## Stage 2: RL with the frozen pool

At the start of an episode, sample one instruction and hold it fixed throughout
the rollout. Let `s_bar` be the exponential moving average of success under the
canonical prompt. The canonical sampling probability is

```text
alpha = clip(s_bar / consolidation_target, minimum, 1)
```

with paper defaults `consolidation_target=0.5` and `minimum=0.05`. When a task
has no admitted prompt, sampling falls back to the canonical prompt.

For a trajectory collected under exploratory prompt `p`, run the current policy
under both `p` and canonical prompt `p_g` and use

```text
mixed_logp = 0.5 * log pi(a | o, p) + 0.5 * log pi(a | o, p_g)
ratio = exp(mixed_logp - old_logp(a | o, p))
```

inside the clipped PPO objective. Evaluation always uses `p_g`. The historical
RLinf implementation is centered in `her_dual_actor_worker.py`; model-specific
rollout and log-probability code lives under `rlinf/models/embodiment/`.

## Reproducing the paper

The first supported release target is pi0.5 on LIBERO-PRO. A full result requires
the exact weak SFT checkpoint, benchmark revision, task split, prompt pools,
training configuration, seed, and environment-step budget. Report prompt-search
rollouts separately and include them in total environment interactions.

The smallest end-to-end check is the LIBERO-90 `close the microwave` task:

1. evaluate the frozen weak checkpoint under the canonical instruction;
2. build a prompt pool with the frozen checkpoint;
3. train standard PPO and PDE from the same initialization and rollout budget;
4. evaluate both under `close the microwave` only.

ManiSkill, GR00T, pi0, real-robot, and LLM experiments are secondary release
targets. Their research code remains in the tree where applicable, but they are
not part of the initial reproducibility claim.

## Artifact validation

```bash
pde validate-pool prompt_pools/task.json
```

The validator checks schema version, reward counts, thresholds, and duplicate
prompts. Prompt pools should contain no videos, local paths, API keys, or model
credentials.
