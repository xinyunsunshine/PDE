# Prompt-Driven Exploration

Code for **Prompt-Driven Exploration: Language as an Exploration Space for VLA
Reinforcement Learning**.

PDE discovers prompts that elicit useful behavior from a frozen VLA, then trains
with a fixed prompt pool while transferring that behavior to the original task
instruction. Evaluation always uses the original instruction.

The method lives in `pde/`: `PDEActor` is our PPO actor, inheriting RLinf's
`EmbodiedFSDPActor`. The unmodified `RLinf/` submodule supplies distributed
execution, policy loading, simulation, PPO and optimization.

## Interactive microwave demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xinyunsunshine/PDE/blob/main/notebooks/microwave_prompt_demo.ipynb)

Change the VLA's instruction, compare two microwave rollouts from the same
initial state, and ask a VLM to summarize the behavior and suggest prompts.
The notebook uses one A100/L4 GPU for a frozen pi0.5 policy and an optional
OpenAI or externally hosted Qwen endpoint for feedback. It defaults to RLinf's
public checkpoint; outcomes depend on the checkpoint and prompt. The real
rollout and Qwen-feedback sequence has been tested on one H100; Google-hosted
Colab itself remains untested. See the [recorded validation results](notebooks/microwave_validation.json).

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

Run discovery once per task, with fixed policy weights. Both providers use the
same video summaries, proposal loop and prompt-pool format.

**OpenAI API** (requires `OPENAI_API_KEY`):

```bash
export OPENAI_API_KEY=...
python -m pde.search \
  search.provider=openai search.vlm_model=gpt-4.1 \
  actor.model.model_path=/path/to/weak-pi05 \
  search.task_index=0 search.task_id=libero_object.task_0 \
  search.output=prompt_pools/task_0.json
```

**Local Qwen-VL**, served through an OpenAI-compatible server such as vLLM.
In a separate serving environment with sufficient GPU memory, for example:

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --host 127.0.0.1 --port 8000 \
  --limit-mm-per-prompt '{"image":80}'
```

Then launch discovery in the RLinf environment:

```bash
python -m pde.search \
  search.provider=local_qwen search.vlm_model=Qwen/Qwen3-VL-8B-Instruct \
  search.base_url=http://127.0.0.1:8000/v1 \
  actor.model.model_path=/path/to/weak-pi05 \
  search.task_index=0 search.task_id=libero_object.task_0 \
  search.output=prompt_pools/task_0.json
```

Local serving does not require an OpenAI key. Set `QWEN_API_KEY` only if your
server requires authentication. `QWEN_BASE_URL` is an alternative to
`search.base_url`; `OPENAI_BASE_URL` is not used by either provider. The OpenAI
provider always calls the official endpoint. Match `search.vlm_model` to the
server's model ID. Local here means a separate inference server, not loading
Qwen into the simulator process.

The default provider is `local_qwen`; its default model remains
`Qwen/Qwen3-VL-235B-A22B-Thinking-FP8`. The smaller 8B example and OpenAI option
are alternative discovery configurations, not equivalent paper experiments.
The server must accept at least `rollouts_per_candidate * frames_per_video`
images per request (80 with the defaults), with enough context for those images.
Set `search.temperature=null` for models that do not accept temperature.
Provider, resolved model ID and temperature are recorded in each pool.
See the [OpenAI vision guide](https://developers.openai.com/api/docs/guides/images-vision)
and [Qwen serving instructions](https://github.com/QwenLM/Qwen3-VL#deployment)
for API and server setup.

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

## Citation

```bibtex
@article{jiang2026promptdriven,
  title={Prompt-Driven Exploration: Language as an Exploration Space for VLA Reinforcement Learning},
  author={Jiang, Sunshine and Marangola, John and Zhang, David and Kowdeed, Raghuram and Luo, Ruiyang and Dashora, Nitish and Li, Richard and Agrawal, Pulkit and Hong, Zhang-Wei},
  journal={NeurIPS},
  year={2026}
}
```

Also available as [CITATION.bib](CITATION.bib). PDE and RLinf use Apache-2.0;
models and benchmarks retain their own licenses.
