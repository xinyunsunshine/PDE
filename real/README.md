# Real-World Inference: Log-Prob Computation

This folder hosts the real-world deployment path for pi0.5 / OpenPI policies
(`vla_server.py`, `deploy.py`, `vla_client.py`, `policy_loader.py`). This
README documents how log-probabilities are computed in the training stack and
how to surface them in the real-world inference path.

## TL;DR

- The pi0.5 policy is a flow-matching denoiser. The "action distribution" is
  the per-step Gaussian transition `x_{t+1} ~ N(mu_t(x_t, cond), sigma_t)`,
  not a categorical over tokens.
- Training and rollout compute `log_prob` with the same closed-form Gaussian
  primitive `get_logprob_norm` and end up with the same per-sample shape
  `[B, action_chunk, action_env_dim]`. PPO's `exp(logprobs - old_logprobs)`
  is element-wise on that shape.
- The current real-world server (`vla_server.py`) goes through OpenPI's
  native `policy.infer()` and never computes logprobs. To get logprobs at
  serving time, switch the server to RLinf's `predict_action_batch`, which
  already returns them.

## Where log_prob lives in the codebase

All references are to `RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py`.

### The Gaussian primitive (line 710-723)

```python
def get_logprob_norm(self, sample, mu, sigma):
    # log p(x|mu,sigma) = -log(sigma) - 0.5*log(2*pi) - 0.5*((x - mu)/sigma)**2
    if self.config.safe_get_logprob:
        log_prob = -torch.pow((sample - mu), 2)        # mean-only, sigma collapsed
    else:
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
            2 * torch.pi * torch.ones_like(sample)
        )
        exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
        log_prob = constant_term + exponent_term
        log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
    return log_prob
```

`safe_get_logprob=True` drops the variance term and is equivalent to scoring
only the mean — useful for numerical stability, but the absolute log-prob
values are no longer comparable across runs with different sigma.

### Training forward — `default_forward` → `get_log_prob_value`

`default_forward` (line 264-311) is the training path. It receives the
saved rollout `chains` and `denoise_inds` and asks
`get_log_prob_value` (line 728-805) to score them under the current policy.

For each step `idx` in `range(num_steps)`:

```python
chains_pre  = chains[:, denoise_ind]        # state before the denoise step
chains_next = chains[:, denoise_ind + 1]    # state after the step (the "action")
x_t_mean, x_t_std, value_t = self.sample_mean_var_val(chains_pre, ...)
log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
```

Two modes via `config.joint_logprob`:

- `joint_logprob=True`  — score every step in the chain, plus the prior
  `log N(x_0 | 0, I)` (line 765-772).
- `joint_logprob=False` — `num_steps = 1`, only the chosen step is scored
  per sample (typical PPO mode).

Post-processing in `default_forward` (line 295-302):

```python
log_probs = log_probs[:, :, :action_chunk, :action_env_dim]   # crop env-relevant
log_probs = log_probs.mean(dim=1)                              # mean over chain dim
return {"logprobs": log_probs, "values": value_t, "entropy": entropy}
```

Final shape: `[B, action_chunk, action_env_dim]`.

### Rollout / inference — `predict_action_batch` → `sample_actions`

`predict_action_batch` (line 391-431) is the rollout/inference entry point.
It calls `sample_actions` (line 434-554) which runs the denoising loop and
returns logprobs *for the actually sampled trajectory*:

```python
for idx in range(num_steps):
    # ... denoising step computes x_t_mean, x_t_std, value_t ...
    x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
    log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)
    chains.append(x_t)
    log_probs.append(log_prob)
```

Post-processing (line 532-541):

```python
log_probs = torch.stack(log_probs, dim=1)[:, :, :action_chunk, :action_env_dim]
if self.config.joint_logprob:
    log_probs = log_probs.mean(dim=1)
else:
    log_probs = log_probs[torch.arange(B), denoise_inds[:, 0]]
```

Final shape: `[B, action_chunk, action_env_dim]` — same as training, by
construction. This is what gets stored as `prev_logprobs` and later compared
against the training `logprobs` in PPO's ratio (`losses.py:75`).

## Per-sample log-prob shape — what training consumes

Both `prev_logprobs` (rollout) and `logprobs` (training forward) are shape
`[B, action_chunk, action_env_dim]`. PPO's clipped ratio runs element-wise:

```python
# rlinf/algorithms/losses.py:75
ratio = torch.where(loss_mask, torch.exp(logprobs - old_logprobs), 0)
```

The loss_mask broadcasts to that shape. There is no further reduction to a
scalar log-prob per sample before the ratio — every action dim and every
chunk index contributes its own ratio term.

## Rollout-side knob: `collect_prev_infos`

`huggingface_worker.py:248-250` only forwards logprobs upstream when
`rollout.collect_prev_infos: True`:

```python
prev_logprobs = result["prev_logprobs"] if cfg.rollout.collect_prev_infos else None
```

In `examples/embodiment/config/realworld_sac_flow_image.yaml` this is
currently `False` because SAC doesn't need them. Set to `True` for any
PPO-style real-world training.

## Adding log_prob to real-world inference

Two cases — pick by which path the server uses today.

### Case A — server already uses `predict_action_batch`

Nothing to add. `result["prev_logprobs"]` is already populated:

```python
with torch.no_grad():
    actions, result = model.predict_action_batch(env_obs=env_obs, mode="eval")
logprobs = result["prev_logprobs"]   # [B, action_chunk, action_env_dim]
```

### Case B — server uses OpenPI's native `policy.infer()` (current state)

`real/vla_server.py:57` calls `policy.infer(warmup_obs)`. That bypasses
RLinf and never computes logprobs. Two options:

**Option 1 (recommended): switch the server to `predict_action_batch`.**

Replace the OpenPI policy load with the RLinf wrapper
(`OpenPi0ForRLActionPrediction`) and in the request handler:

```python
env_obs = {
    "main_images": ...,        # [B, H, W, 3] uint8 or float
    "wrist_images": ...,       # or None
    "states": ...,              # [B, state_dim]
    "task_descriptions": [...], # list[str]
}
with torch.no_grad():
    actions, result = model.predict_action_batch(env_obs=env_obs, mode="eval")
return {
    "actions": actions.cpu().numpy(),
    "logprobs": result["prev_logprobs"].cpu().numpy(),
}
```

This reuses the exact computation training sees, so logprobs are guaranteed
consistent across rollout/train/serve.

**Option 2 (not recommended): keep OpenPI sampler, score offline.**

Capture the raw `chains` and `denoise_inds` from inside OpenPI's sampler and
call `model.get_log_prob_value(images, img_masks, lang_tokens, lang_masks,
state, chains, denoise_inds)` afterward. This requires patching OpenPI to
expose the chain — fragile and easy to drift from the training-time
computation. Avoid unless option 1 is blocked.

## Config knobs that affect log_prob values

In `OpenPi0Config` (line 34-68 of `openpi_action_model.py`):

| Field | Effect |
|---|---|
| `safe_get_logprob` | `True` collapses sigma → log_prob = `-(x-mu)^2`. Numerical stability at the cost of comparability. |
| `joint_logprob` | `True` scores every denoise step + prior; `False` scores one sampled step. |
| `noise_method` | `flow_sde` / `flow_noise` / `flow_cps` — changes how `x_t_std` is set, which directly shifts log_prob. Entropy is only meaningful for `flow_noise`. |
| `num_steps` | Number of denoising steps; with `joint_logprob=True` log_probs sum more terms. |
| `action_chunk`, `action_env_dim` | Crop applied before the final mean — controls which dims of the action contribute. |

Two runs with different `noise_method` or `safe_get_logprob` cannot have
their logprobs directly compared.

## File map

| Concern | File | Lines |
|---|---|---|
| Gaussian log-prob primitive | `rlinf/models/embodiment/openpi/openpi_action_model.py` | 710-723 |
| Training forward | `rlinf/models/embodiment/openpi/openpi_action_model.py` | 264-311 |
| `get_log_prob_value` (per-step scoring) | `rlinf/models/embodiment/openpi/openpi_action_model.py` | 728-805 |
| Rollout sampler with logprobs | `rlinf/models/embodiment/openpi/openpi_action_model.py` | 434-554 |
| Rollout `predict_action_batch` entry | `rlinf/models/embodiment/openpi/openpi_action_model.py` | 391-431 |
| HF rollout worker pulls `prev_logprobs` | `rlinf/workers/rollout/hf/huggingface_worker.py` | 136, 248-250 |
| PPO ratio uses (logprobs - old_logprobs) | `rlinf/algorithms/losses.py` | 75 |
| Real-world WebSocket server (no logprobs) | `real/vla_server.py` | 57 |
| Real-world deploy client (no logprobs) | `real/deploy.py` | 149-151 |
