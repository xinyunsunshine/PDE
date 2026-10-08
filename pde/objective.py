"""Tensor operations for the paper's mixed-likelihood PPO objective."""

from pde.training import mixed_log_probability


def mixed_forward(base_forward, forward_inputs, **kwargs):
    """Evaluate the same actions/noise chain under both prompts, with gradients.

    Values and entropy follow the behavior prompt. Only the current-policy
    likelihood changes; RLinf retains rollout log probabilities and task rewards.
    """
    inputs = dict(forward_inputs)
    tokens = inputs.pop("canonical_prompt", None)
    mask = inputs.pop("canonical_prompt_mask", None)
    inputs.pop("pde_task", None)
    inputs.pop("pde_canonical", None)
    if tokens is None or mask is None:
        raise ValueError("PDE training requires canonical tokens from PDERolloutWorker")
    result = base_forward(forward_inputs=inputs, **kwargs)
    canonical = dict(inputs, tokenized_prompt=tokens, tokenized_prompt_mask=mask)
    canonical_kwargs = dict(kwargs, compute_values=False)
    anchored = base_forward(forward_inputs=canonical, **canonical_kwargs)
    return dict(
        result, logprobs=mixed_log_probability(result["logprobs"], anchored["logprobs"])
    )


def canonical_tokens(model, prompts, forward_inputs):
    """Retokenize a flat rollout batch, retaining each observation's state."""
    batch = len(prompts)
    observations = {"prompt": prompts}
    for key in ("observation/image", "observation/state", "observation/wrist_image"):
        if key in forward_inputs:
            value = forward_inputs[key]
            if value.shape[0] != batch:
                raise ValueError("Expected flat [batch, ...] rollout observations")
            observations[key] = value.cpu()
    tokenized = model.input_transform(observations, transpose=False)
    device = forward_inputs["tokenized_prompt"].device
    return (
        tokenized["tokenized_prompt"].to(device),
        tokenized["tokenized_prompt_mask"].to(device),
    )
