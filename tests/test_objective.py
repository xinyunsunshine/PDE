import pytest
import torch

from pde.objective import canonical_tokens, mixed_forward


def test_mixed_ratio_backpropagates_to_both_prompts():
    logits = torch.tensor([-0.6, -0.8], requires_grad=True)
    old_logp = torch.tensor(-0.7)
    chains = torch.tensor([123.0])
    calls = []

    def forward(forward_inputs, **kwargs):
        assert forward_inputs["chains"] is chains
        assert "canonical_prompt" not in forward_inputs
        index = int(forward_inputs["tokenized_prompt"].item())
        calls.append((index, kwargs["compute_values"]))
        return {"logprobs": logits[index], "values": torch.tensor(float(index))}

    inputs = {
        "tokenized_prompt": torch.tensor([0]),
        "tokenized_prompt_mask": torch.tensor([True]),
        "canonical_prompt": torch.tensor([1]),
        "canonical_prompt_mask": torch.tensor([True]),
        "chains": chains,
    }
    output = mixed_forward(forward, inputs, compute_values=True)
    ratio = (output["logprobs"] - old_logp).exp()
    assert ratio.item() == pytest.approx(1)
    loss = -torch.minimum(ratio, ratio.clamp(0.8, 1.2))
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([-0.5, -0.5]))
    assert calls == [(0, True), (1, False)]
    assert output["values"].item() == 0
    assert "canonical_prompt" in inputs  # caller batch is not mutated


def test_missing_canonical_tokens_fails():
    with pytest.raises(ValueError, match="canonical"):
        mixed_forward(lambda **kw: {}, {})


def test_retokenization_keeps_flat_state_batch():
    class Model:
        def input_transform(self, observations, transpose):
            assert observations["observation/state"].shape == (2, 3)
            assert observations["prompt"] == ["a", "b"]
            return {
                "tokenized_prompt": torch.ones(2, 4),
                "tokenized_prompt_mask": torch.ones(2, 4, dtype=torch.bool),
            }

    tokens, mask = canonical_tokens(
        Model(),
        ["a", "b"],
        {"observation/state": torch.zeros(2, 3), "tokenized_prompt": torch.zeros(2, 4)},
    )
    assert tokens.shape == mask.shape == (2, 4)
