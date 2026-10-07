"""PDE extensions to RLinf policy models."""

from __future__ import annotations

from typing import Any

import torch
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
)


class PDEOpenPiActionModel(OpenPi0ForRLActionPrediction):
    """RLinf's OpenPI policy with prompt retokenization for PDE's second pass."""

    def retokenize_prompts_for_second_pass(
        self,
        new_prompts: list[str],
        ref_forward_inputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(new_prompts)
        observations: dict[str, Any] = {"prompt": new_prompts}
        for key in (
            "observation/image",
            "observation/state",
            "observation/wrist_image",
        ):
            if key not in ref_forward_inputs:
                continue
            value = ref_forward_inputs[key]
            observations[key] = (
                value[:batch_size]
                if value.shape[0] == batch_size
                else value[0, :batch_size]
            ).cpu()

        tokenized = self.input_transform(observations, transpose=False)
        device = ref_forward_inputs["tokenized_prompt"].device
        return (
            tokenized["tokenized_prompt"].to(device=device),
            tokenized["tokenized_prompt_mask"].to(device=device),
        )


def promote_openpi_model(model: Any) -> Any:
    """Promote an RLinf-created OpenPI instance to the PDE subclass in place."""
    if isinstance(model, OpenPi0ForRLActionPrediction) and not isinstance(
        model, PDEOpenPiActionModel
    ):
        model.__class__ = PDEOpenPiActionModel
    return model
