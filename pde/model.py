"""PDE's policy specialization of the pinned RLinf OpenPI implementation."""

from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
)

from pde.objective import mixed_forward


class PDEOpenPiActionModel(OpenPi0ForRLActionPrediction):
    """Use Eq. (3) of the paper in RLinf's existing PPO training loop."""

    def default_forward(self, forward_inputs, **kwargs):
        return mixed_forward(super().default_forward, forward_inputs, **kwargs)


def extend_model(model):
    """Preserve RLinf's loaded weights, transforms, LoRA and state-dict names.

    This stateless subclass has the same instance layout as its RLinf base.
    Promoting this individual model avoids modifying RLinf's global factories.
    """
    if not isinstance(model, OpenPi0ForRLActionPrediction):
        raise TypeError("The released PDE training path requires RLinf OpenPI")
    model.__class__ = PDEOpenPiActionModel
    return model
