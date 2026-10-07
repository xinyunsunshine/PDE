"""PDE runner specialization."""

from rlinf.runners.embodied_runner import EmbodiedRunner


class PDERunner(EmbodiedRunner):
    """RLinf embodied runner used to make the PDE extension boundary explicit."""
