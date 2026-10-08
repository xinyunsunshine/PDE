"""PDE curriculum synchronization using RLinf's runner hooks."""

from rlinf.runners.embodied_runner import EmbodiedRunner


class PDERunner(EmbodiedRunner):
    """Synchronize the globally aggregated curriculum alongside policy weights."""

    def update_rollout_weights(self):
        super().update_rollout_weights()
        states = self.actor.get_curriculum().wait()
        self.rollout.set_curriculum(states[0]).wait()
