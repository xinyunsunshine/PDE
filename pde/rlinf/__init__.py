"""RLinf integration for Prompt-Driven Exploration.

The namespace stays lazy because importing a concrete RLinf model can initialize
optional simulator and accelerator dependencies. Import extension classes from
their defining modules.
"""

__all__ = [
    "PDEActor",
    "PDEDualActor",
    "PDEEnvWorker",
    "PDELiberoEnv",
    "PDERunner",
]
