# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HER ablation dual actor workers: rephrase-only, reward-eval-only, random-reward.

EmbodiedFSDPActorHERDualRephraseOnly:
    Rephrases the original instruction via VLM (text-only, no trajectory video)
    and re-tokenizes prompts. Keeps original env rewards (no VLM reward eval).

EmbodiedFSDPActorHERDualRewardEvalOnly:
    Keeps original instructions on the HER branch, but runs VLM reward eval
    against original instructions and patches rewards (no rephrase).

EmbodiedFSDPActorHERDualRandomReward:
    Assigns random binary rewards (0 or 1) to each trajectory on the HER branch.
    No VLM calls. Rewards placed at the correct terminal timestep.

Config dispatch keys (under ``actor:``):
    her_dual_rephrase_only: {}
    her_dual_reward_eval_only: {}
    her_dual_random_reward: {}
"""

from rlinf.workers.actor.her_ablation_processor import (
    RandomRewardProcessor,
    RephraseOnlyProcessor,
    RewardEvalOnlyProcessor,
)
from rlinf.workers.actor.her_dual_separate_grad_actor_worker import (
    EmbodiedFSDPActorHERDualSeparateGrad,
)


class EmbodiedFSDPActorHERDualRephraseOnly(EmbodiedFSDPActorHERDualSeparateGrad):
    """HER dual variant: rephrase instructions only, keep original env rewards."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._her_processor = RephraseOnlyProcessor(
            cfg=cfg,
            rank=self._rank,
            get_model_fn=lambda: self.model,
            log_warning_fn=self.log_warning,
        )


class EmbodiedFSDPActorHERDualRewardEvalOnly(EmbodiedFSDPActorHERDualSeparateGrad):
    """HER dual variant: VLM reward eval on original instructions, no rephrase."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._her_processor = RewardEvalOnlyProcessor(
            cfg=cfg,
            rank=self._rank,
            get_model_fn=lambda: self.model,
            log_warning_fn=self.log_warning,
        )


class EmbodiedFSDPActorHERDualRandomReward(EmbodiedFSDPActorHERDualSeparateGrad):
    """HER dual variant: random binary rewards, no VLM calls."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._her_processor = RandomRewardProcessor(
            cfg=cfg,
            rank=self._rank,
            get_model_fn=lambda: self.model,
            log_warning_fn=self.log_warning,
        )
