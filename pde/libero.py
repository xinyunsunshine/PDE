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

"""Task selection and perturbed instructions on RLinf's LIBERO environment."""

import os
import re
from pathlib import Path

import numpy as np
from rlinf.envs.libero.libero_env import LiberoEnv


class PDELiberoEnv(LiberoEnv):
    """Retain the released task subset and use the selected BDDL's instruction."""

    def _compute_total_num_group_envs(self):
        selected = self.cfg.get("filter_task_ids")
        self.active_task_ids = (
            list(range(self.task_suite.get_num_tasks()))
            if selected is None
            else list(selected)
        )
        if not self.active_task_ids or len(set(self.active_task_ids)) != len(
            self.active_task_ids
        ):
            raise ValueError("filter_task_ids must be nonempty and unique")
        if any(
            t < 0 or t >= self.task_suite.get_num_tasks() for t in self.active_task_ids
        ):
            raise ValueError("filter_task_ids includes an invalid task")
        self.trial_id_bins = [
            len(self.task_suite.get_task_init_states(t)) for t in self.active_task_ids
        ]
        self.total_num_group_envs = sum(self.trial_id_bins)
        self.cumsum_trial_id_bins = np.cumsum(self.trial_id_bins)

    def _get_task_and_trial_ids_from_reset_state_ids(self, reset_state_ids):
        task_ids, trial_ids = super()._get_task_and_trial_ids_from_reset_state_ids(
            reset_state_ids
        )
        return np.array([self.active_task_ids[t] for t in task_ids]), trial_ids

    def get_env_fn_params(self, env_idx=None):
        params = super().get_env_fn_params(env_idx)
        indices = range(self.num_envs) if env_idx is None else sorted(env_idx)
        for env_id, param in zip(indices, params):
            path = Path(param["bddl_file_name"])
            if self.cfg.get("libero_variant", "standard") == "pro":
                suffix = self.cfg.get("perturbation_suffix")
                if not suffix or suffix not in ("task", "object", "swap", "lan", "all"):
                    raise ValueError(
                        "LIBERO-PRO requires task, object, swap, lan, or all"
                    )
                if not any(
                    path.parent.name.endswith("_" + p)
                    for p in ("task", "object", "swap", "lan")
                ):
                    raise FileNotFoundError(
                        f"No requested LIBERO-PRO variant found: {path}"
                    )
            match = re.search(r"\(:language\s+(.*?)\)", path.read_text(), re.S | re.I)
            if match:
                self.task_descriptions[env_id] = " ".join(match.group(1).split())
        return params

    def _reconfigure(self, reset_state_ids, env_idx):
        # RLinf's pinned implementation applies standard initial states to PRO.
        # PRO and Plus initialize from their perturbed BDDL instead.
        reconfigure = []
        task_ids, trial_ids = self._get_task_and_trial_ids_from_reset_state_ids(
            reset_state_ids
        )
        for j, env_id in enumerate(env_idx):
            changed = self.task_ids[env_id] != task_ids[j]
            self.task_ids[env_id] = task_ids[j]
            self.trial_ids[env_id] = trial_ids[j]
            if changed or not self.cfg.is_eval:
                reconfigure.append(env_id)
        if reconfigure:
            self.env.reconfigure_env_fns(
                self.get_env_fn_params(reconfigure), reconfigure
            )
        self.env.seed(self.seed * len(env_idx))
        self.env.reset(id=env_idx)
        if os.environ.get("LIBERO_TYPE", "standard") not in ("pro", "plus"):
            self.env.set_init_state(
                init_state=self._get_reset_states(env_idx=env_idx), id=env_idx
            )
