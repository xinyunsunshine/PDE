# Copyright 2026 The RLinf Authors.
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
"""OpenPI DataConfig for 10-dim absolute-EE FR3 VLA (pi0.5).

State / action layout: ``[xyz(3), rot6d(6), gripper(1)]``.
"""

import dataclasses
import pathlib

import einops
import numpy as np
import openpi.models.model as _model
import openpi.transforms as _transforms
import torch
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

FR3_VLA_DIM = 10


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FR3VLAInputs(_transforms.DataTransformFn):
    """Pack a FR3 LeRobot frame into the pi0.5 model input format."""

    action_dim: int

    def __call__(self, data: dict) -> dict:
        state = data["observation/state"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        state = _transforms.pad_to_dim(state, self.action_dim)

        base = _parse_image(data["observation/image"])
        zeros = np.zeros_like(base)
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": zeros,
                "right_wrist_0_rgb": zeros,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = _transforms.pad_to_dim(data["actions"], self.action_dim)
        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = (
                prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
            )
        return inputs


@dataclasses.dataclass(frozen=True)
class FR3VLAOutputs(_transforms.DataTransformFn):
    """Slice the model's padded output back to 10 dims (inference only)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :FR3_VLA_DIM])}


@dataclasses.dataclass(frozen=True)
class FR3VLADataConfig(DataConfigFactory):
    default_prompt: str | None = None
    # Dataset stores absolute actions; push DeltaActions at training time so
    # the policy learns deltas for xyz+rot6d and absolute for gripper.
    extra_delta_transform: bool = False

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.global",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[FR3VLAInputs(action_dim=model_config.action_dim)],
            outputs=[FR3VLAOutputs()],
        )

        if not self.extra_delta_transform:
            # xyz(3) + rot6d(6) delta; gripper(1) absolute.
            mask = _transforms.make_bool_mask(9, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(mask)],
                outputs=[_transforms.AbsoluteActions(mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        base = self.create_base_config(assets_dirs, model_config)
        if base.norm_stats is None:
            assets_root = self.assets.assets_dir or assets_dirs
            raise FileNotFoundError(
                f"Norm stats missing for {self.repo_id!r} under {assets_root}. "
                f"Compute them with:\n"
                f"  HF_LEROBOT_HOME=<lerobot_root> uv run python "
                f"external/openpi/scripts/compute_norm_stats.py pi05_fr3vla"
            )
        return dataclasses.replace(
            base,
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )
