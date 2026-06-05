"""Offline round-trip of dataset actions through the VLA server's transform chain.

The openpi ``Policy`` loaded by ``real/policy_loader.py:74-90`` wraps the pi0
model with a transform pipeline that pads state/actions to 32 dims, converts
absolute actions to delta (subtracting state), normalizes with stats from
``<checkpoint>/<asset_id>/norm_stats.json``, runs the model, and then
inverts those transforms on the output. ``real/deploy.py`` never sees any of
that — it passes raw 10-d observations in and receives raw 10-d actions out.

This helper builds a function that runs the action-side half of that chain
offline, without loading the model weights. It applies the INPUT transforms
through ``Normalize`` (forward), then immediately ``Unnormalize`` and the
OUTPUT transforms (backward). If the transforms are self-inverse, the raw
10-d action coming out equals the raw 10-d action that went in.

This is used by ``real/replay_episode.py`` to verify the pipeline: we round
trip every action in a recorded dataset episode and either execute the
round-tripped version on the robot, or just compare against the original
numerically. Divergences localize bugs to the transform chain rather than
to the policy or the robot side.
"""

from __future__ import annotations

import pathlib
from typing import Callable

import numpy as np
from openpi import transforms as _transforms
from openpi.shared import normalize as _normalize

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config


RoundTripFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def build_round_trip_fn(
    checkpoint_dir: str,
    config_name: str,
) -> RoundTripFn:
    """Return ``(state_10d, action_10d, image_hwc) -> round_tripped_action_10d``.

    Mirrors the first half of ``real.policy_loader.load_pi0_policy`` but skips
    the ``pi0_pytorch.PI0Pytorch`` model construction — only the transform
    groups and norm stats are needed here.

    Args:
        checkpoint_dir: Same layout ``load_pi0_policy`` expects — a directory
            that contains ``<asset_id>/norm_stats.json`` (produced by
            ``compute_norm_stats.py`` at training time). ``full_weights.pt``
            does *not* need to be present; we never load model weights.
        config_name: Registered openpi TrainConfig name, e.g. ``pi05_fr3vla``.
            Determines which dataconfig (and therefore which ``data_transforms``
            + ``asset_id``) to use.

    Returns:
        A closure that takes a single (state, action, image) triple and
        returns the action after a ``Normalize`` + ``Unnormalize`` round trip
        through the FR3VLA input/output transform groups. Shapes:
        ``state_10d`` and ``action_10d`` are ``(10,)`` float32;
        ``image_hwc`` is ``(H, W, 3)`` uint8 (only used to satisfy the
        ``FR3VLAInputs`` contract; its values don't affect the action math).
    """
    ckpt = pathlib.Path(checkpoint_dir).expanduser().resolve()
    train_config = get_openpi_config(config_name, model_path=str(ckpt))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError(
            f"DataConfig for '{config_name}' has no asset_id — cannot locate norm stats."
        )
    norm_stats = _normalize.load(ckpt / data_config.asset_id)

    # The two Group halves we need. Note: we deliberately skip
    # model_transforms (TokenizePrompt / image preprocessing) — they don't
    # touch the action tensor, and skipping them means we don't need a real
    # tokenizer or image in the right shape.
    in_transforms = list(data_config.data_transforms.inputs)   # [FR3VLAInputs, DeltaActions]
    out_transforms = list(data_config.data_transforms.outputs)  # [AbsoluteActions, FR3VLAOutputs]
    normalize = _transforms.Normalize(
        norm_stats, use_quantiles=data_config.use_quantile_norm
    )
    unnormalize = _transforms.Unnormalize(
        norm_stats, use_quantiles=data_config.use_quantile_norm
    )

    def round_trip(
        state_10d: np.ndarray,
        action_10d: np.ndarray,
        image_hwc: np.ndarray,
    ) -> np.ndarray:
        # A chunk-of-one action so DeltaActions' broadcast math works
        # (it does `state[..., :dims] * mask` expanded to `(1, dims)` and
        # subtracts from a sequence axis on actions).
        data: dict = {
            "observation/state": np.asarray(state_10d, dtype=np.float32).copy(),
            "observation/image": np.asarray(image_hwc, dtype=np.uint8),
            "actions": np.asarray(action_10d, dtype=np.float32).copy()[None, :],
            "prompt": "",
        }
        for t in in_transforms:
            data = t(data)
        data = normalize(data)
        # Normalize mutates "state" and "actions" in place. Unnormalize will
        # restore both using the same stats, so AbsoluteActions below reads
        # back the raw 32-d state and adds it to the denormalized delta.
        data = unnormalize(data)
        for t in out_transforms:
            data = t(data)
        actions = np.asarray(data["actions"], dtype=np.float32)
        # FR3VLAOutputs gives (1, 10); drop the chunk axis.
        return actions[0]

    return round_trip