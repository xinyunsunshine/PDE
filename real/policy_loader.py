"""Loader for a pi0(.5) SFT checkpoint as an openpi Policy.

Extracted from real/deploy.py so both the in-process deploy path and the
standalone VLA inference server (real/vla_server.py) can share the same
checkpoint-loading logic.
"""

from __future__ import annotations

import pathlib
import tempfile

import torch
from openpi import transforms as _transforms
from openpi.models_pytorch import pi0_pytorch
from openpi.policies import policy as _policy
from openpi.shared import normalize as _normalize

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config


def load_pi0_policy(
    config_name: str,
    checkpoint_dir: str,
    default_prompt: str | None = None,
    pytorch_device: str | None = "cuda:0",
) -> _policy.Policy:
    """Load an OpenPI pi0(.5) policy from an RLinf SFT checkpoint directory.

    Args:
        config_name: Name of the registered TrainConfig (e.g. pi05_fr3vla).
        checkpoint_dir: Path to a directory containing ``full_weights.pt``
            (the RLinf FSDP state dict) and an ``assets/`` subdirectory with
            ``norm_stats.msgpack``.
        default_prompt: Fallback prompt if the obs dict lacks one.
        pytorch_device: Torch device override (e.g. cuda:0).
    """
    ckpt = pathlib.Path(checkpoint_dir).expanduser().resolve()
    weights_file = ckpt / "full_weights.pt"
    if not weights_file.is_file():
        raise FileNotFoundError(f"Checkpoint dir '{ckpt}' is missing full_weights.pt")
    if not (ckpt / "assets").is_dir():
        raise FileNotFoundError(
            f"Checkpoint dir '{ckpt}' is missing 'assets/' with norm stats. "
            "OpenPI expects the same assets layout as training."
        )

    train_config = get_openpi_config(config_name, model_path=str(ckpt))

    model = pi0_pytorch.PI0Pytorch(config=train_config.model)
    state_dict = torch.load(str(weights_file), map_location="cpu", weights_only=False)
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(
            f"[policy_loader] WARNING: {len(result.missing_keys)} missing keys after load "
            f"(first 5): {list(result.missing_keys)[:5]}"
        )
    if result.unexpected_keys:
        print(
            f"[policy_loader] {len(result.unexpected_keys)} unexpected keys dropped from "
            f"checkpoint (first 5): {list(result.unexpected_keys)[:5]}"
        )
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    # Inlined from openpi.training.checkpoints.load_norm_stats to avoid that
    # module's lerobot-dependent import chain (we only need the norm stats).
    # import pdb;pdb.set_trace()

    norm_stats = _normalize.load(pathlib.Path(ckpt) / data_config.asset_id)
    assert norm_stats is not None, "Need normlization statistics to load policy."

    print("### Norm stats loaded ###")
    print(norm_stats)
    print("#" * 30)

    if pytorch_device is None:
        pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"

    return _policy.Policy(
        model,
        transforms=[
            _transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        metadata=train_config.policy_metadata,
        is_pytorch=True,
        pytorch_device=pytorch_device,
    )


def load_pi0_policy_direct(
    config_name: str,
    weights_path: str,
    norm_stats_path: str,
    default_prompt: str | None = None,
    pytorch_device: str | None = "cuda:0",
) -> _policy.Policy:
    """Same as ``load_pi0_policy`` but takes absolute paths to the weights
    and ``norm_stats.json`` files directly, so callers don't have to stage
    them in nested format.

    Under the hood this builds that layout in a tempdir and reuses the
    existing loader path so the resulting Policy is byte-identical.
    """
    weights_file = pathlib.Path(weights_path).expanduser().resolve()
    norm_file = pathlib.Path(norm_stats_path).expanduser().resolve()
    if not weights_file.is_file():
        raise FileNotFoundError(f"weights file not found: {weights_file}")
    if not norm_file.is_file():
        raise FileNotFoundError(f"norm stats file not found: {norm_file}")

    stage_root = pathlib.Path(tempfile.mkdtemp(prefix="pi0_policy_stage_"))
    (stage_root / "full_weights.pt").symlink_to(weights_file)
    (stage_root / "assets").mkdir()  # satisfies the sanity check in load_pi0_policy

    # Resolve the asset_id openpi will expect (falls back to repo_id when the
    # DataConfig doesn't pin one explicitly
    probe_config = get_openpi_config(config_name, model_path=str(stage_root))
    asset_id = probe_config.data.assets.asset_id or probe_config.data.repo_id
    if asset_id is None:
        raise ValueError(
            f"Config '{config_name}' has neither assets.asset_id nor repo_id set; "
            "cannot determine where to stage norm_stats.json."
        )
    (stage_root / asset_id).mkdir(parents=True, exist_ok=True)
    (stage_root / asset_id / "norm_stats.json").symlink_to(norm_file)

    return load_pi0_policy(
        config_name=config_name,
        checkpoint_dir=str(stage_root),
        default_prompt=default_prompt,
        pytorch_device=pytorch_device,
    )


def _stage_checkpoint_dir(
    config_name: str,
    weights_path: str,
    norm_stats_path: str,
) -> pathlib.Path:
    """Stage a .pt + norm_stats.json pair into the directory layout that
    openpi's data_config expects (``<root>/<asset_id>/norm_stats.json`` plus
    ``<root>/full_weights.pt``). Returns the staged root.
    """
    weights_file = pathlib.Path(weights_path).expanduser().resolve()
    norm_file = pathlib.Path(norm_stats_path).expanduser().resolve()
    if not weights_file.is_file():
        raise FileNotFoundError(f"weights file not found: {weights_file}")
    if not norm_file.is_file():
        raise FileNotFoundError(f"norm stats file not found: {norm_file}")

    stage_root = pathlib.Path(tempfile.mkdtemp(prefix="pi0_rl_model_stage_"))
    (stage_root / "full_weights.pt").symlink_to(weights_file)
    (stage_root / "assets").mkdir()

    probe_config = get_openpi_config(config_name, model_path=str(stage_root))
    asset_id = probe_config.data.assets.asset_id or probe_config.data.repo_id
    if asset_id is None:
        raise ValueError(
            f"Config '{config_name}' has neither assets.asset_id nor repo_id set; "
            "cannot determine where to stage norm_stats.json."
        )
    (stage_root / asset_id).mkdir(parents=True, exist_ok=True)
    (stage_root / asset_id / "norm_stats.json").symlink_to(norm_file)
    return stage_root


def load_pi0_rl_model_direct(
    config_name: str,
    weights_path: str,
    norm_stats_path: str,
    default_prompt: str | None = None,
    pytorch_device: str | None = "cuda:0",
    openpi_rl_overrides: dict | None = None,
):
    """Load an ``OpenPi0ForRLActionPrediction`` model from a flat
    """
    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0Config,
        OpenPi0ForRLActionPrediction,
    )

    stage_root = _stage_checkpoint_dir(config_name, weights_path, norm_stats_path)
    train_config = get_openpi_config(config_name, model_path=str(stage_root))

    config_kwargs = dict(train_config.model.__dict__)
    if openpi_rl_overrides:
        config_kwargs.update(openpi_rl_overrides)
    actor_model_config = OpenPi0Config(**config_kwargs)
    model = OpenPi0ForRLActionPrediction(actor_model_config)

    state_dict = torch.load(
        str(stage_root / "full_weights.pt"), map_location="cpu", weights_only=False
    )
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(
            f"[policy_loader] WARNING: {len(result.missing_keys)} missing keys after load "
            f"(first 5): {list(result.missing_keys)[:5]}"
        )
    if result.unexpected_keys:
        print(
            f"[policy_loader] {len(result.unexpected_keys)} unexpected keys dropped from "
            f"checkpoint (first 5): {list(result.unexpected_keys)[:5]}"
        )
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    norm_stats = _normalize.load(pathlib.Path(stage_root) / data_config.asset_id)
    assert norm_stats is not None, "Need normalization statistics to load model."

    model.setup_wrappers(
        transforms=[
            _transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
    )

    if pytorch_device is None:
        pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(pytorch_device)
    model.eval()
    return model
