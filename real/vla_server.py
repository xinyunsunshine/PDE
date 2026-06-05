"""Standalone VLA inference server (RL-enabled).

Loads ``OpenPi0ForRLActionPrediction`` from an RLinf SFT checkpoint and exposes
it over a WebSocket. Compared to a vanilla pi0 server that returns just
``{"actions": ...}``, this server also returns ``forward_inputs`` (chains,
denoise_inds, observation/image, observation/state, tokenized_prompt,
tokenized_prompt_mask, optional observation/wrist_image) — all the conditioning
RLinf's PPO trainer needs to recompute current-policy logprobs

Launch — explicit weights path:
    python -m real.vla_server \\
        inference.weights=/path/to/full_weights.pt \\
        inference.norm_stats=/path/to/norm_stats.json

Launch — auto-resolve from a trainer-managed run_root:
    python -m real.vla_server \\
        inference.run_root=/home/user/real_rl_runs/<run_name> \\
        inference.sft_weights=/path/to/sft/full_weights.pt \\
        inference.norm_stats=/path/to/norm_stats.json
    # For iteration K > 0, weights resolve to
    # <run_root>/iteration{K-1:03d}/train_after_itr{K-1}_model.pt
    # (the file the trainer scp'd after PPO on iter K-1).
    # For iteration K = 0, falls back to inference.sft_weights.
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from openpi.serving import websocket_policy_server
from tqdm import tqdm

from real.iteration_paths import (
    next_iteration_to_run,
    weights_path_for_iteration,
)
from real.policy_loader import load_pi0_rl_model_direct


class _OpenPi0RLPolicyAdapter:
    """Wrap ``OpenPi0ForRLActionPrediction`` so it looks like an openpi
    ``BasePolicy`` (i.e. has ``.infer(obs) -> dict``), suitable for handing
    to ``WebsocketPolicyServer``.

    Translates between openpi-style WebSocket obs (``observation/image``,
    ``observation/wrist_image``, ``observation/state``, ``prompt``) and
    RLinf-style ``env_obs`` (``main_images``, ``wrist_images``, ``states``,
    ``task_descriptions``), and converts torch tensors in ``forward_inputs``
    back to numpy arrays so msgpack-numpy can serialize them over the wire.
    """

    def __init__(
        self,
        model,
        default_prompt: str | None = None,
        mode: str = "train",
    ) -> None:
        self._model = model
        self._default_prompt = default_prompt or ""
        # mode="train" makes sample_actions choose stochastic denoise_inds, which
        # is what the PPO trainer expects to score (see openpi_action_model.py:434).
        # Switch to "eval" if you only want deterministic single-chain inference
        # (e.g. for pure evaluation without ratio computation).
        self._mode = mode
        self._device = next(model.parameters()).device

    @property
    def metadata(self) -> dict:
        return {}

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        prompt = obs.get("prompt") or self._default_prompt
        main_image = np.asarray(obs["observation/image"], dtype=np.uint8)
        state = np.asarray(obs["observation/state"], dtype=np.float32)
        wrist_image = obs.get("observation/wrist_image")

        env_obs = {
            "main_images": torch.from_numpy(main_image).unsqueeze(0).to(self._device),
            "states": torch.from_numpy(state).unsqueeze(0).to(self._device),
            "task_descriptions": [str(prompt)],
            "wrist_images": (
                torch.from_numpy(np.asarray(wrist_image, dtype=np.uint8))
                .unsqueeze(0)
                .to(self._device)
                if wrist_image is not None
                else None
            ),
        }

        with torch.no_grad():
            actions, result = self._model.predict_action_batch(
                env_obs=env_obs,
                mode=self._mode,
                compute_values=False,
            )

        # Squeeze the leading batch dim from actions (openpi.Policy.infer
        # returns (action_horizon, action_dim) without a batch dim, so we
        # match that for compat with deploy.py and real_rl.py).
        actions_np = np.asarray(actions)
        if actions_np.ndim == 3 and actions_np.shape[0] == 1:
            actions_np = actions_np[0]

        # Convert torch tensors in forward_inputs to numpy. Leave batch dim
        # in place — real_rl.py squeezes it in RolloutBuffer.record_step.
        forward_inputs_np: dict[str, Any] = {}
        for key, val in result["forward_inputs"].items():
            if isinstance(val, torch.Tensor):
                forward_inputs_np[key] = val.detach().cpu().numpy()
            elif isinstance(val, np.ndarray):
                forward_inputs_np[key] = val
            elif isinstance(val, list):
                forward_inputs_np[key] = val
            else:
                # numpy can usually handle scalars / nested lists; if it can't,
                # msgpack-numpy will raise a clear error at send time.
                forward_inputs_np[key] = np.asarray(val)

        return {
            "actions": actions_np,
            "forward_inputs": forward_inputs_np,
        }


@hydra.main(config_path="config", config_name="vla_server", version_base=None)
def main(cfg: DictConfig) -> None:
    # Auto-resolve weights path from run_root if not explicitly given.
    # For iter K, weights live at <run_root>/iteration{K-1:03d}/train_after_itr{K-1}_model.pt;
    # for K=0, fall back to inference.sft_weights (no PPO ckpt yet).
    if cfg.inference.get("weights", None) in (None, "", "???"):
        run_root = cfg.inference.get("run_root", None)
        if run_root is None:
            raise ValueError(
                "inference.weights is unset and inference.run_root is also unset. "
                "Either pass inference.weights=/abs/path/to/full_weights.pt, OR "
                "pass inference.run_root=/abs/path/to/run_root + inference.sft_weights=/path/to/sft."
            )
        run_root = Path(str(run_root)).expanduser()
        K = next_iteration_to_run(run_root)
        sft_weights = cfg.inference.get("sft_weights", None)
        resolved = weights_path_for_iteration(run_root, K, sft_weights=sft_weights)
        print(
            f"[vla_server] Auto-resolved inference.weights = {resolved} "
            f"(iter K={K}, run_root={run_root})"
        )
        cfg.inference.weights = str(resolved)

    print(
        f"[vla_server] Loading OpenPi0ForRLActionPrediction: "
        f"config={cfg.inference.config_name} "
        f"weights={cfg.inference.weights} norm_stats={cfg.inference.norm_stats} "
        f"device={cfg.inference.device}"
    )
    openpi_rl_overrides = (
        OmegaConf.to_container(cfg.inference.openpi_rl, resolve=True)
        if "openpi_rl" in cfg.inference
        else None
    )
    if openpi_rl_overrides:
        print(f"[vla_server] OpenPi0Config overrides: {openpi_rl_overrides}")
    model = load_pi0_rl_model_direct(
        config_name=str(cfg.inference.config_name),
        weights_path=str(cfg.inference.weights),
        norm_stats_path=str(cfg.inference.norm_stats),
        default_prompt=str(cfg.inference.default_prompt),
        pytorch_device=str(cfg.inference.device),
        openpi_rl_overrides=openpi_rl_overrides,
    )
    adapter = _OpenPi0RLPolicyAdapter(
        model=model,
        default_prompt=str(cfg.inference.default_prompt),
        mode=str(cfg.inference.get("sample_mode", "train")),
    )
    print("[vla_server] Model loaded; sample_mode=", adapter._mode)

    warmup_steps = int(cfg.inference.get("warmup_steps", 5))
    if warmup_steps > 0:
        warmup_obs = {
            "observation/image": np.zeros((480, 640, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((480, 640, 3), dtype=np.uint8),
            "observation/state": np.zeros(10, dtype=np.float32),
            "prompt": str(cfg.inference.default_prompt),
        }
        for _ in tqdm(range(warmup_steps), desc="[vla_server] warmup"):
            out = adapter.infer(warmup_obs)
        # Quick sanity check: confirm forward_inputs has the keys real_rl.py needs.
        required = {
            "chains", "denoise_inds",
            "observation/image", "observation/state",
            "tokenized_prompt", "tokenized_prompt_mask",
        }
        missing = required - set(out["forward_inputs"].keys())
        if missing:
            raise RuntimeError(
                f"[vla_server] Warmup completed but forward_inputs is missing "
                f"required keys: {sorted(missing)}. Got: "
                f"{sorted(out['forward_inputs'].keys())}."
            )
        print(
            f"[vla_server] Warmup complete. forward_inputs keys: "
            f"{sorted(out['forward_inputs'].keys())}; "
            f"actions.shape={out['actions'].shape}; "
            f"chains.shape={out['forward_inputs']['chains'].shape}."
        )

    metadata = {
        "config_name": str(cfg.inference.config_name),
        "weights": str(cfg.inference.weights),
        "norm_stats": str(cfg.inference.norm_stats),
        "device": str(cfg.inference.device),
        "sample_mode": adapter._mode,
        "returns_forward_inputs": True,
    }

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=adapter,
        host=str(cfg.server.host),
        port=int(cfg.server.port),
        metadata=metadata,
    )
    print(
        f"[vla_server] Listening on ws://{cfg.server.host}:{cfg.server.port} "
        f"(healthz: http://{cfg.server.host}:{cfg.server.port}/healthz)"
    )

    def _sigint(signum, frame):
        print("\n[vla_server] Caught SIGINT, shutting down.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)

    server.serve_forever()


if __name__ == "__main__":
    main()
