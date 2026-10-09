"""Single-GPU microwave comparison and optional VLM feedback for Jupyter."""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path


def find_microwave_task(suite):
    matches = [
        i
        for i in range(suite.get_num_tasks())
        if suite.get_task(i).language.strip().lower() == "close the microwave"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one 'close the microwave' task, found {len(matches)}"
        )
    return matches[0]


def run_episode(
    env, model, model_cfg, prompt, seed, steps, video_path, sampling_mode="eval"
):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from rlinf.envs.action_utils import prepare_actions

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    observations, _ = env.reset()
    # Reset inference noise after reset, independently of simulator random draws.
    torch.manual_seed(seed)
    first = observations["main_images"][0].cpu().numpy().astype(np.uint8)
    initial_hash = hashlib.sha256(first.tobytes()).hexdigest()
    canonical = observations["task_descriptions"][0]
    success = False
    frames = [first]
    with imageio.get_writer(str(video_path), fps=20) as writer:
        writer.append_data(first)
        for _ in range(steps // model_cfg.num_action_chunks):
            inputs = dict(observations, task_descriptions=[prompt])
            inputs.setdefault("extra_view_images", None)
            with torch.inference_mode():
                actions, _ = model.predict_action_batch(
                    env_obs=inputs, mode=sampling_mode
                )
            actions = prepare_actions(
                raw_chunk_actions=actions,
                env_type="libero",
                model_type="openpi",
                num_action_chunks=model_cfg.num_action_chunks,
                action_dim=model_cfg.action_dim,
                policy=model_cfg.get("policy_setup"),
            )
            obs_list, _, _, _, infos = env.chunk_step(actions)
            for obs in obs_list:
                frame = obs["main_images"][0].cpu().numpy().astype(np.uint8)
                writer.append_data(frame)
            observations = obs_list[-1]
            frames.append(observations["main_images"][0].cpu().numpy().astype(np.uint8))
            success |= bool(infos[-1]["episode"]["success_once"][0])
    selected = np.linspace(0, len(frames) - 1, min(8, len(frames)), dtype=int)
    np.savez_compressed(
        video_path.with_suffix(".npz"), frames=np.stack([frames[i] for i in selected])
    )
    return {
        "prompt": prompt,
        "canonical_prompt": canonical,
        "success": success,
        "seed": seed,
        "steps": steps,
        "sampling_mode": sampling_mode,
        "initial_image_sha256": initial_hash,
        "video": video_path.name,
        "frames": video_path.with_suffix(".npz").name,
    }


def compare_prompts(
    env_factory,
    model,
    model_cfg,
    reference,
    prompt,
    seed,
    steps,
    output,
    sampling_mode="eval",
):
    results = []
    for name, prompt in [("reference", reference), ("custom", prompt)]:
        # Recreate the simulator so controller state from the previous rollout
        # cannot survive a physics-state reset. Keep the same loaded policy.
        env = env_factory()
        try:
            print(f"Running {name}: {prompt}", flush=True)
            results.append(
                run_episode(
                    env,
                    model,
                    model_cfg,
                    prompt,
                    seed,
                    steps,
                    output / f"{name}.mp4",
                    sampling_mode,
                )
            )
        finally:
            env.env.close()
    if results[0]["initial_image_sha256"] != results[1]["initial_image_sha256"]:
        raise RuntimeError("Initial observations differ; comparison is not matched")
    return results


def compare(args):
    # Set before simulator/model imports. JAX performs preprocessing on CPU.
    os.environ["LIBERO_TYPE"] = "standard"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    config_dir = Path(args.output).resolve() / "libero-config"
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(config_dir))
    Path(os.environ["LIBERO_CONFIG_PATH"]).mkdir(parents=True, exist_ok=True)
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict
    from pde.configuration import configure_rlinf
    from pde.provenance import verify_rlinf

    root = configure_rlinf()
    verify_rlinf(root)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError(
            "Select an Ampere-or-newer GPU (e.g. A100, L4 or H100) for BF16 pi0.5"
        )
    if args.steps < 10 or args.steps % 10:
        raise ValueError("steps must be a positive multiple of 10")
    from pde.libero import PDELiberoEnv
    from rlinf.envs.libero.utils import get_benchmark_overridden
    from rlinf.models import get_model

    suite = get_benchmark_overridden("libero_90")()
    task_index = find_microwave_task(suite)
    if not 0 <= args.trial < len(suite.get_task_init_states(task_index)):
        raise ValueError("trial is outside this task's available initial states")
    with initialize_config_dir(
        config_dir=str(Path(__file__).parent / "configs"), version_base="1.1"
    ):
        cfg = compose(config_name="libero")
    with open_dict(cfg):
        cfg.actor.model.model_path = args.checkpoint
        cfg.actor.model.add_value_head = False
        cfg.env.train.task_suite_name = "libero_90"
        cfg.env.train.filter_task_ids = [task_index]
        cfg.env.train.specific_reset_id = args.trial
        cfg.env.train.use_fixed_reset_state_ids = True
        cfg.env.train.seed = args.seed
        cfg.env.train.group_size = 1
        cfg.env.train.is_eval = True
        cfg.env.train.ignore_terminations = True
        cfg.env.train.max_episode_steps = args.steps
        cfg.env.train.max_steps_per_rollout_epoch = args.steps
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    print("Loading one frozen pi0.5 policy...", flush=True)
    model = get_model(cfg.actor.model).eval().requires_grad_(False)
    results = compare_prompts(
        lambda: PDELiberoEnv(cfg.env.train, 1, 0, 1, None),
        model,
        cfg.actor.model,
        args.reference,
        args.prompt,
        args.seed,
        args.steps,
        output,
        getattr(args, "sampling_mode", "eval"),
    )
    report = {
        "checkpoint": args.checkpoint,
        "gpu_name": torch.cuda.get_device_name(),
        "peak_cuda_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "task_index": task_index,
        "trial": args.trial,
        "rlinf_revision": "fce5435df9472e2c61957e4f849fb903fc70827c",
        "results": results,
    }
    source = Path(args.checkpoint) / "demo_source.json"
    if source.exists():
        report["checkpoint_source"] = json.loads(source.read_text())
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    render_comparison_preview(output)
    print(json.dumps(report, indent=2))


def render_comparison_preview(directory):
    """Save an animated simulator preview that also renders in GitHub notebooks."""
    import textwrap

    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    directory = Path(directory)
    report = json.loads((directory / "comparison.json").read_text())
    results = report["results"]
    readers = [imageio.get_reader(directory / r["video"]) for r in results]
    captions = [textwrap.wrap(r["prompt"], width=36) for r in results]
    header = max(64, 14 * max(map(len, captions)) + 24)
    previews = []
    try:
        for index, frames in enumerate(zip(*readers)):
            if index % 4:
                continue
            canvas = Image.new("RGB", (256 * len(results), header + 256), "white")
            draw = ImageDraw.Draw(canvas)
            for j, (frame, result) in enumerate(zip(frames, results)):
                caption = "\n".join(captions[j])
                draw.text((256 * j + 5, 3), caption, fill="black")
                draw.text(
                    (256 * j + 5, header - 17),
                    f"Simulator success: {result['success']}",
                    fill="black",
                )
                canvas.paste(
                    Image.fromarray(frame).resize((256, 256)), (256 * j, header)
                )
            previews.append(canvas.quantize(colors=64))
    finally:
        for reader in readers:
            reader.close()
    if not previews:
        raise ValueError("Rollout videos contain no frames")
    destination = directory / "comparison.gif"
    previews[0].save(
        destination, save_all=True, append_images=previews[1:], duration=200, loop=0
    )
    return destination


def feedback(args):
    import numpy as np
    from pde.prompt_pool import PromptPool
    from pde.vlm import VLMSupervisor

    directory = Path(args.output)
    report = json.loads((directory / "comparison.json").read_text())
    result = report["results"][0]  # canonical/reference rollout
    frames = np.load(directory / result["frames"], allow_pickle=False)["frames"]
    supervisor = VLMSupervisor(
        model=args.model, provider=args.provider, base_url=args.base_url
    )
    try:
        summary = supervisor.summarize(
            result["canonical_prompt"], result["prompt"], [frames]
        )
        pool = PromptPool(
            "libero_90.close_the_microwave",
            result["canonical_prompt"],
            report["checkpoint"],
            "LIBERO-90",
            metadata={
                "canonical_evaluation": {
                    "prompt": result["prompt"],
                    "summary": summary,
                    "success_rate": float(result["success"]),
                    "rollouts": 1,
                }
            },
        )
        proposals = supervisor(pool, 3)
        response = {
            "provider": supervisor.provider,
            "model": supervisor.model,
            "summary": summary,
            "new_prompts": proposals,
            "simulator_success": result["success"],
        }
        (directory / "vlm_feedback.json").write_text(
            json.dumps(response, indent=2) + "\n"
        )
        print(json.dumps(response, indent=2))
    finally:
        supervisor.client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    rollout = commands.add_parser("compare")
    rollout.add_argument("--checkpoint", required=True)
    rollout.add_argument("--prompt", required=True)
    rollout.add_argument("--reference", default="close the microwave")
    rollout.add_argument("--seed", type=int, default=0)
    rollout.add_argument("--trial", type=int, default=0)
    rollout.add_argument("--steps", type=int, default=240)
    rollout.add_argument(
        "--sampling-mode",
        choices=["eval", "train"],
        default="eval",
        help="eval: ODE inference; train: PDE exploration noise, with weights frozen",
    )
    rollout.add_argument("--output", required=True)
    rollout.set_defaults(func=compare)
    vlm = commands.add_parser("feedback")
    vlm.add_argument("--output", required=True)
    vlm.add_argument("--provider", choices=["openai", "local_qwen"], default="openai")
    vlm.add_argument("--model", default=None)
    vlm.add_argument("--base-url", default=None)
    vlm.set_defaults(func=feedback)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
