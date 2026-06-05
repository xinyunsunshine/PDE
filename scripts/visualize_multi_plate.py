"""Visualize PutOnPlateInScene25MultiPlate with a random policy and save videos."""

import argparse
import os

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils import common
from mani_skill.utils.visualization.misc import images_to_video, tile_images

# Register the custom env
import rlinf.envs.maniskill.tasks.variants  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="logs/visualize_multi_plate")
    parser.add_argument("--obj-set", type=str, default="train", choices=["train", "test", "all"])
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    env = gym.make(
        "PutOnPlateInScene25MultiPlate-v1",
        num_envs=args.num_envs,
        obs_mode="rgb+segmentation",
        control_mode=None,
        sim_backend="gpu",
        sim_config=dict(sim_freq=500, control_freq=5),
        max_episode_steps=args.max_steps,
        sensor_configs=dict(shader_pack="default"),
        render_mode="all",
        obj_set=args.obj_set,
    )

    for ep in range(args.num_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        instruction = env.unwrapped.get_language_instruction()
        print(f"Episode {ep} | instruction: {instruction[0]}")

        frames = []
        for step in range(args.max_steps):
            # Render before stepping
            img = common.to_numpy(env.render())
            if img.ndim == 3:
                img = img[None]
            # Tile parallel envs into a single image
            if img.ndim == 4 and img.shape[0] > 1:
                img = tile_images(img, nrows=int(np.sqrt(args.num_envs)))
            elif img.ndim == 4:
                img = img[0]
            frames.append(img)

            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            done = torch.logical_or(terminated, truncated)
            if done.all():
                print(f"  all envs done at step {step}")
                break

        # Save video
        video_name = f"episode_{ep}"
        images_to_video(frames, output_dir=args.output_dir, video_name=video_name, fps=args.fps)
        print(f"  saved {args.output_dir}/{video_name}.mp4 ({len(frames)} frames)")

    env.close()
    print(f"Done. Videos saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
