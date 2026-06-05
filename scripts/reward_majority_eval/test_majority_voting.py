"""Minimal test for majority-voting reward evaluation.

Usage
-----
    python scripts/reward_majority_eval/test_majority_voting.py \
        --video scripts/subtraj/her_near_miss.mp4 \
        --instruction "pick up the bowl" \
        --num-samples 3

    # Compare single vs majority voting:
    python scripts/reward_majority_eval/test_majority_voting.py \
        --video scripts/subtraj/her_near_miss.mp4 \
        --instruction "pick up the bowl" \
        --num-samples 5 --compare
"""

import argparse
import os
import time

import cv2
import torch

from rlinf.workers.actor.her_vlm_client import HERVLMClient

DEFAULT_ENDPOINT = "http://VLM_ENDPOINT_HOST:PORT/v1"
DEFAULT_MODEL = "Qwen/Qwen3-VL-235B-A22B-Thinking-FP8"


def load_video_frames(video_path: str) -> list[torch.Tensor]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frames.append(torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames loaded from {video_path}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test majority-voting reward evaluation.",
    )
    parser.add_argument(
        "--video",
        default="scripts/vlm_prompt_tuning/libero_10_openvlaoft_her_sft-collect/videos/step_00002/traj_0000.mp4",
        help="Path to an .mp4 video",
    )
    parser.add_argument(
        "--instruction",
        default="pick up the book and place it in the back compartment of the caddy",
        help="Task instruction",
    )
    parser.add_argument(
        "--endpoint", default=DEFAULT_ENDPOINT, help="VLM endpoint URL"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="VLM model name")
    parser.add_argument(
        "--video-mode",
        default="frames",
        choices=["mp4", "frames", "frames_no_thinking"],
        help="How to send video to VLM (default: frames)",
    )
    parser.add_argument(
        "--prompt-version",
        type=int,
        default=6,
        choices=[1, 2, 3, 4, 6],
        help="Reward eval prompt version (default: 6)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of VLM samples for majority voting (default: 3)",
    )
    parser.add_argument(
        "--vote-temperature",
        type=float,
        default=0.6,
        help="Sampling temperature for majority voting (default: 0.6)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both single-sample and majority-voting and compare",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "none")
    client = HERVLMClient(
        endpoint=args.endpoint,
        model=args.model,
        video_mode=args.video_mode,
        api_key=api_key,
        reward_eval_prompt_version=args.prompt_version,
        log_warning_fn=lambda msg: print(f"  WARN: {msg}"),
    )

    print(f"Video:        {args.video}")
    print(f"Instruction:  {args.instruction}")
    print(f"Endpoint:     {args.endpoint}")
    print(f"Model:        {args.model}")
    print(f"Prompt:       v{args.prompt_version}")
    print(f"Num samples:  {args.num_samples}")
    print(f"Temperature:  {args.vote_temperature}")
    print()

    frames = load_video_frames(args.video)
    print(f"Loaded {len(frames)} frames\n")

    if args.compare:
        print("=== Single sample (num_samples=1) ===")
        t0 = time.time()
        reward_single, _ = client.request_reward_eval(
            frames, args.instruction, num_samples=1
        )
        dt_single = time.time() - t0
        print(f"Reward: {reward_single}  ({dt_single:.1f}s)\n")

        print(f"=== Majority voting (num_samples={args.num_samples}) ===")
        t0 = time.time()
        reward_mv, _ = client.request_reward_eval(
            frames,
            args.instruction,
            num_samples=args.num_samples,
            vote_temperature=args.vote_temperature,
        )
        dt_mv = time.time() - t0
        print(f"Reward: {reward_mv}  ({dt_mv:.1f}s)\n")

        print("=== Comparison ===")
        print(f"Single:   {reward_single}")
        print(f"Majority: {reward_mv}")
        match = "AGREE" if reward_single == reward_mv else "DISAGREE"
        print(f"Result:   {match}")
    else:
        t0 = time.time()
        reward, _ = client.request_reward_eval(
            frames,
            args.instruction,
            num_samples=args.num_samples,
            vote_temperature=args.vote_temperature,
        )
        dt = time.time() - t0
        print(f"Reward: {reward}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
