#!/usr/bin/env python3
"""
Parallel LIBERO profiler using subprocess-per-env (matches real SubprocVectorEnv setup).

Tests two modes:
  --mode baseline  : all envs forced to GPU 0 (MUJOCO_EGL_DEVICE_ID=0)
  --mode spread    : envs spread across GPUs   (MUJOCO_EGL_DEVICE_ID=env_id % num_gpus)

Usage:
    python scripts/libero_sim_profile_parallel.py --mode baseline --num-envs 8 --num-steps 200
    python scripts/libero_sim_profile_parallel.py --mode spread   --num-envs 8 --num-steps 200
"""

import argparse
import multiprocessing
import os
import sys
import time

import numpy as np


def _env_worker(rank, mode, num_gpus, task_id, camera_height, camera_width,
                num_steps, result_queue):
    """Subprocess worker: creates one env and steps it num_steps times."""
    # Set EGL device before importing anything GL-related
    if mode == "baseline":
        os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    else:  # spread
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(rank % num_gpus)

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    try:
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from rlinf.envs.libero.utils import get_benchmark_overridden
    except ImportError as e:
        result_queue.put((rank, None, str(e)))
        return

    benchmark = get_benchmark_overridden("libero_10")()
    task = benchmark.get_task(task_id % benchmark.get_num_tasks())
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=camera_height,
        camera_widths=camera_width,
    )
    env.seed(42 + rank)
    env.reset()

    # Warmup
    rng = np.random.default_rng(rank)
    for _ in range(5):
        env.step(rng.standard_normal(7) * 0.1)

    # Profile
    t_start = time.perf_counter()
    for _ in range(num_steps):
        obs, reward, done, info = env.step(rng.standard_normal(7) * 0.1)
        if done:
            env.reset()
    elapsed = time.perf_counter() - t_start

    env.close()
    egl_device = os.environ.get("MUJOCO_EGL_DEVICE_ID", "?")
    result_queue.put((rank, elapsed, egl_device))


def run_profile(mode, num_envs, num_steps, num_gpus, task_id,
                camera_height, camera_width):
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()

    workers = []
    for rank in range(num_envs):
        p = ctx.Process(
            target=_env_worker,
            args=(rank, mode, num_gpus, task_id, camera_height, camera_width,
                  num_steps, result_queue),
            daemon=True,
        )
        workers.append(p)

    t_wall_start = time.perf_counter()
    for p in workers:
        p.start()
    for p in workers:
        p.join()
    t_wall_end = time.perf_counter()

    results = {}
    while not result_queue.empty():
        rank, elapsed, egl_or_err = result_queue.get()
        results[rank] = (elapsed, egl_or_err)

    wall_time = t_wall_end - t_wall_start
    total_steps = num_steps * num_envs
    aggregate_sps = total_steps / wall_time  # parallel throughput

    print(f"\n[{mode.upper()}] Results ({num_envs} envs x {num_steps} steps in parallel):")
    print(f"  Wall time:      {wall_time:.2f}s")
    print(f"  Total steps:    {total_steps}")
    print(f"  Aggregate SPS:  {aggregate_sps:.1f} steps/sec (all envs combined)")
    print(f"  Per-env SPS:    {aggregate_sps / num_envs:.1f} steps/sec/env")
    print(f"  Per-rank EGL device assignments:")
    for rank in sorted(results):
        elapsed, egl_device = results[rank]
        if elapsed is not None:
            sps = num_steps / elapsed
            print(f"    rank {rank:2d}: GPU={egl_device}  {sps:.1f} sps  ({elapsed:.2f}s)")
        else:
            print(f"    rank {rank:2d}: FAILED — {egl_device}")

    return aggregate_sps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "spread", "both"],
                        default="both")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs to spread across. "
                             "Defaults to CUDA_VISIBLE_DEVICES count.")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    args = parser.parse_args()

    if args.num_gpus is None:
        cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        args.num_gpus = len(cuda_vis.split(",")) if cuda_vis else 1

    print(f"[Profile] num_envs={args.num_envs}  num_steps={args.num_steps}  "
          f"num_gpus={args.num_gpus}  mode={args.mode}")

    if args.mode in ("baseline", "both"):
        baseline_sps = run_profile(
            "baseline", args.num_envs, args.num_steps, args.num_gpus,
            args.task, args.camera_height, args.camera_width,
        )

    if args.mode in ("spread", "both"):
        spread_sps = run_profile(
            "spread", args.num_envs, args.num_steps, args.num_gpus,
            args.task, args.camera_height, args.camera_width,
        )

    if args.mode == "both":
        print(f"\n[Summary]")
        print(f"  Baseline (all GPU 0): {baseline_sps:.1f} agg steps/sec")
        print(f"  Spread   ({args.num_gpus} GPUs):  {spread_sps:.1f} agg steps/sec")
        print(f"  Speedup: {spread_sps / baseline_sps:.2f}x")


if __name__ == "__main__":
    main()
