#!/usr/bin/env python3
"""
Minimal LIBERO simulation speed profiler.
Tests different GL backends (egl, osmesa, glfw).

Usage:
    python scripts/libero_sim_profile.py [--backend egl] [--num-steps 1000] [--num-envs 4]
"""

import argparse
import os
import sys
import time
import numpy as np

# Set GL backend BEFORE importing libero
def set_gl_backend(backend):
    """Set OpenGL backend via environment variables."""
    os.environ["MUJOCO_GL"] = backend
    os.environ["PYOPENGL_PLATFORM"] = backend
    print(f"[Profile] Set MUJOCO_GL={backend}, PYOPENGL_PLATFORM={backend}")


def main():
    parser = argparse.ArgumentParser(description="Profile LIBERO simulation speed")
    parser.add_argument("--backend", type=str, default="egl",
                       choices=["egl", "osmesa", "glfw"],
                       help="OpenGL backend to use")
    parser.add_argument("--num-steps", type=int, default=1000,
                       help="Number of simulation steps to run")
    parser.add_argument("--num-envs", type=int, default=4,
                       help="Number of parallel environments")
    parser.add_argument("--camera-height", type=int, default=256,
                       help="Camera height in pixels")
    parser.add_argument("--camera-width", type=int, default=256,
                       help="Camera width in pixels")
    parser.add_argument("--task", type=int, default=0,
                       help="LIBERO task ID")

    args = parser.parse_args()

    # Set backend early
    set_gl_backend(args.backend)

    try:
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from rlinf.envs.libero.utils import get_benchmark_overridden
    except ImportError as e:
        print(f"[Error] Failed to import LIBERO: {e}")
        sys.exit(1)

    print(f"[Profile] Starting LIBERO speed profiling")
    print(f"  Backend: {args.backend}")
    print(f"  Num envs: {args.num_envs}")
    print(f"  Num steps: {args.num_steps}")
    print(f"  Camera: {args.camera_height}x{args.camera_width}")

    # Get LIBERO benchmark and task
    benchmark = get_benchmark_overridden("libero_10")()
    task = benchmark.get_task(args.task % benchmark.get_num_tasks())
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    print(f"[Profile] Task: {task.language} ({task_bddl_file})")

    # Create environments
    print(f"[Profile] Creating {args.num_envs} environments...")
    envs = []
    for i in range(args.num_envs):
        try:
            env = OffScreenRenderEnv(
                bddl_file_name=task_bddl_file,
                camera_heights=args.camera_height,
                camera_widths=args.camera_width,
            )
            env.seed(42 + i)
            envs.append(env)
        except Exception as e:
            print(f"[Error] Failed to create env {i}: {e}")
            sys.exit(1)

    print(f"[Profile] Resetting environments...")
    for env in envs:
        env.reset()

    # Warmup
    print(f"[Profile] Warmup (10 steps)...")
    rng = np.random.default_rng(42)
    for _ in range(10):
        for env in envs:
            action = rng.standard_normal(7) * 0.1
            env.step(action)

    # Profile
    print(f"[Profile] Profiling {args.num_steps} steps...")
    t_start = time.perf_counter()

    for step in range(args.num_steps):
        for env in envs:
            action = rng.standard_normal(7) * 0.1
            obs, reward, done, info = env.step(action)
            if done:
                env.reset()

        if (step + 1) % 100 == 0:
            elapsed = time.perf_counter() - t_start
            steps_per_sec = (step + 1) * args.num_envs / elapsed
            print(f"  Step {step + 1}/{args.num_steps}: {steps_per_sec:.1f} steps/sec")

    t_end = time.perf_counter()
    total_time = t_end - t_start
    total_steps = args.num_steps * args.num_envs
    steps_per_sec = total_steps / total_time

    print(f"\n[Profile] Results for backend '{args.backend}':")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Total steps: {total_steps}")
    print(f"  Steps/sec: {steps_per_sec:.1f}")
    print(f"  Avg time per step: {1000 * total_time / total_steps:.2f}ms")

    # Cleanup
    for env in envs:
        env.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
