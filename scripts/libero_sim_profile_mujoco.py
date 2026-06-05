#!/usr/bin/env python3
"""
LIBERO profiler comparing robosuite renderer vs MuJoCo native renderer.

For a fair comparison the native renderer test disables robosuite's internal
offscreen rendering (has_offscreen_renderer=False) so physics+native render
is measured without double-rendering.

Usage:
    python scripts/libero_sim_profile_mujoco.py [--num-steps 1000] [--num-envs 4]
"""

import argparse
import os
import sys
import time
import numpy as np

# Set GL backend BEFORE importing anything else
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"


def make_envs(task_bddl_file, num_envs, camera_height, camera_width, offscreen_render):
    from libero.libero.envs import OffScreenRenderEnv

    envs = []
    for i in range(num_envs):
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=camera_height,
            camera_widths=camera_width,
            has_renderer=False,
            has_offscreen_renderer=offscreen_render,
        )
        env.seed(42 + i)
        envs.append(env)
    return envs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()

    try:
        import mujoco
        from libero.libero import get_libero_path
        from rlinf.envs.libero.utils import get_benchmark_overridden
    except ImportError as e:
        print(f"[Error] Failed to import: {e}")
        sys.exit(1)

    print(f"[Profile] MuJoCo version: {mujoco.__version__}")
    print(f"[Profile] Num envs: {args.num_envs}, Num steps: {args.num_steps}, Camera: {args.camera_height}x{args.camera_width}")

    benchmark = get_benchmark_overridden("libero_10")()
    task = benchmark.get_task(args.task % benchmark.get_num_tasks())
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    print(f"[Profile] Task: {task.language}")

    rng = np.random.default_rng(42)

    # -------------------------------------------------------------------------
    # Phase 1: Robosuite baseline (rendering happens inside env.step)
    # -------------------------------------------------------------------------
    print(f"\n[Profile] === Phase 1: ROBOSUITE rendering (baseline) ===")
    print(f"[Profile] Creating {args.num_envs} envs with offscreen rendering ON...")
    envs_rs = make_envs(task_bddl_file, args.num_envs, args.camera_height, args.camera_width, offscreen_render=True)
    for env in envs_rs:
        env.reset()

    print("[Profile] Warmup (10 steps)...")
    for _ in range(10):
        for env in envs_rs:
            obs, _, done, _ = env.step(rng.standard_normal(7) * 0.1)
            if done:
                env.reset()

    print(f"[Profile] Profiling {args.num_steps} steps...")
    t_start = time.perf_counter()
    for step in range(args.num_steps):
        for env in envs_rs:
            _, _, done, _ = env.step(rng.standard_normal(7) * 0.1)
            if done:
                env.reset()
        if (step + 1) % 100 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  Step {step+1}/{args.num_steps}: {(step+1)*args.num_envs/elapsed:.1f} steps/sec")

    robosuite_time = time.perf_counter() - t_start
    robosuite_sps = args.num_steps * args.num_envs / robosuite_time
    print(f"[Profile] Robosuite: {robosuite_sps:.1f} steps/sec  ({robosuite_time:.2f}s total)")

    for env in envs_rs:
        env.close()

    # -------------------------------------------------------------------------
    # Phase 2: MuJoCo native renderer (robosuite offscreen rendering OFF)
    # -------------------------------------------------------------------------
    print(f"\n[Profile] === Phase 2: MuJoCo native rendering (offscreen OFF) ===")
    print(f"[Profile] Creating {args.num_envs} envs with offscreen rendering OFF...")
    envs_mj = make_envs(task_bddl_file, args.num_envs, args.camera_height, args.camera_width, offscreen_render=False)
    for env in envs_mj:
        env.reset()

    # Build native renderers — one per env, two cameras each (agentview + wrist)
    renderers = []
    for env in envs_mj:
        native_model = env.sim.model._model
        r_agent = mujoco.Renderer(native_model, height=args.camera_height, width=args.camera_width)
        r_wrist = mujoco.Renderer(native_model, height=args.camera_height, width=args.camera_width)
        renderers.append((r_agent, r_wrist))

    print("[Profile] Warmup (10 steps)...")
    for _ in range(10):
        for i, env in enumerate(envs_mj):
            _, _, done, _ = env.step(rng.standard_normal(7) * 0.1)
            native_data = env.sim.data._data
            renderers[i][0].update_scene(native_data, camera="agentview")
            renderers[i][0].render()
            renderers[i][1].update_scene(native_data, camera="robot0_eye_in_hand")
            renderers[i][1].render()
            if done:
                env.reset()

    print(f"[Profile] Profiling {args.num_steps} steps...")
    t_start = time.perf_counter()
    for step in range(args.num_steps):
        for i, env in enumerate(envs_mj):
            _, _, done, _ = env.step(rng.standard_normal(7) * 0.1)
            native_data = env.sim.data._data
            renderers[i][0].update_scene(native_data, camera="agentview")
            renderers[i][0].render()
            renderers[i][1].update_scene(native_data, camera="robot0_eye_in_hand")
            renderers[i][1].render()
            if done:
                env.reset()
        if (step + 1) % 100 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  Step {step+1}/{args.num_steps}: {(step+1)*args.num_envs/elapsed:.1f} steps/sec")

    mujoco_time = time.perf_counter() - t_start
    mujoco_sps = args.num_steps * args.num_envs / mujoco_time
    print(f"[Profile] MuJoCo native: {mujoco_sps:.1f} steps/sec  ({mujoco_time:.2f}s total)")

    for env in envs_mj:
        env.close()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n[Profile] === Comparison (serial, {args.num_envs} envs, {args.camera_height}x{args.camera_width}) ===")
    print(f"  Robosuite:     {robosuite_sps:.1f} steps/sec")
    print(f"  MuJoCo native: {mujoco_sps:.1f} steps/sec")
    if mujoco_sps > robosuite_sps:
        print(f"  Speedup:       {mujoco_sps/robosuite_sps:.2f}x  (native is faster)")
    else:
        print(f"  Speedup:       {mujoco_sps/robosuite_sps:.2f}x  (robosuite is faster)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
