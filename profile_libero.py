"""
LIBERO-10 Profiling Script for L40 GPU

Measures simulation speed, rendering speed, and overall throughput.
"""

import time
import numpy as np
import torch
import psutil
import os
from typing import Dict, List
from pathlib import Path

# GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_GPU_MONITORING = True
except ImportError:
    HAS_GPU_MONITORING = False

from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv


class LiberoProfiler:
    def __init__(self, num_envs: int = 1, num_tasks: int = 10, seed: int = 0):
        """Initialize profiler with environments."""
        self.num_envs = num_envs
        self.num_tasks = num_tasks
        self.seed = seed
        self.envs = []
        self.benchmark = None
        self.results: Dict[str, List[float]] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Device: {self.device}")
        print(f"Number of environments: {num_envs}")
        print(f"Number of tasks: {num_tasks}")

    def setup_environments(self):
        """Initialize LIBERO-10 environments."""
        print("\n" + "="*60)
        print("Setting up environments...")
        print("="*60)

        # Get benchmark
        self.benchmark = get_benchmark("libero_10")()

        if self.num_tasks > self.benchmark.get_num_tasks():
            self.num_tasks = self.benchmark.get_num_tasks()

        # Create environments for each task
        for task_id in range(self.num_tasks):
            task = self.benchmark.get_task(task_id)
            bddl_path = self.benchmark.get_task_bddl_file_path(task_id)

            env = OffScreenRenderEnv(
                bddl_file_name=bddl_path,
                camera_heights=256,
                camera_widths=256,
                robots=["Panda"],
                controller="OSC_POSE",
                gripper_types="default",
                has_offscreen_renderer=True,
                has_renderer=False,
            )
            env.seed(self.seed + task_id)
            self.envs.append(env)

        print(f"Created {len(self.envs)} environments")

    def get_gpu_memory(self):
        """Get current GPU memory usage."""
        if not HAS_GPU_MONITORING:
            return None
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                'used_mb': mem_info.used / 1024 / 1024,
                'total_mb': mem_info.total / 1024 / 1024,
                'percent': (mem_info.used / mem_info.total) * 100
            }
        except:
            return None

    def profile_initialization(self):
        """Profile environment initialization time."""
        print("\n" + "="*60)
        print("Profiling Initialization")
        print("="*60)

        init_times = []
        for i, env in enumerate(self.envs):
            start = time.perf_counter()
            obs = env.reset()
            elapsed = time.perf_counter() - start
            init_times.append(elapsed)
            print(f"  Task {i}: {elapsed*1000:.2f}ms")

        self.results['init_times'] = init_times
        print(f"Average init time: {np.mean(init_times)*1000:.2f}ms")
        print(f"Std dev: {np.std(init_times)*1000:.2f}ms")

    def profile_step_simulation(self, num_steps: int = 1000):
        """Profile simulation speed (step computation)."""
        print("\n" + "="*60)
        print(f"Profiling Step Simulation ({num_steps} steps)")
        print("="*60)

        step_times = []

        for env_idx, env in enumerate(self.envs):
            obs = env.reset()

            # Warmup
            for _ in range(10):
                action = np.zeros(7)  # 7D action: xyz + rpy + gripper
                obs, reward, done, info = env.step(action)

            # Profile
            start = time.perf_counter()
            for step in range(num_steps):
                action = np.zeros(7)
                obs, reward, done, info = env.step(action)

                if done:
                    obs = env.reset()

            elapsed = time.perf_counter() - start
            step_time = elapsed / num_steps * 1000  # ms per step
            step_times.append(step_time)
            fps = num_steps / elapsed
            print(f"  Task {env_idx}: {step_time:.3f}ms/step ({fps:.1f} Hz)")

        self.results['step_times'] = step_times
        print(f"\nAverage: {np.mean(step_times):.3f}ms/step")
        print(f"Average FPS: {1000/np.mean(step_times):.1f} Hz")
        print(f"Std dev: {np.std(step_times):.3f}ms/step")

    def profile_rendering(self, num_steps: int = 100):
        """Profile rendering speed (image generation)."""
        print("\n" + "="*60)
        print(f"Profiling Rendering ({num_steps} steps)")
        print("="*60)

        render_times = []

        for env_idx, env in enumerate(self.envs):
            obs = env.reset()

            # Measure time to generate images
            render_step_times = []
            for step in range(num_steps):
                start = time.perf_counter()
                action = np.zeros(7)
                obs, reward, done, info = env.step(action)
                elapsed = time.perf_counter() - start
                render_step_times.append(elapsed)

                if done:
                    obs = env.reset()

            avg_render_time = np.mean(render_step_times) * 1000  # ms
            render_times.append(avg_render_time)

            # Check image shapes
            main_img_shape = obs['agentview_image'].shape if 'agentview_image' in obs else "N/A"
            wrist_img_shape = obs.get('robot0_eye_in_hand_image', np.array([])).shape if 'robot0_eye_in_hand_image' in obs else "N/A"

            print(f"  Task {env_idx}: {avg_render_time:.3f}ms/step")
            print(f"    Main cam shape: {main_img_shape}")
            print(f"    Wrist cam shape: {wrist_img_shape}")

        self.results['render_times'] = render_times
        print(f"\nAverage render time: {np.mean(render_times):.3f}ms/step")
        print(f"Average render FPS: {1000/np.mean(render_times):.1f} Hz")

    def profile_throughput(self, num_steps: int = 500):
        """Profile overall throughput with multiple sequential steps."""
        print("\n" + "="*60)
        print(f"Profiling Throughput ({num_steps} steps)")
        print("="*60)

        throughput = []

        for env_idx, env in enumerate(self.envs):
            obs = env.reset()

            # Warmup
            for _ in range(10):
                action = np.zeros(7)
                obs, reward, done, info = env.step(action)

            # Profile
            start = time.perf_counter()
            for step in range(num_steps):
                action = np.zeros(7)
                obs, reward, done, info = env.step(action)

                if done:
                    obs = env.reset()

            elapsed = time.perf_counter() - start
            steps_per_sec = num_steps / elapsed
            throughput.append(steps_per_sec)

            print(f"  Task {env_idx}: {steps_per_sec:.1f} steps/sec")

        self.results['throughput'] = throughput
        print(f"\nAverage throughput: {np.mean(throughput):.1f} steps/sec")
        print(f"Std dev: {np.std(throughput):.1f} steps/sec")

    def profile_memory(self):
        """Profile memory usage."""
        print("\n" + "="*60)
        print("Memory Usage")
        print("="*60)

        # CPU memory
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"CPU Memory: {mem_info.rss / 1024 / 1024:.1f} MB")

        # GPU memory
        gpu_mem = self.get_gpu_memory()
        if gpu_mem:
            print(f"GPU Memory: {gpu_mem['used_mb']:.1f} MB / {gpu_mem['total_mb']:.1f} MB ({gpu_mem['percent']:.1f}%)")

    def run_full_profile(self):
        """Run all profiling benchmarks."""
        print("\n" + "="*80)
        print("LIBERO-10 PROFILING ON L40 GPU")
        print("="*80)

        self.setup_environments()
        self.profile_initialization()
        self.profile_memory()
        self.profile_step_simulation(num_steps=500)
        self.profile_rendering(num_steps=100)
        self.profile_throughput(num_steps=500)
        self.profile_memory()

        self.print_summary()

    def print_summary(self):
        """Print summary of results."""
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        print("\nInitialization:")
        print(f"  Mean: {np.mean(self.results['init_times'])*1000:.2f}ms")
        print(f"  Std:  {np.std(self.results['init_times'])*1000:.2f}ms")
        print(f"  Min:  {np.min(self.results['init_times'])*1000:.2f}ms")
        print(f"  Max:  {np.max(self.results['init_times'])*1000:.2f}ms")

        print("\nStep Time (ms/step):")
        print(f"  Mean: {np.mean(self.results['step_times']):.3f}ms")
        print(f"  Std:  {np.std(self.results['step_times']):.3f}ms")
        print(f"  Min:  {np.min(self.results['step_times']):.3f}ms")
        print(f"  Max:  {np.max(self.results['step_times']):.3f}ms")

        print("\nRendering Time (ms/step):")
        print(f"  Mean: {np.mean(self.results['render_times']):.3f}ms")
        print(f"  Std:  {np.std(self.results['render_times']):.3f}ms")
        print(f"  Min:  {np.min(self.results['render_times']):.3f}ms")
        print(f"  Max:  {np.max(self.results['render_times']):.3f}ms")

        print("\nThroughput (steps/sec):")
        print(f"  Mean: {np.mean(self.results['throughput']):.1f}")
        print(f"  Std:  {np.std(self.results['throughput']):.1f}")
        print(f"  Min:  {np.min(self.results['throughput']):.1f}")
        print(f"  Max:  {np.max(self.results['throughput']):.1f}")

        print("\n" + "="*80)

    def export_results(self, output_path: str = "libero_profile_results.txt"):
        """Export results to file."""
        with open(output_path, 'w') as f:
            f.write("LIBERO-10 PROFILING RESULTS\n")
            f.write("="*60 + "\n\n")

            f.write(f"Device: {self.device}\n")
            f.write(f"Number of Tasks: {self.num_tasks}\n")
            f.write(f"Number of Environments: {self.num_envs}\n\n")

            f.write("Initialization Times (ms):\n")
            for i, t in enumerate(self.results['init_times']):
                f.write(f"  Task {i}: {t*1000:.2f}ms\n")
            f.write(f"Mean: {np.mean(self.results['init_times'])*1000:.2f}ms\n\n")

            f.write("Step Times (ms/step):\n")
            for i, t in enumerate(self.results['step_times']):
                f.write(f"  Task {i}: {t:.3f}ms\n")
            f.write(f"Mean: {np.mean(self.results['step_times']):.3f}ms\n\n")

            f.write("Render Times (ms/step):\n")
            for i, t in enumerate(self.results['render_times']):
                f.write(f"  Task {i}: {t:.3f}ms\n")
            f.write(f"Mean: {np.mean(self.results['render_times']):.3f}ms\n\n")

            f.write("Throughput (steps/sec):\n")
            for i, t in enumerate(self.results['throughput']):
                f.write(f"  Task {i}: {t:.1f}\n")
            f.write(f"Mean: {np.mean(self.results['throughput']):.1f}\n\n")

        print(f"Results exported to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Profile LIBERO-10 on L40")
    parser.add_argument("--num-tasks", type=int, default=10, help="Number of tasks to profile")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output", type=str, default="libero_profile_results.txt", help="Output file")
    parser.add_argument("--quick", action="store_true", help="Run quick profile (fewer steps)")

    args = parser.parse_args()

    profiler = LiberoProfiler(
        num_envs=args.num_envs,
        num_tasks=args.num_tasks,
        seed=args.seed
    )

    profiler.run_full_profile()
    profiler.export_results(args.output)


if __name__ == "__main__":
    main()
