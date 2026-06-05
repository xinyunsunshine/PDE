#!/usr/bin/env python3
"""
Minimal script to keep all GPUs busy with minimal memory usage (<1GB total).
Performs lightweight matrix operations to maintain non-zero GPU utilization.
"""

import torch
import time
import threading
from typing import List
import pynvml


def gpu_worker(device_id: int, stop_event: threading.Event):
    """Worker function to keep one GPU busy with minimal operations."""
    device = torch.device(f'cuda:{device_id}')

    # Create small tensors (~10MB each)
    a = torch.randn(1000, 1000, device=device, dtype=torch.float32)
    b = torch.randn(1000, 1000, device=device, dtype=torch.float32)

    print(f"GPU {device_id}: Started with ~20MB memory usage")

    while not stop_event.is_set():
        # Check GPU utilization
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

        if util < 10:  # Only run workload if utilization below 10%
            c = torch.matmul(a, b)
            d = torch.matmul(b, a)
            a = torch.relu(c * 0.001)
            b = torch.tanh(d * 0.001)
            time.sleep(0.002)
        else:
            time.sleep(0.1)  # Check less frequently when busy

    print(f"GPU {device_id}: Stopped")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    # Initialize NVML
    pynvml.nvmlInit()

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPU(s)")

    # Calculate approximate memory usage
    memory_per_gpu = 20  # MB
    total_memory = num_gpus * memory_per_gpu
    print(f"Estimated total memory usage: ~{total_memory}MB")

    if total_memory > 1000:
        print("Warning: May exceed 1GB memory limit with many GPUs")

    stop_event = threading.Event()
    threads: List[threading.Thread] = []

    try:
        # Start worker thread for each GPU
        for gpu_id in range(num_gpus):
            thread = threading.Thread(
                target=gpu_worker,
                args=(gpu_id, stop_event),
                daemon=True
            )
            thread.start()
            threads.append(thread)

        print(f"All {num_gpus} GPUs are now busy. Press Ctrl+C to stop...")

        # Keep main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping all GPU workers...")
        stop_event.set()

        # Wait for all threads to finish
        for thread in threads:
            thread.join(timeout=2)

        print("All workers stopped.")


if __name__ == "__main__":
    main()
