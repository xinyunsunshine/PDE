"""Minimal FR3 bring-up probe — no policy, no cameras, no dataset.

Connects to the OSC server, reads qpos + ee_pose, and reports round-trip
latency. Run after real/run_osc_server.sh is up to sanity-check the ZMQ
link before launching deploy.py / collect_sft_data.py.

Usage:
    python -m real.probe_robot
"""

from __future__ import annotations

import time

import hydra
from omegaconf import DictConfig

from real.controller import FR3Controller


@hydra.main(config_path="config", config_name="deploy", version_base=None)
def main(cfg: DictConfig) -> None:
    print("[probe] Constructing FR3Controller...")
    controller = FR3Controller(cfg)

    print("[probe] start() ...")
    t0 = time.perf_counter()
    controller.start()
    print(f"[probe] start() returned in {(time.perf_counter() - t0) * 1e3:.1f} ms")

    # Let the control loop run for a bit, then touch it.
    time.sleep(0.5)

    try:
        t1 = time.perf_counter()
        qpos = controller.get_qpos()
        ee = controller.get_ee_pose()
        print(
            f"[probe] qpos = {qpos}\n"
            f"[probe] ee translation = {ee[:3, 3]}\n"
            f"[probe] round-trip (qpos + ee_pose) = {(time.perf_counter() - t1) * 1e3:.2f} ms"
        )
        print("[probe] OK — OSC server is alive and responsive.")
    except Exception as e:
        print(f"[probe] FAILED: {e!r}")
        raise
    finally:
        try:
            controller.close()
        except Exception as e:
            print(f"[probe] close() error: {e!r}")


if __name__ == "__main__":
    main()
