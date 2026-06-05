"""
Robot ZMQ servers and clients.

Usage:
    # Run OSC server (task-space control) — launches an in-process
    # aiofranka FrankaController on the same side as the 1 kHz libfranka
    # loop, and exposes set_ee_pose / get_ee_pose / reset over ZMQ REP.
    python -m real.server.osc_server
"""

from real.server.protocol import (
    CMD_STEP,
    CMD_GET_QPOS,
    CMD_GET_EE_POSE,
    CMD_GET_Q_D,
    CMD_RESET,
    CMD_STOP,
    CMD_PING,
    RESP_OK,
    RESP_ERR,
    RESP_PONG,
)

__all__ = [
    "CMD_STEP",
    "CMD_GET_QPOS",
    "CMD_GET_EE_POSE",
    "CMD_GET_Q_D",
    "CMD_RESET",
    "CMD_STOP",
    "CMD_PING",
    "RESP_OK",
    "RESP_ERR",
    "RESP_PONG",
]
