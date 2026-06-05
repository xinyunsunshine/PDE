"""
Shared ZMQ protocol constants for robot servers.
"""

# Command types (6 bytes, padded with null bytes)
CMD_STEP = b"STEP"
CMD_SET_TARGET = b"TARGET"
CMD_GET_QPOS = b"QPOS"
CMD_GET_QVEL = b"QVEL"
CMD_GET_EE_POSE = b"EEPOSE"
CMD_GET_Q_D = b"QD"  # Desired joint positions (impedance only)
CMD_RESET = b"RESET"
CMD_STOP = b"STOP"
CMD_PING = b"PING"
CMD_SET_HOME_QPOS = b"HOMEQ"

RESP_OK = b"OK"
RESP_ERR = b"ERR"
RESP_PONG = b"PONG"

# Protocol constants
CMD_PADDING = 6


def pad_cmd(cmd: bytes) -> bytes:
    """Pad a command to CMD_PADDING bytes with null bytes."""
    return cmd.ljust(CMD_PADDING, b'\x00')


def parse_cmd(msg: bytes) -> tuple[bytes, bytes]:
    """Parse message into (command, payload) tuple."""
    cmd = msg[:CMD_PADDING].rstrip(b'\x00')
    payload = msg[CMD_PADDING:] if len(msg) > CMD_PADDING else b""
    return cmd, payload
