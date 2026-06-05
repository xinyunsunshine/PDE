"""
ZMQ server for OSCController (end-effector control).

Usage:
    python -m real.server.osc_server
"""

from omegaconf import DictConfig
import asyncio
import numpy as np
import zmq
import zmq.asyncio  # Native async ZMQ support
import msgpack
import signal
from typing import Optional, Tuple

from real.server.robot import OSCController
from real.server.protocol import (
    CMD_STEP,
    CMD_GET_QPOS,
    CMD_GET_EE_POSE,
    CMD_RESET,
    CMD_STOP,
    CMD_PING,
    RESP_OK,
    RESP_ERR,
    RESP_PONG,
    pad_cmd,
    parse_cmd,
)


class OSCControlRobotServer:
    """ZMQ server wrapping OSCController for remote end-effector control."""

    def __init__(self, controller: OSCController, zmq_url: str, recv_timeout_ms: int = 1000):
        self._controller = controller
        self._zmq_url = zmq_url
        self._recv_timeout_ms = recv_timeout_ms
        self._running = False
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None

    def _setup_zmq(self):
        """Initialize ZMQ context and socket with async support."""
        self._context = zmq.asyncio.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)  # Don't block on close
        self._socket.bind(self._zmq_url)
        print(f"[OSCControlRobotServer] Bound to {self._zmq_url}")

    def _cleanup_zmq(self):
        """Clean up ZMQ resources."""
        if self._socket:
            self._socket.close()
            self._socket = None
        if self._context:
            self._context.term()
            self._context = None

    async def _handle_message(self, msg: bytes) -> bytes:
        """Process incoming message and return response."""
        try:
            cmd, payload = parse_cmd(msg)

            if cmd == CMD_PING:
                return RESP_PONG

            elif cmd == CMD_GET_QPOS:
                qpos = self._controller.qpos
                return RESP_OK + msgpack.packb(qpos.tolist())

            elif cmd == CMD_GET_EE_POSE:
                ee_pose = self._controller.ee_pose
                return RESP_OK + msgpack.packb(ee_pose.tolist())

            elif cmd == CMD_STEP:
                # Unpack 4x4 pose matrix (16 floats)
                pose_flat = msgpack.unpackb(payload)
                pose = np.array(pose_flat, dtype=np.float64).reshape(4, 4)
                await self._controller.set_ee_pose(pose)
                return RESP_OK

            elif cmd == CMD_RESET:
                await self._controller.reset()
                return RESP_OK

            elif cmd == CMD_STOP:
                self._running = False
                return RESP_OK

            else:
                return RESP_ERR + b"Unknown command"

        except Exception as e:
            return RESP_ERR + str(e).encode()

    async def run(self):
        """Run the server main loop (async)."""
        self._setup_zmq()
        self._running = True

        print("[OSCControlRobotServer] Server running. Press Ctrl+C to stop.")

        while self._running:
            try:
                # Native async poll - efficiently waits without busy polling
                # Uses timeout so we can check _running flag periodically
                if await self._socket.poll(timeout=1000):  # 100ms timeout # HACK JOHN
                    msg = await self._socket.recv()
                    response = await self._handle_message(msg)
                    await self._socket.send(response)
            except zmq.ZMQError as e:
                if self._running:
                    print(f"[OSCControlRobotServer] ZMQ error: {e}")
                break

        print("[OSCControlRobotServer] Shutting down...")
        self._cleanup_zmq()


######
# ZMQ Client for OSCController
######
class OSCRobotClient:
    """ZMQ client for remote OSC robot control (end-effector)."""

    def __init__(self, zmq_url: str, timeout_ms: int = 5000):
        self._zmq_url = zmq_url
        self._timeout_ms = timeout_ms
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._connected = False

    def connect(self) -> bool:
        """Establish connection to robot server. Returns True on success."""
        try:
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.REQ)
            self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
            self._socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.connect(self._zmq_url)

            # Test connection with ping
            if self._ping():
                self._connected = True
                print(f"[OSCRobotClient] Connected to {self._zmq_url}")
                return True
            else:
                self.disconnect()
                return False

        except zmq.ZMQError as e:
            print(f"[OSCRobotClient] Connection failed: {e}")
            self.disconnect()
            return False

    def disconnect(self):
        """Close connection to robot server."""
        self._connected = False
        if self._socket:
            self._socket.close()
            self._socket = None
        if self._context:
            self._context.term()
            self._context = None

    def _send_recv(self, msg: bytes, *, timeout_ms: Optional[int] = None) -> Tuple[bool, bytes]:
        """Send message and receive response. Returns (success, response).

        ``timeout_ms`` overrides the socket's default RCVTIMEO for a single
        request — used by ``reset()`` so the home move can take longer than
        the normal 5 s state-query budget without looking like a failure.
        """
        if not self._connected or not self._socket:
            return False, b"Not connected"
        restore_timeout = None
        if timeout_ms is not None and timeout_ms != self._timeout_ms:
            restore_timeout = self._timeout_ms
            self._socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
            self._socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        try:
            self._socket.send(msg)
            response = self._socket.recv()
            if response.startswith(RESP_ERR):
                err_msg = response[len(RESP_ERR):].decode(errors="replace")
                print(f"[OSCRobotClient] Server returned error: {err_msg}")
            return True, response
        except zmq.Again:
            print(f"[OSCRobotClient] Request timed out (timeout_ms={timeout_ms or self._timeout_ms})")
            # Reset socket state for REQ/REP pattern
            self._reconnect()
            return False, b"Timeout"
        except zmq.ZMQError as e:
            print(f"[OSCRobotClient] ZMQ error: {e}")
            return False, str(e).encode()
        finally:
            if restore_timeout is not None and self._socket is not None:
                self._socket.setsockopt(zmq.RCVTIMEO, restore_timeout)
                self._socket.setsockopt(zmq.SNDTIMEO, restore_timeout)

    def _reconnect(self):
        """Reconnect socket after timeout (required for REQ/REP pattern)."""
        if self._socket:
            self._socket.close()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self._zmq_url)

    def _ping(self) -> bool:
        """Test connection with ping."""
        try:
            self._socket.send(pad_cmd(CMD_PING))
            response = self._socket.recv()
            return response == RESP_PONG
        except:
            return False

    def step(self, pose: np.ndarray) -> bool:
        """
        Send desired end-effector pose to robot.

        Args:
            pose: 4x4 homogeneous transformation matrix

        Returns:
            True on success, False on failure
        """
        if pose.shape != (4, 4):
            raise ValueError(f"Pose must be 4x4 matrix, got {pose.shape}")

        payload = msgpack.packb(pose.flatten().tolist())
        msg = pad_cmd(CMD_STEP) + payload

        success, response = self._send_recv(msg)
        return success and response.startswith(RESP_OK)

    def get_qpos(self) -> Optional[np.ndarray]:
        """
        Get current joint positions.

        Returns:
            7D numpy array of joint positions, or None on failure
        """
        msg = pad_cmd(CMD_GET_QPOS)
        success, response = self._send_recv(msg)

        if success and response.startswith(RESP_OK):
            data = msgpack.unpackb(response[len(RESP_OK):])
            return np.array(data, dtype=np.float64)
        return None

    def get_ee_pose(self) -> Optional[np.ndarray]:
        """
        Get current end-effector pose.

        Returns:
            4x4 homogeneous transformation matrix, or None on failure
        """
        msg = pad_cmd(CMD_GET_EE_POSE)
        success, response = self._send_recv(msg)

        if success and response.startswith(RESP_OK):
            data = msgpack.unpackb(response[len(RESP_OK):])
            return np.array(data, dtype=np.float64).reshape(4, 4)
        return None

    def reset(self, *, timeout_ms: int = 30000) -> bool:
        """Reset robot to home position. Returns True on success.

        Uses a longer per-request timeout than other commands because the
        server's home move can take several seconds; falling back to the
        default 5 s RCVTIMEO would make a healthy move look like a failure.
        """
        msg = pad_cmd(CMD_RESET)
        success, response = self._send_recv(msg, timeout_ms=timeout_ms)
        return success and response.startswith(RESP_OK)

    def stop_server(self) -> bool:
        """Send stop command to server. Returns True on success."""
        msg = pad_cmd(CMD_STOP)
        success, response = self._send_recv(msg)
        return success and response.startswith(RESP_OK)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


######
# Helper Functions
######
def connect_to_osc_server(zmq_url: str, timeout_ms: int = 5000) -> OSCRobotClient:
    """
    Create and connect an OSCRobotClient.

    Args:
        zmq_url: ZMQ server URL (e.g., "tcp://localhost:5555")
        timeout_ms: Request timeout in milliseconds

    Returns:
        Connected OSCRobotClient instance

    Raises:
        ConnectionError: If connection fails
    """
    client = OSCRobotClient(zmq_url, timeout_ms)
    if not client.connect():
        raise ConnectionError(f"Failed to connect to OSC robot server at {zmq_url}")
    return client


async def run_osc_server(cfg: DictConfig):
    """Launch OSC robot server from config."""
    # Create OSC controller
    print("[OSCControlRobotServer] Initializing OSCController...")
    controller = await OSCController.create(cfg.robot)

    # Create and run server
    server = OSCControlRobotServer(
        controller=controller,
        zmq_url=cfg.zmq.url,
        recv_timeout_ms=cfg.zmq.recv_timeout_ms
    )

    # Graceful shutdown
    def signal_handler(sig, frame):
        print("\n[OSCControlRobotServer] Received shutdown signal...")
        server._running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await server.run()
    finally:
        print("[OSCControlRobotServer] Stopping controller...")
        await controller.stop()
        print("[OSCControlRobotServer] Server stopped.")


######
# Server Entry Point
######
if __name__ == "__main__":
    import hydra

    @hydra.main(version_base=None, config_path="../config", config_name="osc")
    def main(cfg: DictConfig):
        asyncio.run(run_osc_server(cfg))

    main()
