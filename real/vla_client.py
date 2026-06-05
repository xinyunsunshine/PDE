"""Thin client wrapper for the VLA inference server.

Exposes ``connect_vla_client(host, port)`` which returns a
``WebsocketClientPolicy``. The returned object has ``.infer(obs) -> dict``
matching openpi's ``BasePolicy`` contract; ``out["actions"]`` holds the raw
action chunk.

Usage (from real/deploy.py):

    from real.vla_client import connect_vla_client

    client = connect_vla_client("localhost", 8000)
    out = client.infer({
        "observation/image": ...,
        "observation/state": ...,
        "prompt": ...,
    })
    chunk = out["actions"]  # (action_horizon, action_dim)

If the server process isn't reachable yet, the constructor blocks and
retries every 5 seconds (see WebsocketClientPolicy._wait_for_server).
"""

from __future__ import annotations

from openpi_client import websocket_client_policy


def connect_vla_client(
    host: str = "localhost",
    port: int = 8000,
    api_key: str | None = None,
) -> websocket_client_policy.WebsocketClientPolicy:
    """Open a blocking WebSocket client to the VLA inference server.

    Args:
        host: Server hostname. Use "localhost" for same-machine inference.
        port: Server port (matches ``server.port`` in real/config/vla_server.yaml).
        api_key: Optional API key; only used if the server was started with one.

    Returns:
        A WebsocketClientPolicy. Call ``.infer(obs) -> dict`` to get the action
        chunk (``out["actions"]``) plus ``out["server_timing"]`` metadata.
    """
    client = websocket_client_policy.WebsocketClientPolicy(
        host=host, port=port, api_key=api_key
    )
    metadata = client.get_server_metadata()
    print(f"[vla_client] Connected to ws://{host}:{port} — server metadata: {metadata}")
    return client
