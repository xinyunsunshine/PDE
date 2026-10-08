"""Exercise both providers through the real SDK with an in-memory HTTP server."""

import json

import httpx
import numpy as np
import pytest
from openai import OpenAI

from pde.prompt_pool import PromptPool
from pde.vlm import VLMSupervisor


@pytest.mark.parametrize(
    "provider,host,key",
    [
        ("openai", "api.openai.com", "test-openai"),
        ("local_qwen", "127.0.0.1", "EMPTY"),
    ],
)
def test_provider_images_and_proposals(monkeypatch, provider, host, key):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://wrong.invalid/v1")
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_BASE_URL", raising=False)
    requests = []

    def respond(request):
        requests.append(request)
        content = (
            "Robot missed the handle."
            if len(requests) == 1
            else '{"new_prompts":["grasp the handle"]}'
        )
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
            },
        )

    monkeypatch.setattr(
        "pde.vlm.OpenAI",
        lambda **kw: OpenAI(
            **kw, http_client=httpx.Client(transport=httpx.MockTransport(respond))
        ),
    )
    supervisor = VLMSupervisor(provider=provider, temperature=None)
    try:
        summary = supervisor.summarize(
            "open door", "pull handle", [[np.zeros((8, 8, 3), dtype=np.uint8)]]
        )
        pool = PromptPool(
            "task",
            "open door",
            "checkpoint",
            "libero",
            metadata={"canonical_evaluation": {"summary": summary}},
        )
        assert supervisor(pool, 1) == ["grasp the handle"]
        for request in requests:
            assert request.url.host == host
            assert request.url.path == "/v1/chat/completions"
            assert request.headers["authorization"] == f"Bearer {key}"
            assert "temperature" not in json.loads(request.content)
        body = json.loads(requests[0].content)
        assert body["messages"][1]["content"][-1]["image_url"]["url"].startswith(
            "data:image/jpeg;base64,"
        )
    finally:
        supervisor.client.close()


def test_credentials_and_provider_validation(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        VLMSupervisor(provider="openai")
    with pytest.raises(ValueError, match="custom endpoint"):
        VLMSupervisor(provider="openai", base_url="http://localhost:8000/v1")
    with pytest.raises(ValueError, match="provider must"):
        VLMSupervisor(provider="unknown")


def test_local_server_settings(monkeypatch):
    monkeypatch.setenv("QWEN_BASE_URL", "http://localhost:9000/v1")
    monkeypatch.setenv("QWEN_API_KEY", "test-local")
    supervisor = VLMSupervisor(model="served-qwen")
    try:
        assert str(supervisor.client.base_url) == "http://localhost:9000/v1/"
        assert supervisor.client.api_key == "test-local"
        assert supervisor.model == "served-qwen"
    finally:
        supervisor.client.close()
