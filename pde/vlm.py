"""OpenAI-compatible VLM supervision for frozen-policy prompt discovery."""

import base64
import io
import json
import os

from openai import OpenAI
from PIL import Image


class VLMSupervisor:
    """Propose instructions from successful pools and positive/negative history."""

    def __init__(
        self,
        model=None,
        base_url=None,
        frames_per_video=8,
        *,
        provider="local_qwen",
        temperature=0.8,
    ):
        if provider == "openai":
            if base_url and base_url.rstrip("/") != "https://api.openai.com/v1":
                raise ValueError("Use provider=local_qwen for a custom endpoint")
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Set OPENAI_API_KEY for provider=openai")
            endpoint = "https://api.openai.com/v1"
            model = model or "gpt-4.1"
        elif provider == "local_qwen":
            endpoint = (
                base_url
                or os.environ.get("QWEN_BASE_URL")
                or "http://127.0.0.1:8000/v1"
            )
            api_key = os.environ.get("QWEN_API_KEY") or "EMPTY"
            model = model or "Qwen/Qwen3-VL-235B-A22B-Thinking-FP8"
        else:
            raise ValueError("provider must be openai or local_qwen")
        self.client = OpenAI(base_url=endpoint, api_key=api_key)
        self.provider = provider
        self.model = model
        self.frames_per_video = frames_per_video
        self.temperature = temperature

    def _request(self, content):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in robotic manipulation and prompt engineering for vision-language-action models. "
                    "Analyze behavior, identify failures, and propose concise task-relevant instructions.",
                },
                {"role": "user", "content": content},
            ],
            **(
                {"temperature": self.temperature}
                if self.temperature is not None
                else {}
            ),
        )
        text = response.choices[0].message.content
        if not text:
            raise ValueError("VLM returned an empty response")
        return text

    def summarize(self, goal, prompt, episodes):
        content = [
            {
                "type": "text",
                "text": f"Goal: {goal}\nRollout instruction: {prompt}\n"
                "Describe what the robot attempted and where it failed in one sentence. "
                "The following frames are in temporal order, grouped by episode.",
            }
        ]
        for index, frames in enumerate(episodes):
            content.append({"type": "text", "text": f"Episode {index + 1}"})
            for frame in frames:
                image = Image.fromarray(frame).convert("RGB")
                image.thumbnail((256, 256))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG")
                encoded = base64.b64encode(buffer.getvalue()).decode()
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + encoded},
                    }
                )
        return self._request(content).strip()

    def __call__(self, pool, candidates):
        history = [
            {"prompt": c.prompt, "success_rate": c.success_rate, "summary": c.summary}
            for c in pool.candidates
        ]
        history.insert(0, pool.metadata["canonical_evaluation"])
        query = {
            "goal": pool.canonical_prompt,
            "admitted_prompts": [c.prompt for c in pool.admitted],
            "history": history,
            "instruction": f"Return ONLY valid JSON with new_prompts: a list of {candidates} unique new instructions. "
            "Address observed failures; vary verbs, spatial references and specificity. "
            "Use 5-15 words per instruction. Preserve the task goal.",
        }
        text = self._request(json.dumps(query)).strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        prompts = json.loads(text)["new_prompts"]
        if not isinstance(prompts, list) or any(
            not isinstance(p, str) for p in prompts
        ):
            raise ValueError("VLM new_prompts must be a list of strings")
        return prompts
