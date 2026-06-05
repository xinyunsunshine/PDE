#!/usr/bin/env python3
"""Minimal script to caption robot videos using hindsight relabeling."""

import base64
import re
import sys
import os
import cv2
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# LiteLLM backend configuration (for the remote client)
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")

# vLLM backend configuration (for Qwen3-VL)
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://VLM_ENDPOINT_HOST:PORT/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-235B-A22B-Thinking-FP8")
_RELABEL_VIDEO_FPS = 2

HINDSIGHT_RELABEL_PROMPT_TEMPLATE = (
    "Your task is to caption the robot action shown in this video "
    "with a hindsight instruction.\n"
    "The robot was originally asked to: \"{original_instruction}\"\n"
    "Watch the full video and write ONE instruction that best describes "
    "what the robot actually did in the entire duration of the video.\n"
    "Requirements:\n"
    "- Instruction must start with a verb.\n"
    "- Instruction must be imperative and specific.\n"
    "- Stay close to the original instruction's phrasing when possible.\n"
    "- Avoid mentioning uncertainty.\n"
    "- Keep it under 20 words.\n"
    "Output format must be exactly:\n"
    # "<thought>your reasoning</thought>\n"
    "<final>one concise instruction</final>"
)


def extract_frames(video_path: str, num_frames: int = 64) -> list[str]:
    """Extract frames from video using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    for i in range(num_frames):
        frame_idx = int((total_frames * i) / num_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = base64.standard_b64encode(buffer).decode("utf-8")
            frames.append(frame_data)

    cap.release()
    return frames


def caption_video(video_path: str, original_instruction: str, provider: str = "openai") -> str:
    """Generate hindsight instruction for a robot video."""
    print("Extracting frames...")
    frames = extract_frames(video_path)
    print(f"Extracted {len(frames)} frames")

    prompt_text = HINDSIGHT_RELABEL_PROMPT_TEMPLATE.format(
        original_instruction=original_instruction or "unknown"
    )

    if provider == "vllm":
        # vLLM backend: send the whole video as a single video_url entry.
        with open(video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")
        content = [
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
            {"type": "text", "text": prompt_text},
        ]
        client = OpenAI(api_key="EMPTY", base_url=VLLM_BASE_URL, timeout=600)
        model = VLLM_MODEL
        extra_kwargs = dict(extra_body={
            "mm_processor_kwargs": {"fps": _RELABEL_VIDEO_FPS, "do_sample_frames": True},
            "chat_template_kwargs": {"enable_thinking": True},
        })
    elif provider == "vllm_frames":
        # vLLM backend with extracted frames: send frames as image_url entries.
        content = []
        for frame_data in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_data}"}
            })
        content.append({"type": "text", "text": prompt_text})
        client = OpenAI(api_key="EMPTY", base_url=VLLM_BASE_URL, timeout=600)
        model = VLLM_MODEL
        extra_kwargs = dict(extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
        })
    else:
        # OpenAI / Anthropic: send extracted frames as image_url entries.
        content = []
        for frame_data in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_data}"}
            })
        content.append({"type": "text", "text": prompt_text})
        extra_kwargs = {}

        if provider == "anthropic":
            client = OpenAI(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY, timeout=600)
            # model = "claude-opus-4-5-20251101"
            model = "claude-opus-4-6"
            # model = "claude-sonnet-4-6"
        else:
            client = OpenAI()  # Uses OPENAI_API_KEY from .env
            model = "gpt-4o"

    print(f"Using {provider} ({model})...")

    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
        **extra_kwargs,
    )

    raw = response.choices[0].message.content
    if isinstance(raw, list):
        raw = "\n".join(
            str(item.get("text", ""))
            for item in raw
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(raw)


def parse_instruction(raw: str) -> str:
    """Extract final instruction from model output."""
    matched = re.search(r"<final>\s*(.*?)\s*</final>", raw, flags=re.DOTALL)
    instruction = matched.group(1).strip() if matched else raw.strip()
    # Truncate to 20 words if needed
    if len(instruction.split()) > 20:
        instruction = " ".join(instruction.split()[:20])
    return instruction


if __name__ == "__main__":
    """
    Example usage:
    python scripts/video_caption/video_caption.py scripts/video_caption/her_video.mp4 "put hamburger on plate" anthropic
    python scripts/video_caption/video_caption.py scripts/video_caption/her_video.mp4 "put hamburger on plate" vllm
    python scripts/video_caption/video_caption.py scripts/video_caption/her_video.mp4 "put hamburger on plate" vllm_frames
    """
    if len(sys.argv) < 3:
        print("Usage: python video_caption.py <video_file> <original_instruction> [openai|anthropic|vllm|vllm_frames]")
        sys.exit(1)

    video_file = sys.argv[1]
    original_instruction = sys.argv[2]
    provider = sys.argv[3] if len(sys.argv) > 3 else "openai"

    print(f"Captioning: {video_file}")
    print(f"Original instruction: {original_instruction}\n")

    raw_output = caption_video(video_file, original_instruction, provider)
    print("Raw output:")
    print(raw_output)
    print("\n" + "=" * 40)
    print(f"Final instruction: {parse_instruction(raw_output)}")
