import base64
import io
import os
import re
import sys

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image

RITS_BASE_URL = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/qwen3-vl-235b-a22b-thinking/v1"
RITS_MODEL = "Qwen/Qwen3-VL-235B-A22B-Thinking"
# RITS_BASE_URL = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/moonshotai-kimi-k2-5/v1"
# RITS_MODEL = "moonshotai/Kimi-K2.5"
# RITS_BASE_URL = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/zai-org-glm-4-6v/v1"
# RITS_MODEL = "zai-org/GLM-4.6V"

REWORD_RITS_BASE_URL = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/qwen2-5-72b-instruct/v1"
REWROD_RITS_MODEL = "Qwen/Qwen2.5-72B-Instruct"
REWORD_RITS_BASE_URL = RITS_BASE_URL
REWROD_RITS_MODEL = RITS_MODEL

_RELABEL_VIDEO_FPS = 2

HINDSIGHT_RELABEL_PROMPT_TEMPLATE = (
    "Your task is to caption the robot action shown in this video "
    "with a hindsight instruction.\n"
    'The robot was originally asked to: "{original_instruction}"\n'
    "Watch the full video and write ONE instruction that best describes "
    "what the robot actually did in the entire duration of the video.\n"
    "Requirements:\n"
    "- Instruction must start with a verb.\n"
    "- Instruction must be imperative and specific.\n"
    "- Stay close to the original instruction's phrasing when possible.\n"
    "- Avoid mentioning uncertainty.\n"
    "- Keep it under 20 words.\n"
    "Output format must be exactly:\n"
    "<thought>your reasoning</thought>\n"
    "<final>one concise instruction</final>"
)

HINDSIGHT_REWARD_EVAL_PROMPT_TEMPLATE = (
    "Watch this video of a robot performing a task.\n"
    'The robot was asked to: "{instruction}"\n'
    "Did the robot successfully complete this instruction?\n"
    "Requirements:\n"
    "- Consider the full video from start to finish.\n"
    "- Answer based on whether the final state matches the instruction.\n"
    "Output format must be exactly:\n"
    "<thought>your reasoning</thought>\n"
    "<answer>yes</answer> or <answer>no</answer>"
)

REWORDING_PROMPT_TEMPLATE = (
    "Rephrase the following robot instruction using different words and "
    "sentence structure while preserving its meaning. Be creative with "
    "the wording — do not reuse the same phrases. Keep it under 20 words.\n\n"
    "Original instruction: {instruction}\n\n"
    "Output format must be exactly:\n"
    "your reasoning\n"
    "<final>one concise instruction</final>"
)


def _extract_text(response) -> str:
    raw = response.choices[0].message.content
    if isinstance(raw, list):
        raw = "\n".join(
            str(item.get("text", ""))
            for item in raw
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(raw).strip()


def caption_video_rits(video_path: str, original_instruction: str) -> str:
    api_key = os.environ["RITS_API_KEY"]
    client = OpenAI(
        api_key=api_key,
        base_url=RITS_BASE_URL,
        default_headers={"RITS_API_KEY": api_key},
        timeout=600,
    )

    with open(video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt_text = HINDSIGHT_RELABEL_PROMPT_TEMPLATE.format(
        original_instruction=original_instruction or "unknown"
    )

    content = [
        {
            "type": "video_url",
            "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
        },
        {"type": "text", "text": prompt_text},
    ]

    response = client.chat.completions.create(
        model=RITS_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=1024,
        extra_body={
            "mm_processor_kwargs": {
                "fps": _RELABEL_VIDEO_FPS,
                "do_sample_frames": True,
            },
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    return _extract_text(response)


def eval_reward_rits(video_path: str, instruction: str) -> str:
    api_key = os.environ["RITS_API_KEY"]
    client = OpenAI(
        api_key=api_key,
        base_url=RITS_BASE_URL,
        default_headers={"RITS_API_KEY": api_key},
        timeout=600,
    )

    with open(video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt_text = HINDSIGHT_REWARD_EVAL_PROMPT_TEMPLATE.format(instruction=instruction)

    content = [
        {
            "type": "video_url",
            "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
        },
        {"type": "text", "text": prompt_text},
    ]

    response = client.chat.completions.create(
        model=RITS_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=512,
        extra_body={
            "mm_processor_kwargs": {
                "fps": _RELABEL_VIDEO_FPS,
                "do_sample_frames": True,
            },
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    return _extract_text(response)


def _extract_frames(video_path: str, fps: float) -> list[np.ndarray]:
    """Extract frames from a video at the given FPS (RGB order)."""
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(video_fps / fps))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return frames


def _frames_to_content_items(frames: list[np.ndarray]) -> list[dict]:
    """Encode RGB numpy frames as base64 JPEG image_url content items."""
    items = []
    for frame in frames:
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        items.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            }
        )
    return items


def caption_frames_rits(
    video_path: str, original_instruction: str, fps: float = _RELABEL_VIDEO_FPS
) -> str:
    """Hindsight-caption a robot episode by sending extracted frames as images."""
    api_key = os.environ["RITS_API_KEY"]
    client = OpenAI(
        api_key=api_key,
        base_url=RITS_BASE_URL,
        default_headers={"RITS_API_KEY": api_key},
        timeout=60,
    )

    frames = _extract_frames(video_path, fps)
    prompt_text = HINDSIGHT_RELABEL_PROMPT_TEMPLATE.format(
        original_instruction=original_instruction or "unknown"
    )
    content = _frames_to_content_items(frames) + [{"type": "text", "text": prompt_text}]

    response = client.chat.completions.create(
        model=RITS_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=1024,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    return _extract_text(response)


def eval_reward_frames_rits(
    video_path: str, instruction: str, fps: float = _RELABEL_VIDEO_FPS
) -> str:
    """Evaluate task success by sending extracted frames as images instead of a video."""
    api_key = os.environ["RITS_API_KEY"]
    client = OpenAI(
        api_key=api_key,
        base_url=RITS_BASE_URL,
        default_headers={"RITS_API_KEY": api_key},
        timeout=600,
    )

    frames = _extract_frames(video_path, fps)
    prompt_text = HINDSIGHT_REWARD_EVAL_PROMPT_TEMPLATE.format(instruction=instruction)
    content = _frames_to_content_items(frames) + [{"type": "text", "text": prompt_text}]

    response = client.chat.completions.create(
        model=RITS_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=512,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    return _extract_text(response)


def parse_reward(raw: str) -> float:
    matched = re.search(r"<answer>\s*(yes|no)\s*</answer>", raw.lower(), flags=re.DOTALL)
    if matched:
        return 1.0 if matched.group(1).strip() == "yes" else 0.0
    return -1.0  # unparseable


def reword_instruction_rits(instruction: str) -> str:
    api_key = os.environ["RITS_API_KEY"]
    client = OpenAI(
        api_key=api_key,
        base_url=REWORD_RITS_BASE_URL,
        default_headers={"RITS_API_KEY": api_key},
        timeout=120,
    )

    prompt_text = REWORDING_PROMPT_TEMPLATE.format(instruction=instruction)

    response = client.chat.completions.create(
        model=REWROD_RITS_MODEL,
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=256,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    return _extract_text(response)


def parse_instruction(raw: str) -> str:
    matched = re.search(r"<final>\s*(.*?)\s*</final>", raw, flags=re.DOTALL)
    instruction = matched.group(1).strip() if matched else raw.strip()
    if len(instruction.split()) > 20:
        instruction = " ".join(instruction.split()[:20])
    return instruction


def test_mi_scoring_rits(video_path: str, correct_instruction: str) -> None:
    """Test MI reward scoring via RITS thinking model.

    Scores the correct instruction against a pool of distractors and checks
    that the correct instruction ranks #1.
    """
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    from rlinf.algorithms.rewards.mi import MIReward, infonce_rewards
    import numpy as np

    distractors = [
        "pick up the cream cheese and place it in the basket",
        "pick up the ketchup and place it in the basket",
        "pick up the tomato sauce and place it in the bowl",
        "pick up the butter and place it on the plate",
        "open the top drawer of the wooden cabinet",
        "push the plate to the front of the stove",
    ]

    api_key = os.environ["RITS_API_KEY"]
    mi = MIReward(
        vllm_url=RITS_BASE_URL,
        model=RITS_MODEL,
        backend="rits",
        api_key=api_key,
    )

    with open(video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")

    all_instructions = [correct_instruction] + distractors
    client = mi._make_client()
    scores = mi._score_rollout(client, video_b64, all_instructions)

    ranked = sorted(enumerate(zip(all_instructions, scores)), key=lambda x: -x[1][1])
    print(f"\n  {'Rank':<5} {'Score':>6}  Instruction")
    print(f"  {'-' * 56}")
    for rank, (orig_idx, (instr, sc)) in enumerate(ranked, 1):
        marker = " <-- CORRECT" if orig_idx == 0 else ""
        print(f"  {rank:<5} {sc:>6.2f}  {instr}{marker}")

    correct_rank = next(rank for rank, (idx, _) in enumerate(ranked, 1) if idx == 0)
    score_matrix = np.array([scores], dtype=np.float32)
    reward = infonce_rewards(score_matrix, intended=[0])[0]
    print(f"\n  Correct rank: {correct_rank} / {len(all_instructions)}")
    print(f"  InfoNCE reward: {reward:.4f}")
    print(f"  -> {'PASS' if correct_rank == 1 else f'FAIL (ranked #{correct_rank})'}")


if __name__ == "__main__":
    """
    Usage:
        python scripts/ibm_rits_api/test_rits_vl.py [video_file] [original_instruction]

    Defaults to libero_video.mp4 / "pick up the tomato sauce and place it in the basket".
    Runs hindsight captioning (frames), reward eval, rewording, and MI reward scoring.
    """
    video_file = sys.argv[1] if len(sys.argv) > 1 else "scripts/ibm_rits_api/libero_video.mp4"
    original_instruction = sys.argv[2] if len(sys.argv) > 2 else "pick up the tomato sauce and place it in the basket"

    print("=" * 40)
    print("TEST 1: Hindsight captioning (frames)")
    print("=" * 40)
    print(f"Video: {video_file}")
    print(f"Original instruction: {original_instruction}")
    print(f"Sending to RITS ({RITS_MODEL})...")

    raw = caption_frames_rits(video_file, original_instruction)
    print("\nRaw output:")
    print(raw)
    print(f"\nFinal instruction: {parse_instruction(raw)}")

    print("=" * 40)
    print("TEST 2: Reward eval (frames)")
    print("=" * 40)
    print(f"Video: {video_file}")
    print(f"Instruction: {original_instruction}")
    print(f"Sending to RITS ({RITS_MODEL})...")

    raw = eval_reward_frames_rits(video_file, original_instruction)
    print("\nRaw output:")
    print(raw)
    print(f"\nReward: {parse_reward(raw)}")

    print("\n" + "=" * 40)
    print("TEST 3: Rewording (text-only)")
    print("=" * 40)
    print(f"Original instruction: {original_instruction}")
    print(f"Sending to RITS ({REWROD_RITS_MODEL})...")

    raw = reword_instruction_rits(original_instruction)
    print("\nRaw output:")
    print(raw)
    print(f"\nFinal instruction: {parse_instruction(raw)}")

    print("\n" + "=" * 40)
    print("TEST 4: MI reward scoring (RITS thinking model)")
    print("=" * 40)
    print(f"Video: {video_file}")
    print(f"Correct instruction: {original_instruction}")
    print(f"Sending to RITS ({RITS_MODEL}) with thinking enabled...")
    test_mi_scoring_rits(video_file, original_instruction)
