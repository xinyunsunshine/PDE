"""Test VLM instruction relabeling prompts on collected trajectory videos.

Reads metadata.json files from a collected-video directory, sends each video
to the configured VLM endpoint with one or more prompt variants, and writes
results to a JSON file for comparison.

Usage::

    python scripts/vlm_prompt_tuning/test_relabel_prompts.py \
        --video-dir scripts/vlm_prompt_tuning/libero_10_openvlaoft_her_sft-collect/videos \
        --backend openai \
        --endpoint http://VLM_ENDPOINT_HOST:PORT/v1 \
        --model Qwen/Qwen3-VL-235B-A22B-Instruct \
        --steps 0 1 2 \
        --max-trajs 4 \
        --prompts v1 v2 \
        --output scripts/vlm_prompt_tuning/results.json

Backends:
  rits_frames  — RITS endpoint, sends frames as separate image_url messages
  rits         — RITS endpoint, sends a single video_url (mp4 base64)
  openai       — OpenAI-compatible endpoint, sends frames as image_url
  anthropic    — Anthropic API, sends frames as image_url
"""

import argparse
import base64
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Prompt variants — add / edit freely to test different phrasings
# ---------------------------------------------------------------------------

PROMPTS: dict[str, str] = {
    "v1": (
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
        "<thought>your reasoning</thought>\n"
        "<final>one concise instruction</final>"
    ),
    "v2": (
        "You are labeling a robot manipulation trajectory with a hindsight "
        "instruction that accurately describes what the robot ACTUALLY DID, "
        "not what it was asked to do.\n\n"
        "Original instruction: \"{original_instruction}\"\n\n"
        "## Analysis steps\n"
        "1. **Identify objects**: Which objects does the gripper contact or move? "
        "Use the exact object names visible in the scene.\n"
        "2. **Track gripper trajectory**: Where does the gripper start, what does "
        "it grasp (if anything), where does it transport the object, and where "
        "does it release?\n"
        "3. **Note the end-state**: What is the final spatial relationship between "
        "the manipulated object(s) and the surrounding landmarks?\n"
        "4. **Compare to original**: Did the robot complete the original task, "
        "partially complete it, or do something entirely different?\n\n"
        "## Instruction rules\n"
        "- Start with an imperative verb.\n"
        "- Describe the FULL trajectory from start to end.\n"
        "- Reference concrete objects and spatial landmarks visible in the scene.\n"
        "- Do NOT hallucinate contact or success that is not visible.\n"
        "- Do NOT restate the original instruction unless it exactly matches "
        "the observed behavior.\n"
        "- Keep it under 20 words.\n\n"
        "Output format (strict):\n"
        "<thought>\n"
        "- Objects contacted: ...\n"
        "- Gripper path: ...\n"
        "- End-state: ...\n"
        "- Matches original? ...\n"
        "</thought>\n"
        "<final>one imperative instruction describing the full trajectory</final>"
    ),
    # ── V3: task-perturbation-optimized ──────────────────────────────────
    # Key changes vs V1/V2:
    #   - Does NOT encourage staying close to the original instruction
    #   - Explicitly tells the VLM the original might be wrong/irrelevant
    #   - Focuses on observable physical outcomes (object, action, destination)
    #   - Asks for LIBERO-style phrasing: "verb + object + spatial goal"
    "v3": (
        "Watch this video of a robot arm performing a manipulation task.\n"
        "The robot was originally instructed to: \"{original_instruction}\"\n"
        "However, the robot may have FAILED or done something DIFFERENT from "
        "the original instruction. Your job is to describe what the robot "
        "ACTUALLY DID based solely on visual evidence.\n\n"
        "Focus on three things:\n"
        "1. Which object(s) did the gripper contact or grasp?\n"
        "2. Where did the object(s) move to?\n"
        "3. What is the final spatial arrangement?\n\n"
        "Rules:\n"
        "- Write ONE imperative instruction (e.g., 'pick up the X and place "
        "it on the Y').\n"
        "- Name objects and landmarks exactly as they appear in the scene.\n"
        "- If the robot failed to grasp anything, describe its movement path "
        "(e.g., 'move the gripper above the table').\n"
        "- Do NOT copy the original instruction — describe what you SEE.\n"
        "- Under 20 words.\n\n"
        "Output:\n"
        "<thought>brief reasoning about what happened</thought>\n"
        "<final>your instruction</final>"
    ),
    # ── V4: zero-bias ablation (no original instruction provided) ────────
    # Tests whether removing the original instruction entirely produces
    # more diverse and accurate relabeling.
    "v4": (
        "Watch this video of a robot arm performing a tabletop manipulation "
        "task. Write ONE instruction that describes what the robot actually "
        "did from start to finish.\n\n"
        "Focus on:\n"
        "- Which object(s) did the gripper contact or move?\n"
        "- Where did the object(s) end up?\n"
        "- What is the final spatial arrangement?\n\n"
        "Rules:\n"
        "- Write an imperative instruction starting with a verb.\n"
        "- Name objects and landmarks exactly as they appear.\n"
        "- If the robot failed to grasp anything, describe the gripper's "
        "movement path.\n"
        "- Under 20 words.\n\n"
        "Output:\n"
        "<thought>brief reasoning</thought>\n"
        "<final>your instruction</final>"
    ),
    # ── V5: precise verbs, concise single-action format ─────────────────────
    # Like v3/v4 length but with specific physical verbs instead of "move"/"put".
    "v5": (
        "Watch this video of a robot arm manipulating objects on a table.\n"
        "The robot was originally instructed to: \"{original_instruction}\"\n"
        "The robot may have failed or done something different. Describe what "
        "ACTUALLY HAPPENED based on visual evidence only.\n\n"
        "Think about what verb best describes what happened to the object:\n"
        "- Was it lifted, slid, pushed, tilted, dropped, rotated, dragged, "
        "or placed?\n"
        "- Do NOT use generic verbs like 'move' or 'put'.\n\n"
        "Rules:\n"
        "- Write ONE short imperative instruction describing the single main "
        "action and its outcome (e.g., 'slide the cup to the edge of the "
        "table').\n"
        "- Do NOT list multiple steps or chain actions with commas.\n"
        "- Name objects and landmarks exactly as they appear.\n"
        "- Under 15 words.\n\n"
        "Output:\n"
        "<thought>what verb best describes what happened to the object</thought>\n"
        "<final>short instruction here</final>"
    ),
    # ── V8: scene-object-aware, relaxed "Nothing" but assertive on contact ───
    # Like v7 but reduces false "Nothing" by also describing clear approaches.
    # Keeps v7's assertive outcome-focused style for confirmed manipulations.
    "v8": (
        "Watch this video of a robot arm performing a manipulation task.\n"
        "The robot was originally instructed to: \"{original_instruction}\"\n"
        "However, the robot may have FAILED or done something DIFFERENT from "
        "the original instruction. Your job is to describe what the robot "
        "ACTUALLY DID based solely on visual evidence.\n\n"
        "Objects present in the scene: {scene_objects_text}\n\n"
        "Classify the robot's behavior into one of THREE categories:\n\n"
        "CATEGORY A — CONFIRMED MANIPULATION: The gripper physically contacted "
        "an object AND the object visibly moved, shifted, or changed position. "
        "Describe the OUTCOME assertively (e.g., 'place the bowl on the plate', "
        "'push the cream cheese toward the stove', 'open the drawer').\n\n"
        "CATEGORY B — CLEAR APPROACH: The gripper deliberately moved toward a "
        "specific object, got close, and either closed around it or touched it, "
        "but the object did NOT visibly move. The gripper must have gotten within "
        "touching distance of a specific object — merely waving near the general "
        "area does NOT count. Describe what happened (e.g., 'grasp the bottle "
        "without lifting it', 'touch the drawer handle').\n\n"
        "CATEGORY C — NOTHING: The arm stayed still, moved aimlessly without "
        "approaching any specific object, or only hovered in open space without "
        "getting close to anything. This includes:\n"
        "- Random movements that don't target a specific object\n"
        "- The gripper moving around but never getting close to an object\n"
        "- Small jittery motions in place\n"
        "Output exactly:\n"
        "<thought>brief reasoning</thought>\n"
        "<final>Nothing</final>\n\n"
        "IMPORTANT: If an object moved AT ALL, even slightly, that is Category A "
        "— describe the result, not the attempt. Never say 'attempt' when the "
        "object actually moved.\n\n"
        "Rules:\n"
        "- Write ONE imperative instruction.\n"
        "- Use SHORT, GENERIC names for objects: strip any numeric suffixes "
        "(_1, _2) and brand/material prefixes "
        "(e.g. 'akita_black_bowl_1' → 'bowl', 'flat_stove_1' → 'stove').\n"
        "- Do NOT copy the original instruction — describe what you SEE.\n"
        "- Under 20 words.\n\n"
        "Output:\n"
        "<thought>Category [A/B/C]. Brief reasoning about what happened.</thought>\n"
        "<final>your instruction OR Nothing</final>"
    ),
    # ── V9: object-identification grounded + A/B/C categories ───────────────
    # Inspired by traj_eval prompts: forces VLM to locate each object visually
    # before describing actions. Less Nothing than v7.
    "v9": (
        "Watch this video of a robot arm performing a manipulation task.\n"
        "The robot was originally instructed to: \"{original_instruction}\"\n"
        "However, the robot may have FAILED or done something DIFFERENT. "
        "Describe what the robot ACTUALLY DID based on visual evidence.\n\n"
        "Objects in the scene: {scene_objects_text}\n\n"
        "STEP 1 — LOCATE OBJECTS: For each object above, find it in the video. "
        "Do not confuse visually similar objects (plate vs bowl, lid vs plate, "
        "box vs tray). Note each object's position so you name them correctly.\n\n"
        "STEP 2 — CLASSIFY BEHAVIOR:\n"
        "A) CONFIRMED MANIPULATION: An object visibly moved or changed position. "
        "Describe the outcome assertively.\n"
        "B) CONTACT WITHOUT MOVEMENT: The gripper clearly approached and touched "
        "or closed around a specific object, but the object did not move. "
        "Describe what happened (e.g. 'grasp the bottle without lifting it').\n"
        "C) NOTHING: The arm stayed still, moved aimlessly without targeting "
        "any object, or only hovered in open space never getting close to "
        "anything. Small jittery motions in place count as Nothing.\n\n"
        "IMPORTANT:\n"
        "- If an object moved AT ALL, that is (A) — describe the result.\n"
        "- If the gripper touched an object but it didn't move, that is (B).\n"
        "- Only say Nothing (C) if the arm never targeted a specific object.\n"
        "- Never say 'attempt'.\n\n"
        "Rules:\n"
        "- Write ONE imperative instruction or the word Nothing.\n"
        "- Only describe movements you can visually confirm. Do NOT infer "
        "unseen effects (pouring, filling, heating). If the bowl moved toward "
        "the plate, say 'move the bowl toward the plate', not 'pour from bowl'.\n"
        "- Only mention objects from the scene list above.\n"
        "- Name objects using their LISTED NAME stripped of suffixes/prefixes "
        "(e.g. 'akita_black_bowl_1' → 'bowl', 'cream_cheese_1' → 'cream cheese', "
        "'flat_stove_1' → 'stove'). Do NOT rename by visual appearance.\n"
        "- Do NOT copy the original instruction — describe what you SEE.\n"
        "- Under 20 words.\n\n"
        "Output:\n"
        "<thought>Objects: [locate each]. Category [A/B/C]: [reasoning]</thought>\n"
        "<final>your answer</final>"
    ),
    # ── V10: v9 + spatial position verification ───────────────────────────────
    # Same as v9 but requires explicit spatial positions in object ID step,
    # and a verification step before writing the final instruction.
    "v10": (
        "Watch this video of a robot arm performing a manipulation task.\n"
        "The robot was originally instructed to: \"{original_instruction}\"\n"
        "However, the robot may have FAILED or done something DIFFERENT. "
        "Describe what the robot ACTUALLY DID based on visual evidence.\n\n"
        "Objects in the scene: {scene_objects_text}\n\n"
        "STEP 1 — LOCATE OBJECTS: For each object, note its position "
        "(left/right/center, front/back). Do not confuse similar objects "
        "(plate vs bowl, stove vs tray).\n\n"
        "STEP 2 — CLASSIFY:\n"
        "A) Object moved → describe outcome.\n"
        "B) Gripper touched object but it didn't move → describe contact.\n"
        "C) Arm never targeted any object → Nothing.\n\n"
        "STEP 3 — VERIFY DESTINATION: If an object moved, use the positions "
        "from Step 1 to confirm WHICH object it moved toward. The stove and "
        "the bowl are different objects at different positions.\n\n"
        "Rules:\n"
        "- Output ONE short imperative instruction starting with a verb "
        "(e.g. 'push the X toward the Y', 'grasp the X without moving it') "
        "or the word Nothing.\n"
        "- Only describe visible movements, not inferred effects.\n"
        "- Only mention objects from the scene list.\n"
        "- Use listed names stripped of suffixes "
        "('akita_black_bowl_1'→'bowl', 'cream_cheese_1'→'cream cheese', "
        "'flat_stove_1'→'stove'). Never rename by appearance.\n"
        "- Do NOT copy the original instruction.\n"
        "- Never say 'attempt'. Under 20 words.\n\n"
        "Output:\n"
        "<thought>[positions, classify, verify]</thought>\n"
        "<final>imperative instruction or Nothing</final>"
    ),
    # ── V11: subtask-framed relabeling (lenient) ─────────────────────────────
    "v11": (
        "Watch this video of a robot arm performing a manipulation task.\n"
        "The robot was instructed to: \"{original_instruction}\"\n"
        "The robot may have FAILED or only partially completed the task. "
        "Your job is to describe what PARTIAL PROGRESS the robot made "
        "toward the original task, framed as a subtask.\n\n"
        "Objects in the scene: {scene_objects_text}\n\n"
        "SUBTASK HIERARCHY (weakest to strongest progress):\n"
        "1. reach toward [object] — gripper moved toward the target but no contact\n"
        "2. grasp [object] — gripper closed around the object but didn't lift it\n"
        "3. pick up [object] — object lifted off the surface\n"
        "4. move [object] toward [destination] — object in transit\n"
        "5. place [object] on/in [destination] — object released at destination\n\n"
        "CONTACT CHECK — before claiming contact, verify:\n"
        "- Did the gripper fingers visibly close around or press against the object?\n"
        "- Did the object move, tilt, or shift in response?\n"
        "If NEITHER is true, the robot only reached toward the object.\n\n"
        "INSTRUCTIONS:\n"
        "- Identify which object(s) from the original task the robot interacted with.\n"
        "- Select the HIGHEST level of progress confirmed by visual evidence.\n"
        "- If the robot contacted a DIFFERENT object than the task target, "
        "still describe it as a subtask using that object "
        "(e.g., 'reach toward the bowl' if it approached the bowl instead).\n"
        "- If the robot made NO directed movement toward any object, output Nothing.\n\n"
        "Rules:\n"
        "- Output ONE imperative instruction that reads as a subtask of the "
        "original task.\n"
        "- Use SHORT, GENERIC object names: strip numeric suffixes and prefixes "
        "('akita_black_bowl_1' -> 'bowl', 'flat_stove_1' -> 'stove').\n"
        "- Under 20 words.\n\n"
        "Output:\n"
        "<thought>[identify target, check contact, determine progress level]</thought>\n"
        "<final>subtask instruction or Nothing</final>"
    ),
    # ── V11_strict: subtask-framed relabeling (strict task-relevance) ──────
    "v11_strict": (
        "Watch this video of a robot arm performing a manipulation task.\n"
        "The robot was instructed to: \"{original_instruction}\"\n"
        "The robot may have FAILED or only partially completed the task. "
        "Your job is to describe what PARTIAL PROGRESS the robot made "
        "toward the original task, framed as a subtask.\n\n"
        "Objects in the scene: {scene_objects_text}\n\n"
        "SUBTASK HIERARCHY (weakest to strongest progress):\n"
        "1. reach toward [object] — gripper moved toward the task-relevant target\n"
        "2. grasp [object] — gripper closed around the target object\n"
        "3. pick up [object] — target object lifted off the surface\n"
        "4. move [object] toward [destination] — target in transit to destination\n"
        "5. place [object] on/in [destination] — target released at destination\n\n"
        "CONTACT CHECK — before claiming contact, verify:\n"
        "- Did the gripper fingers visibly close around or press against the object?\n"
        "- Did the object move, tilt, or shift in response?\n"
        "If NEITHER is true, the robot only reached toward the object.\n\n"
        "TASK-RELEVANCE CHECK:\n"
        "The robot's behavior counts as progress ONLY if it interacted with or "
        "moved toward an object mentioned in the original instruction, AND moved "
        "in a direction consistent with the task goal.\n"
        "- If the robot contacted a DIFFERENT object not in the original task: Nothing.\n"
        "- If the robot moved AWAY from the task targets: Nothing.\n"
        "- If the robot made no directed movement toward any object: Nothing.\n\n"
        "Rules:\n"
        "- Output ONE imperative instruction that is a genuine subtask of the "
        "original task, or Nothing.\n"
        "- Use SHORT, GENERIC object names: strip numeric suffixes and prefixes "
        "('akita_black_bowl_1' -> 'bowl', 'flat_stove_1' -> 'stove').\n"
        "- Under 20 words.\n\n"
        "Output:\n"
        "<thought>[identify task targets, check relevance, determine progress]</thought>\n"
        "<final>subtask instruction or Nothing</final>"
    ),
    # ── V12: task-conditioned progress description ───────────────────────────
    "v12": (
        "Watch this video of a robot arm performing a manipulation task.\n"
        "The robot was originally instructed to: \"{original_instruction}\"\n"
        "However, the robot may have FAILED or only partially completed the "
        "task. Your job is to describe what progress the robot made toward "
        "the original task based solely on visual evidence.\n\n"
        "Objects present in the scene: {scene_objects_text}\n\n"
        "First, determine whether the robot made any progress toward the "
        "original task.\n"
        "Progress means the robot PHYSICALLY CONTACTED an object — "
        "the gripper fingers visibly touched or closed around it.\n"
        "No progress means the robot merely hovered near an object, "
        "approached without making contact, or stayed idle.\n\n"
        "CONTACT CHECK — before claiming the robot touched anything:\n"
        "- Did the gripper fingers visibly close around or press against it?\n"
        "- Did the object move, tilt, or shift in response?\n"
        "If NEITHER is true, the robot did NOT contact the object.\n\n"
        "If no progress was made (no confirmed physical contact), "
        "output exactly:\n"
        "<thought>brief reasoning</thought>\n"
        "<final>Nothing</final>\n\n"
        "If progress WAS made, describe what the robot accomplished toward "
        "the original task.\n\n"
        "Rules:\n"
        "- Write ONE imperative instruction starting with a verb.\n"
        "- Use SHORT, GENERIC object names (strip _1, _2 suffixes).\n"
        "- Do NOT copy the original instruction — describe what you SEE.\n"
        "- Under 20 words.\n\n"
        "You MUST end your response with:\n"
        "<final>your instruction or Nothing</final>"
    ),
    # ── V7: scene-object-aware with contact check (from HER utils) ─────────
    # Requires scene_objects in metadata. Uses {scene_objects_text} placeholder.
    "v7": (
        "Watch this video of a robot arm performing a manipulation task.\n"
        "The robot was originally instructed to: \"{original_instruction}\"\n"
        "However, the robot may have FAILED or done something DIFFERENT from "
        "the original instruction. Your job is to describe what the robot "
        "ACTUALLY DID based solely on visual evidence.\n\n"
        "Objects present in the scene: {scene_objects_text}\n\n"
        "First, determine whether the robot did anything INTERESTING.\n"
        "Interesting behavior means the robot PHYSICALLY CONTACTED an object — "
        "the gripper fingers visibly touched or closed around it.\n"
        "Uninteresting behavior means the robot merely hovered near an object, "
        "approached without making contact, or stayed idle.\n\n"
        "CONTACT CHECK — before claiming the robot touched anything, ask yourself:\n"
        "- Did the gripper fingers visibly close around or press against the object?\n"
        "- Did the object move, tilt, or shift in response to the gripper?\n"
        "- Did the object's shadow change?\n"
        "If NONE of these are true, the robot did NOT contact the object — "
        "even if the gripper moved close to it.\n\n"
        "If the behavior is UNINTERESTING (no confirmed physical contact), "
        "output exactly:\n"
        "<thought>brief reasoning about why nothing interesting happened</thought>\n"
        "<final>Nothing</final>\n\n"
        "If the behavior IS interesting (confirmed contact), focus on:\n"
        "1. Which object(s) did the gripper physically contact?\n"
        "2. Where did the object(s) move to?\n"
        "3. What is the final spatial arrangement?\n\n"
        "Rules:\n"
        "- Write ONE imperative instruction (e.g., 'pick up the mug and place "
        "it in the microwave').\n"
        "- Use SHORT, GENERIC names for objects: strip any numeric suffixes "
        "(_1, _2) and brand/material prefixes "
        "(e.g. 'akita_black_bowl_1' → 'bowl', 'flat_stove_1' → 'stove').\n"
        "- Do NOT copy the original instruction — describe what you SEE.\n"
        "- Under 20 words.\n\n"
        "Output:\n"
        "<thought>brief reasoning about what happened</thought>\n"
        "<final>your instruction</final>"
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _video_to_base64(video_path: str) -> str:
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _video_to_frame_base64s(video_path: str, max_frames: int = 32) -> list[str]:
    """Extract up to max_frames from an mp4, return as base64 JPEGs."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    # Select indices: first 2, last 4, evenly spaced middle.
    n_head, n_tail = 2, 4
    if total <= max_frames:
        indices = list(range(total))
    else:
        n_middle = max_frames - n_head - n_tail
        span = total - n_head - n_tail
        step = span / (n_middle + 1)
        middle = [int(n_head + step * (i + 1)) for i in range(n_middle)]
        indices = sorted(set(list(range(n_head)) + middle + list(range(total - n_tail, total))))

    b64_frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        _, buf = cv2.imencode(".jpg", frame)
        b64_frames.append(base64.b64encode(buf).decode("utf-8"))
    cap.release()
    return b64_frames


def _make_client(backend: str, api_key: str, endpoint: str):
    from openai import OpenAI
    extra = {}
    if backend in ("rits", "rits_frames"):
        extra["default_headers"] = {"RITS_API_KEY": api_key}
    return OpenAI(api_key=api_key, base_url=endpoint or None, timeout=600, **extra)


def _call_vlm(
    backend: str,
    client,
    model: str,
    video_path: str,
    prompt_text: str,
    max_tokens: int = 1024,
) -> str:
    if backend in ("vllm", "rits"):
        video_b64 = _video_to_base64(video_path)
        content = [
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
            {"type": "text", "text": prompt_text},
        ]
        extra_body = {
            "mm_processor_kwargs": {"fps": 2, "do_sample_frames": True},
            "chat_template_kwargs": {"enable_thinking": True},
        }
    elif backend == "rits_frames":
        frame_b64s = _video_to_frame_base64s(video_path)
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{fb64}"}}
            for fb64 in frame_b64s
        ]
        content.append({"type": "text", "text": prompt_text})
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    else:
        # openai / anthropic
        frame_b64s = _video_to_frame_base64s(video_path)
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{fb64}"}}
            for fb64 in frame_b64s
        ]
        content.append({"type": "text", "text": prompt_text})
        extra_body = {}

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        **({"extra_body": extra_body} if extra_body else {}),
    )
    raw = response.choices[0].message.content
    if isinstance(raw, list):
        raw = "\n".join(item.get("text", "") for item in raw if isinstance(item, dict))
    return str(raw).strip()


def _parse_final(raw: str) -> str | None:
    m = re.search(r"<final>\s*(.*?)\s*</final>", raw, flags=re.DOTALL)
    return m.group(1).strip() if m else None


def _run_one(
    *,
    backend: str,
    client,
    model: str,
    video_path: str,
    original_instruction: str,
    scene_objects_text: str,
    prompt_name: str,
    prompt_template: str,
) -> dict:
    prompt_text = prompt_template.format(
        original_instruction=original_instruction,
        scene_objects_text=scene_objects_text,
    )
    try:
        raw = _call_vlm(backend, client, model, video_path, prompt_text)
        final = _parse_final(raw)
        return {
            "prompt": prompt_name,
            "raw_response": raw,
            "parsed_instruction": final,
            "parse_ok": final is not None,
        }
    except Exception as e:
        return {
            "prompt": prompt_name,
            "raw_response": None,
            "parsed_instruction": None,
            "parse_ok": False,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video-dir", required=True, help="Root videos/ directory from collect_rollout_videos")
    parser.add_argument("--backend", default="rits_frames", choices=["rits_frames", "rits", "openai", "anthropic", "vllm"])
    parser.add_argument("--endpoint", default="http://VLM_ENDPOINT_HOST:PORT/v1", help="VLM API endpoint URL")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-235B-A22B-Thinking-FP8")
    parser.add_argument("--api-key", default="", help="API key (falls back to RITS_API_KEY / OPENAI_API_KEY env vars)")
    parser.add_argument("--steps", nargs="*", type=int, default=None, help="Step indices to process (default: all)")
    parser.add_argument("--max-trajs", type=int, default=4, help="Max trajectories per step (default: 4)")
    parser.add_argument("--prompts", nargs="+", default=list(PROMPTS.keys()), help=f"Prompt variants to test (default: all). Available: {list(PROMPTS.keys())}")
    parser.add_argument("--workers", type=int, default=8, help="Parallel VLM calls")
    parser.add_argument("--output", default="results.json", help="Output JSON path")
    args = parser.parse_args()

    # Resolve API key
    api_key = args.api_key
    if not api_key:
        if args.backend in ("rits", "rits_frames"):
            api_key = os.environ.get("RITS_API_KEY", "")
        elif args.backend == "anthropic":
            api_key = os.environ.get("LITELLM_API_KEY", "")
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    if not api_key:
        print(f"WARNING: no API key found for backend '{args.backend}'", file=sys.stderr)

    prompt_variants = {k: PROMPTS[k] for k in args.prompts if k in PROMPTS}
    if not prompt_variants:
        print(f"No valid prompt names in {args.prompts}. Available: {list(PROMPTS.keys())}", file=sys.stderr)
        sys.exit(1)

    video_dir = Path(args.video_dir)
    step_dirs = sorted(video_dir.glob("step_*"))
    if args.steps is not None:
        step_dirs = [d for d in step_dirs if int(d.name.split("_")[1]) in args.steps]
    if not step_dirs:
        print(f"No step directories found under {video_dir}", file=sys.stderr)
        sys.exit(1)

    client = _make_client(args.backend, api_key, args.endpoint)

    # Gather all (step, traj, prompt) jobs
    jobs = []
    for step_dir in step_dirs:
        meta_path = step_dir / "metadata.json"
        if not meta_path.exists():
            print(f"SKIP {step_dir}: no metadata.json", file=sys.stderr)
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        trajs = meta["trajectories"][: args.max_trajs]
        for traj in trajs:
            video_path = str(video_dir / traj["video"])
            if not os.path.exists(video_path):
                print(f"SKIP {video_path}: not found", file=sys.stderr)
                continue
            scene_objects = traj.get("scene_objects")
            scene_objects_text = ", ".join(scene_objects) if scene_objects else "not specified"
            for prompt_name, prompt_template in prompt_variants.items():
                jobs.append({
                    "step": meta["step"],
                    "traj": traj["traj"],
                    "original_instruction": traj.get("instruction", ""),
                    "scene_objects_text": scene_objects_text,
                    "total_reward": traj.get("total_reward"),
                    "video_path": video_path,
                    "prompt_name": prompt_name,
                    "prompt_template": prompt_template,
                })

    print(f"Running {len(jobs)} VLM calls ({len(step_dirs)} steps × {args.max_trajs} trajs × {len(prompt_variants)} prompts) ...", flush=True)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _run_one,
                backend=args.backend,
                client=client,
                model=args.model,
                video_path=j["video_path"],
                original_instruction=j["original_instruction"],
                scene_objects_text=j["scene_objects_text"],
                prompt_name=j["prompt_name"],
                prompt_template=j["prompt_template"],
            ): j
            for j in jobs
        }
        for i, fut in enumerate(as_completed(futures), 1):
            j = futures[fut]
            res = fut.result()
            record = {
                "step": j["step"],
                "traj": j["traj"],
                "video": j["video_path"],
                "original_instruction": j["original_instruction"],
                "scene_objects": j["scene_objects_text"],
                "total_reward": j["total_reward"],
                **res,
            }
            results.append(record)
            status = "OK" if res["parse_ok"] else "FAIL"
            print(
                f"[{i}/{len(jobs)}] step={j['step']} traj={j['traj']} prompt={j['prompt_name']} {status}: "
                f"{res.get('parsed_instruction') or res.get('error', '(no <final> tag)')}",
                flush=True,
            )

    # Sort for stable output
    results.sort(key=lambda r: (r["step"], r["traj"], r["prompt"]))

    with open(args.output, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"\nResults written to {args.output}")

    # Summary table
    print("\n=== Summary ===")
    for prompt_name in prompt_variants:
        subset = [r for r in results if r["prompt"] == prompt_name]
        n_ok = sum(1 for r in subset if r["parse_ok"])
        unchanged = sum(
            1 for r in subset
            if r["parse_ok"] and r["parsed_instruction"] and
            r["parsed_instruction"].lower() == r["original_instruction"].lower()
        )
        print(f"  {prompt_name}: {n_ok}/{len(subset)} parsed, {unchanged} unchanged from original")
    print()
    # Print all results grouped by (step, traj)
    keys = sorted({(r["step"], r["traj"]) for r in results})
    for step, traj in keys:
        subset = [r for r in results if r["step"] == step and r["traj"] == traj]
        orig = subset[0]["original_instruction"]
        reward = subset[0]["total_reward"]
        print(f"step={step} traj={traj} reward={reward}  original: {orig!r}")
        for r in sorted(subset, key=lambda x: x["prompt"]):
            print(f"  [{r['prompt']}] {r.get('parsed_instruction') or '(parse failed)'}")
        print()


if __name__ == "__main__":
    main()
