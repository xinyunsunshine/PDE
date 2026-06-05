"""Test HER reward evaluation prompts against collected trajectory videos.

Supports two modes:
1. **Batch mode** (default): reads metadata.json files from a collected-video
   directory and evaluates all trajectories with one or more prompt versions.
2. **Single-video mode**: evaluates a single video + instruction pair (legacy).

Usage
-----
Batch (default — all videos under the collected directory)::

    python scripts/traj_eval/test_reward_eval_prompt.py \
        --endpoint http://VLM_ENDPOINT_HOST:PORT/v1 \
        --model Qwen/Qwen3-VL-235B-A22B-Thinking-FP8 \
        --prompt both

Batch with filters::

    python scripts/traj_eval/test_reward_eval_prompt.py \
        --video-dir path/to/videos \
        --steps 0 1 2 \
        --max-trajs 4 \
        --prompt v2 \
        --output results.json

Single video::

    python scripts/traj_eval/test_reward_eval_prompt.py \
        --video scripts/subtraj/her_near_miss.mp4 \
        --instruction "pick up the bowl" \
        --prompt v1
"""

import argparse
import base64
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Prompts (copied from her_utils.py)
# ---------------------------------------------------------------------------

HER_REWARD_EVAL_PROMPT_TEMPLATE = (
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

HER_REWARD_EVAL_PROMPT_TEMPLATE_V2 = (
    "Watch this video of a robot arm performing a manipulation task.\n"
    'The robot was instructed to: "{instruction}"\n\n'
    "Determine whether the robot FULLY and SUCCESSFULLY completed the instruction.\n\n"
    "Evaluation criteria (all must be satisfied for YES):\n"
    "1. The robot makes physical contact with the correct object(s) named in the instruction.\n"
    "2. The robot performs the correct action (pick, place, push, open, etc.) as stated.\n"
    "3. The final state of the scene matches the goal described in the instruction "
    "(e.g. object is in the correct location, container is open/closed).\n"
    "4. The task reaches completion — the robot does NOT merely attempt the action "
    "or partially execute it.\n\n"
    "Strict rules:\n"
    "- Answer NO if the robot touches the wrong object, drops the object before completing "
    "placement, or ends the episode without the target object reaching its goal state.\n"
    "- Answer NO if the robot's gripper releases early, misses the target, or the object "
    "falls out of the destination.\n"
    "- Answer NO if the final frame does not clearly show the task goal achieved.\n"
    "- Do NOT give benefit of the doubt. If you are uncertain, answer NO.\n\n"
    "CRITICAL — final frame verification:\n"
    "Before answering, explicitly describe what you see in the LAST frame of the video. "
    "The task is only complete if the final frame shows the goal state achieved "
    "(e.g. for 'pick up X': the object must be visibly held in the gripper or elevated "
    "off the table in the last frame — NOT just touched or lifted mid-video then set down).\n\n"
    "Examples of near-misses that must be answered NO:\n"
    "- 'pick up the bowl': the robot touches or nudges the bowl but it remains on the table "
    "in the final frame.\n"
    "- 'pick up the bowl': the robot lifts the bowl mid-video but drops it before the episode ends.\n"
    "- 'place the bowl on the plate': the robot picks up the bowl but does not set it on "
    "the plate by the final frame.\n"
    "- 'open the drawer': the robot grabs the drawer handle but the drawer is still closed "
    "in the final frame.\n\n"
    "Output format must be exactly:\n"
    "<thought>1) Describe what the robot did step by step. "
    "2) Describe exactly what you see in the final frame. "
    "3) State whether each criterion is satisfied based on the final frame.</thought>\n"
    "<answer>yes</answer> or <answer>no</answer>"
)

HER_REWARD_EVAL_PROMPT_TEMPLATE_V3 = (
    "Watch this video of a robot arm performing a manipulation task.\n"
    'The robot was instructed to: "{instruction}"\n\n'
    "Determine whether the robot successfully completed the instruction.\n\n"
    "Follow these steps:\n"
    "1. FIRST FRAME: Describe the positions of all relevant objects at the start.\n"
    "2. LAST FRAME: Describe the positions of all relevant objects at the end.\n"
    "3. CHANGES: What specifically changed between the first and last frames? "
    "List each object that moved and where it moved to.\n"
    "4. JUDGMENT: Do the observed changes satisfy the instruction?\n\n"
    "Rules:\n"
    "- Only describe what you can clearly see in the frames. Do NOT infer or assume "
    "actions that are not visible — narrating a plausible sequence of events is not "
    "evidence of completion.\n"
    "- The task is complete ONLY if the last frame shows the goal state described in "
    "the instruction (objects in their target positions).\n"
    "- Answer YES if target objects have clearly moved to approximately the correct "
    "positions, even if placement is slightly imperfect.\n"
    "- Answer NO if the scene is largely unchanged from the first frame, or if objects "
    "were merely touched/nudged without meaningful displacement toward the goal.\n"
    "- Answer NO if the robot is still gripping an object that should have been placed.\n"
    "- Answer NO if objects were dropped, knocked over, or placed in the wrong location.\n\n"
    "Output format must be exactly:\n"
    "<thought>your step-by-step analysis following steps 1-4 above</thought>\n"
    "<answer>yes</answer> or <answer>no</answer>"
)

HER_REWARD_EVAL_PROMPT_TEMPLATE_V4 = (
    "Watch this video of a robot arm performing a manipulation task.\n"
    'The robot was instructed to: "{instruction}"\n\n'
    "Determine whether the robot FULLY and SUCCESSFULLY completed the instruction.\n\n"
    "Follow these steps:\n"
    "1. FIRST FRAME: Describe the positions of all relevant objects at the start.\n"
    "2. LAST FRAME: Describe the positions of all relevant objects at the end.\n"
    "3. CHANGES: What specifically changed between the first and last frames? "
    "List each object that moved and where it moved to.\n"
    "4. VERIFICATION: For each requirement in the instruction, state whether it is "
    "satisfied based on what you can clearly see in the last frame.\n"
    "5. JUDGMENT: Are ALL requirements satisfied?\n\n"
    "Rules:\n"
    "- Only describe what you can clearly see in the frames. Do NOT infer or assume "
    "actions that are not visible — narrating a plausible sequence of events is not "
    "evidence of completion.\n"
    "- The task is complete ONLY if the last frame shows the EXACT goal state: "
    "correct objects in the correct locations as specified in the instruction.\n"
    "- Moving an object to the WRONG location is a failure (e.g. wrong compartment, "
    "wrong side of the table), even if the object was moved.\n"
    "- Answer NO if the robot is still gripping an object that should have been placed.\n"
    "- Answer NO if you cannot clearly verify the target location in the last frame. "
    "Do NOT give benefit of the doubt — if uncertain, answer NO.\n\n"
    "Output format must be exactly:\n"
    "<thought>your step-by-step analysis following steps 1-5 above</thought>\n"
    "<answer>yes</answer> or <answer>no</answer>"
)

HER_REWARD_EVAL_PROMPT_TEMPLATE_V5 = (
    "Watch this video of a robot arm performing a manipulation task.\n"
    'The robot was instructed to: "{instruction}"\n\n'
    "Determine whether the instruction is satisfied in the FINAL STATE of the video.\n\n"
    "Steps:\n"
    "1. REQUIREMENTS: List each sub-goal in the instruction separately.\n"
    "2. SPATIAL GROUNDING: If the instruction uses spatial terms (front/back, "
    "left/right, specific compartments), anchor each term to a visible landmark "
    "before proceeding.\n"
    "3. FIRST FRAME: Describe the position of every object named in the requirements.\n"
    "4. LAST FRAME: Describe the position of every object named in the requirements.\n"
    "5. REQUIREMENT CHECK: For each requirement, state whether it is satisfied "
    "based on what you see in the last frame.\n"
    "6. JUDGMENT: YES only if ALL requirements are satisfied. NO otherwise.\n\n"
    "Rules:\n"
    "- Only describe what you can clearly see. Do NOT infer actions not visible "
    "in the frames.\n"
    "- Evaluate the final state against the instruction — it does not matter "
    "whether the robot moved the object there or it was already there.\n"
    "- 'In' a compartment/container means INSIDE the enclosed space — not on top, "
    "not adjacent, not leaning against it.\n"
    "- The robot arm may still be touching the object — what matters is whether "
    "the object is at the target location.\n"
    "- If you cannot clearly verify the object is at the precise target location, "
    "answer NO.\n\n"
    "Output format must be exactly:\n"
    "<thought>your analysis following steps 1-6</thought>\n"
    "<answer>yes</answer> or <answer>no</answer>"
)

HER_REWARD_EVAL_PROMPT_TEMPLATE_V6 = (
    "Watch this video of a robot arm performing a manipulation task.\n"
    'The robot was instructed to: "{instruction}"\n\n'
    "Determine whether the robot FULLY and SUCCESSFULLY completed the instruction.\n\n"
    "Follow these steps:\n"
    "1. FIRST FRAME: Describe the positions of all relevant objects at the start.\n"
    "2. LAST FRAME: Describe the positions of all relevant objects at the end.\n"
    "3. CHANGES: What specifically changed between the first and last frames? "
    "List each object that moved and where it moved to.\n"
    "4. VERIFICATION: For each requirement in the instruction, state whether it is "
    "satisfied based on what you can clearly see in the last frame.\n"
    "5. JUDGMENT: Are ALL requirements satisfied?\n\n"
    "Rules:\n"
    "- Only describe what you can clearly see in the frames. Do NOT infer or assume "
    "actions that are not visible — narrating a plausible sequence of events is not "
    "evidence of completion.\n"
    "- The task is complete ONLY if the last frame shows the EXACT goal state: "
    "correct objects in the correct locations as specified in the instruction.\n"
    "- 'In' a compartment/container means INSIDE the enclosed space — not on top, "
    "not adjacent, not leaning against it.\n"
    "- Moving an object to the WRONG location is a failure (e.g. wrong compartment, "
    "wrong side of the table), even if the object was moved.\n"
    "- The robot arm may still be touching the object — what matters is whether "
    "the object is at the target location.\n"
    "- If you cannot clearly verify the target location in the last frame, "
    "answer UNSURE.\n\n"
    "Output format must be exactly:\n"
    "<thought>your step-by-step analysis following steps 1-5 above</thought>\n"
    "<answer>yes</answer> or <answer>no</answer> or <answer>unsure</answer>\n\n"
    "Use YES if the goal state is clearly achieved. "
    "Use NO if it clearly is not achieved. "
    "Use UNSURE if the evidence is ambiguous (e.g. you cannot tell which "
    "compartment an object is in, or whether it is on top vs inside)."
)

HER_REWARD_EVAL_PROMPT_TEMPLATE_V7 = (
    "Watch this video of a robot arm performing a manipulation task.\n"
    'The robot was instructed to: "{instruction}"\n\n'
    "Determine whether the robot FULLY and SUCCESSFULLY completed the "
    "instruction.\n\n"
    "Follow these steps:\n"
    "1. OBJECT IDENTIFICATION: For each object named in the instruction, "
    "identify EXACTLY which object in the video it refers to. Each noun "
    "in the instruction corresponds to one specific object in the scene — "
    "do not confuse visually similar objects (e.g. a plate vs a bowl, "
    "a lid vs a flat plate, a tray vs a box). Describe the distinguishing "
    "features (shape, size, color, position) that let you identify each one.\n"
    "2. FIRST FRAME: Describe the positions of all identified objects at "
    "the start.\n"
    "3. LAST FRAME: Describe the positions of all identified objects at "
    "the end.\n"
    "4. CHANGES: What specifically changed between the first and last "
    "frames? List each object that moved and where it moved to.\n"
    "5. VERIFICATION: For each requirement in the instruction, state "
    "whether it is satisfied based on what you can clearly see in the "
    "last frame. Make sure you are evaluating the CORRECT object — refer "
    "back to your identification in step 1.\n"
    "6. JUDGMENT: Are ALL requirements satisfied?\n\n"
    "Rules:\n"
    "- Only describe what you can clearly see in the frames. Do NOT infer "
    "or assume actions that are not visible — narrating a plausible "
    "sequence of events is not evidence of completion.\n"
    "- The task is complete ONLY if the last frame shows the EXACT goal "
    "state: correct objects in the correct locations as specified in the "
    "instruction.\n"
    "- 'In' a compartment/container means INSIDE the enclosed space — not "
    "on top, not adjacent, not leaning against it.\n"
    "- Moving an object to the WRONG location is a failure (e.g. wrong "
    "compartment, wrong side of the table), even if the object was moved.\n"
    "- The robot arm may still be touching the object — what matters is "
    "whether the object is at the target location.\n"
    "- If you cannot clearly verify the target location in the last frame, "
    "answer UNSURE.\n\n"
    "Output format must be exactly:\n"
    "<thought>your step-by-step analysis following steps 1-6 above</thought>\n"
    "<answer>yes</answer> or <answer>no</answer> or <answer>unsure</answer>\n\n"
    "Use YES if the goal state is clearly achieved. "
    "Use NO if it clearly is not achieved. "
    "Use UNSURE if the evidence is ambiguous (e.g. you cannot tell which "
    "compartment an object is in, or whether it is on top vs inside)."
)

HER_REWARD_EVAL_PROMPT_TEMPLATE_V8 = (
    "Watch this video of a robot arm performing a manipulation task.\n"
    'The robot was instructed to: "{instruction}"\n\n'
    "Determine whether the robot FULLY and SUCCESSFULLY completed the "
    "instruction.\n\n"
    "Follow these steps:\n"
    "1. OBJECT IDENTIFICATION: For each object named in the instruction, "
    "identify EXACTLY which object in the video it refers to. Each noun "
    "in the instruction corresponds to one specific object in the scene — "
    "do not confuse visually similar objects (e.g. a plate vs a bowl, "
    "a lid vs a flat plate, a tray vs a box, a knob vs a burner). "
    "Describe the distinguishing features (shape, size, color, position) "
    "that let you identify each one.\n"
    "2. FIRST FRAME: Describe the exact position of each identified "
    "object. Use concrete spatial references (e.g. 'on the table 5cm "
    "left of the stove', 'inside the drawer').\n"
    "3. LAST FRAME: Describe the exact position of each identified "
    "object using the same spatial references.\n"
    "4. POSITION COMPARISON: For the object that should have moved, "
    "state its position in the first frame and its position in the last "
    "frame. Did it actually change location, or is it in approximately "
    "the same place?\n"
    "5. CONTACT/PLACEMENT CHECK: For the goal to be achieved, the moved "
    "object must be physically resting on/in the target IN THE LAST "
    "FRAME. Check:\n"
    "   - Is the object visibly supported by the target surface/container "
    "(not held by the gripper mid-air, not falling, not merely nearby)?\n"
    "   - Is the object centered on the target (not hanging off the edge, "
    "not beside it, not just touching the side)?\n"
    "   - For 'in' tasks: is the object BELOW the rim of the container?\n"
    "   - For 'on' tasks: is the object ABOVE and RESTING ON the surface?\n"
    "   - For state-change tasks (open/close/turn on/turn off): is there "
    "clear visual evidence of the state change in the last frame "
    "(e.g. visible gap for open drawer, color change for stove)?\n"
    "6. JUDGMENT: Based ONLY on step 5, is the goal achieved?\n\n"
    "CRITICAL RULES — read these before answering:\n"
    "- The robot MOVING TOWARD the target is NOT evidence of success. "
    "Only the FINAL RESTING STATE matters.\n"
    "- If the object was lifted/moved but you cannot see it clearly AT "
    "the target location in the last frame, answer NO.\n"
    "- Do NOT say 'the robot moved it there so it must be there' — that "
    "is inference, not observation. Many attempts FAIL: the robot drops "
    "the object, misses the target, or doesn't move it far enough.\n"
    "- If the object is still in the gripper and the gripper is above "
    "the target, that is NOT completion — the object must be released "
    "and resting.\n"
    "- 'In' a compartment/container means the object is INSIDE the "
    "enclosed space and below the rim — not on top, not adjacent, not "
    "leaning against it.\n"
    "- 'On' a surface means the object is physically resting on that "
    "surface — not hovering above it, not beside it.\n"
    "- For 'push X to the front of Y': the object must be at the FRONT "
    "EDGE of Y in the last frame, not just moved slightly forward.\n"
    "- When in doubt, answer UNSURE rather than YES. It is better to "
    "be uncertain than to claim success when the evidence is unclear. "
    "Only answer YES when success is unambiguous.\n\n"
    "Output format must be exactly:\n"
    "<thought>your step-by-step analysis following steps 1-6 above"
    "</thought>\n"
    "<answer>yes</answer> or <answer>no</answer> or "
    "<answer>unsure</answer>\n\n"
    "Use YES only if the object is clearly, unambiguously at the target "
    "location in the final frame. "
    "Use NO if the object is clearly NOT at the target. "
    "Use UNSURE if you cannot determine the final position with "
    "confidence."
)

PROMPTS = {
    "v1": HER_REWARD_EVAL_PROMPT_TEMPLATE,
    "v2": HER_REWARD_EVAL_PROMPT_TEMPLATE_V2,
    "v3": HER_REWARD_EVAL_PROMPT_TEMPLATE_V3,
    "v4": HER_REWARD_EVAL_PROMPT_TEMPLATE_V4,
    "v5": HER_REWARD_EVAL_PROMPT_TEMPLATE_V5,
    "v6": HER_REWARD_EVAL_PROMPT_TEMPLATE_V6,
    "v7": HER_REWARD_EVAL_PROMPT_TEMPLATE_V7,
    "v8": HER_REWARD_EVAL_PROMPT_TEMPLATE_V8,
}

_VIDEO_FPS = 2
_DEFAULT_VIDEO_DIR = (
    "scripts/vlm_prompt_tuning/libero_10_openvlaoft_her_sft-collect/videos"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _video_to_base64(video_path: str, hflip: bool = False) -> str:
    if not hflip:
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    import tempfile

    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        writer = cv2.VideoWriter(
            tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(cv2.flip(frame, 1))
        cap.release()
        writer.release()
        with open(tmp.name, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


def _call_vlm(
    client: OpenAI,
    model: str,
    video_b64: str,
    prompt_text: str,
    temperature: float = 0.0,
) -> str:
    content = [
        {
            "type": "video_url",
            "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
        },
        {"type": "text", "text": prompt_text},
    ]
    extra_body = {
        "mm_processor_kwargs": {
            "fps": _VIDEO_FPS,
            "do_sample_frames": True,
        },
        "chat_template_kwargs": {"enable_thinking": True},
    }
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=16384,
        temperature=temperature,
        extra_body=extra_body,
    )
    raw = response.choices[0].message.content
    if isinstance(raw, list):
        raw = "\n".join(
            str(item.get("text", ""))
            for item in raw
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(raw).strip()


def _parse_answer(raw: str) -> str:
    matched = re.search(
        r"<answer>\s*(yes|no|unsure)\s*</answer>", raw.lower(), flags=re.DOTALL
    )
    if matched:
        return matched.group(1).strip()
    return "PARSE_FAILED"


def _parse_thought(raw: str) -> str:
    matched = re.search(r"<thought>(.*?)</thought>", raw, flags=re.DOTALL)
    if matched:
        return matched.group(1).strip()
    return ""


_VOTE_SCORES = {"yes": 1.0, "unsure": 0.5, "no": 0.0}


def _majority_vote(votes: list[str], threshold: float = 0.7) -> str:
    valid = [v for v in votes if v in _VOTE_SCORES]
    if not valid:
        return "PARSE_FAILED"
    avg = sum(_VOTE_SCORES[v] for v in valid) / len(valid)
    if avg >= threshold:
        return "yes"
    if avg <= 1.0 - threshold:
        return "no"
    return "unsure"


# ---------------------------------------------------------------------------
# Single-video mode (legacy)
# ---------------------------------------------------------------------------


def _run_single(args: argparse.Namespace) -> None:
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Video:       {video_path}")
    print(f"Instruction: {args.instruction}")
    print(f"Endpoint:    {args.endpoint}")
    print(f"Model:       {args.model}")
    print(f"Prompt(s):   {args.prompt}")

    client = OpenAI(api_key="none", base_url=args.endpoint, timeout=600)
    video_b64 = _video_to_base64(str(video_path), hflip=args.hflip)

    versions = list(PROMPTS.keys()) if args.prompt == "all" else [args.prompt]
    for v in versions:
        prompt_text = PROMPTS[v].format(instruction=args.instruction)
        print(f"\n{'=' * 60}")
        print(f"Prompt {v.upper()}")
        print(f"{'=' * 60}")
        raw = _call_vlm(client, args.model, video_b64, prompt_text)
        answer = _parse_answer(raw)
        print(raw)
        print(f"\n>>> Answer: {answer.upper()}")


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------


def _run_single_vote(
    *,
    client: OpenAI,
    model: str,
    video_b64: str,
    prompt_text: str,
    temperature: float,
) -> dict:
    max_retries = 5
    last_raw = None
    for attempt in range(max_retries):
        try:
            raw = _call_vlm(
                client, model, video_b64, prompt_text, temperature=temperature
            )
            last_raw = raw
            answer = _parse_answer(raw)
            if answer != "PARSE_FAILED":
                return {
                    "raw_response": raw,
                    "thought": _parse_thought(raw),
                    "answer": answer,
                    "parse_ok": True,
                    "attempts": attempt + 1,
                }
        except Exception as e:
            last_raw = str(e)
    return {
        "raw_response": last_raw,
        "thought": _parse_thought(last_raw or ""),
        "answer": "PARSE_FAILED",
        "parse_ok": False,
        "attempts": max_retries,
    }


def _run_one_batch(
    *,
    client: OpenAI,
    model: str,
    video_path: str,
    instruction: str,
    prompt_name: str,
    prompt_template: str,
    hflip: bool = False,
    num_votes: int = 1,
    vote_temperature: float = 0.6,
    vote_threshold: float = 0.7,
) -> dict:
    prompt_text = prompt_template.format(instruction=instruction)
    video_b64 = _video_to_base64(video_path, hflip=hflip)
    temperature = 0.0 if num_votes == 1 else vote_temperature

    if num_votes == 1:
        vote = _run_single_vote(
            client=client,
            model=model,
            video_b64=video_b64,
            prompt_text=prompt_text,
            temperature=temperature,
        )
        return {"prompt": prompt_name, **vote}

    with ThreadPoolExecutor(max_workers=num_votes) as pool:
        futures = [
            pool.submit(
                _run_single_vote,
                client=client,
                model=model,
                video_b64=video_b64,
                prompt_text=prompt_text,
                temperature=temperature,
            )
            for _ in range(num_votes)
        ]
        votes = [f.result() for f in futures]

    individual_answers = [v["answer"] for v in votes]
    final_answer = _majority_vote(individual_answers, threshold=vote_threshold)
    all_parsed = all(v["parse_ok"] for v in votes)
    return {
        "prompt": prompt_name,
        "raw_response": votes[0]["raw_response"],
        "thought": votes[0]["thought"],
        "answer": final_answer,
        "parse_ok": final_answer != "PARSE_FAILED",
        "attempts": sum(v["attempts"] for v in votes),
        "num_votes": num_votes,
        "individual_votes": individual_answers,
        "all_parsed": all_parsed,
    }


def _run_batch(args: argparse.Namespace) -> None:
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"Error: video directory not found: {video_dir}", file=sys.stderr)
        sys.exit(1)

    step_dirs = sorted(video_dir.glob("step_*"))
    if args.steps is not None:
        step_set = set(args.steps)
        step_dirs = [
            d
            for d in step_dirs
            if int(d.name.removeprefix("step_").lstrip("0") or "0") in step_set
        ]
    if not step_dirs:
        print(f"No step directories found under {video_dir}", file=sys.stderr)
        sys.exit(1)

    versions = list(PROMPTS.keys()) if args.prompt == "all" else [args.prompt]
    prompt_variants = {k: PROMPTS[k] for k in versions}

    client = OpenAI(api_key="none", base_url=args.endpoint, timeout=600)

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
            if not Path(video_path).exists():
                print(f"SKIP {video_path}: not found", file=sys.stderr)
                continue
            for prompt_name, prompt_template in prompt_variants.items():
                jobs.append(
                    {
                        "step": meta["step"],
                        "traj": traj["traj"],
                        "instruction": traj.get("instruction", ""),
                        "total_reward": traj.get("total_reward"),
                        "video_path": video_path,
                        "prompt_name": prompt_name,
                        "prompt_template": prompt_template,
                    }
                )

    n_vlm_calls = len(jobs) * args.num_votes
    vote_str = f" x {args.num_votes} votes" if args.num_votes > 1 else ""
    print(
        f"Running {n_vlm_calls} VLM calls "
        f"({len(step_dirs)} steps x <= {args.max_trajs} trajs x "
        f"{len(prompt_variants)} prompts{vote_str}) ...",
        flush=True,
    )

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _run_one_batch,
                client=client,
                model=args.model,
                video_path=j["video_path"],
                instruction=j["instruction"],
                prompt_name=j["prompt_name"],
                prompt_template=j["prompt_template"],
                hflip=args.hflip,
                num_votes=args.num_votes,
                vote_temperature=args.vote_temperature,
                vote_threshold=args.vote_threshold,
            ): j
            for j in jobs
        }
        for i, fut in enumerate(as_completed(futures), 1):
            j = futures[fut]
            res = fut.result()
            gt = j["total_reward"]
            correctness = None
            if gt is not None:
                if res["answer"] == "unsure":
                    correctness = 0.5
                elif res["answer"] in ("yes", "no"):
                    correctness = 1.0 if (res["answer"] == "yes") == (gt > 0) else 0.0
                else:
                    correctness = 0.0
            record = {
                "step": j["step"],
                "traj": j["traj"],
                "video": j["video_path"],
                "instruction": j["instruction"],
                "total_reward": j["total_reward"],
                "correctness": correctness,
                **res,
            }
            results.append(record)
            status = res["answer"].upper()
            if correctness is not None:
                verdict = (
                    "CORRECT" if correctness == 1.0
                    else "UNSURE" if correctness == 0.5
                    else "WRONG"
                )
            else:
                verdict = ""
            err = f"  {res['error']}" if "error" in res else ""
            votes_str = ""
            if "individual_votes" in res:
                votes_str = f"votes={res['individual_votes']}"
            parts = [s for s in [status, verdict, votes_str, err.strip()] if s]
            print(
                f"[{i}/{len(jobs)}] step={j['step']} traj={j['traj']} "
                f"prompt={j['prompt_name']} → {' | '.join(parts)}",
                flush=True,
            )

    results.sort(key=lambda r: (r["step"], r["traj"], r["prompt"]))

    output_data = {
        "config": vars(args),
        "vote_threshold": args.vote_threshold,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults written to {args.output}")

    # --- Summary ---
    vote_label = f" (num_votes={args.num_votes})" if args.num_votes > 1 else ""
    print(f"\n=== Summary{vote_label} ===")
    for prompt_name in prompt_variants:
        subset = [r for r in results if r["prompt"] == prompt_name]
        n_ok = sum(1 for r in subset if r["parse_ok"])
        n_yes = sum(1 for r in subset if r["answer"] == "yes")
        n_no = sum(1 for r in subset if r["answer"] == "no")
        n_unsure = sum(1 for r in subset if r["answer"] == "unsure")
        parts = [f"{n_yes} yes", f"{n_no} no"]
        if n_unsure:
            parts.append(f"{n_unsure} unsure")
        print(
            f"  {prompt_name}: {n_ok}/{len(subset)} parsed | "
            f"{', '.join(parts)}"
        )

    # Agreement with ground-truth reward
    gt_available = [r for r in results if r["total_reward"] is not None]
    if gt_available:
        print("\n=== Agreement with ground-truth reward ===")
        print("  (unsure counts as 0.5 reward for scoring)")
        for prompt_name in prompt_variants:
            subset = [
                r
                for r in gt_available
                if r["prompt"] == prompt_name
                and r["answer"] in ("yes", "no", "unsure")
            ]
            if not subset:
                continue
            agree = sum(
                1
                for r in subset
                if r["answer"] != "unsure"
                and (r["answer"] == "yes") == (r["total_reward"] > 0)
            )
            n_unsure = sum(1 for r in subset if r["answer"] == "unsure")
            score = agree + 0.5 * n_unsure
            print(
                f"  {prompt_name}: {agree}/{len(subset)} exact"
                + (f" + {n_unsure} unsure" if n_unsure else "")
                + f" (score {score:.1f}/{len(subset)}, {score/len(subset):.1%})"
            )

    # Average correctness (parse fail = 0, unsure = 0.5)
    print("\n=== Average correctness (parse_fail=0, unsure=0.5) ===")
    for prompt_name in prompt_variants:
        scored = [
            r for r in results
            if r["prompt"] == prompt_name and r["correctness"] is not None
        ]
        if not scored:
            continue
        avg = sum(r["correctness"] for r in scored) / len(scored)
        n_parse_fail = sum(1 for r in scored if not r["parse_ok"])
        n_unsure = sum(1 for r in scored if r["answer"] == "unsure")
        detail_parts = [f"{len(scored)} trajectories"]
        if n_parse_fail:
            detail_parts.append(f"{n_parse_fail} parse fails (=0)")
        if n_unsure:
            detail_parts.append(f"{n_unsure} unsure (=0.5)")
        print(f"  {prompt_name}: {avg:.4f} ({', '.join(detail_parts)})")

    # Per-trajectory breakdown
    print("\n=== Per-trajectory results ===")
    keys = sorted({(r["step"], r["traj"]) for r in results})
    for step, traj in keys:
        subset = [r for r in results if r["step"] == step and r["traj"] == traj]
        inst = subset[0]["instruction"]
        gt = subset[0]["total_reward"]
        print(f"step={step} traj={traj} gt_reward={gt}  instruction: {inst!r}")
        for r in sorted(subset, key=lambda x: x["prompt"]):
            c = r.get("correctness")
            c_str = f" (correctness={c:.1f})" if c is not None else ""
            print(f"  [{r['prompt']}] {r['answer'].upper()}{c_str}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test HER reward eval prompts on trajectory videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Path to a single .mp4 video (single-video mode)",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="Task instruction (required for single-video mode)",
    )
    parser.add_argument(
        "--video-dir",
        default=_DEFAULT_VIDEO_DIR,
        help="Root videos/ directory for batch mode (default: %(default)s)",
    )
    parser.add_argument(
        "--steps",
        nargs="*",
        type=int,
        default=None,
        help="Step indices to process in batch mode (default: all)",
    )
    parser.add_argument(
        "--max-trajs",
        type=int,
        default=999,
        help="Max trajectories per step in batch mode (default: all)",
    )
    parser.add_argument(
        "--endpoint",
        default="http://VLM_ENDPOINT_HOST:PORT/v1",
        help="OpenAI-compatible endpoint URL",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-235B-A22B-Thinking-FP8",
        help="Model name",
    )
    parser.add_argument(
        "--prompt",
        choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "all"],
        default="all",
        help="Which prompt version(s) to run (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel VLM calls for batch mode (default: 8)",
    )
    parser.add_argument(
        "--output",
        default="scripts/traj_eval/reward_eval_results.json",
        help="Output JSON path for batch mode",
    )
    parser.add_argument(
        "--hflip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Horizontally flip video frames before sending to VLM (default: True)",
    )
    parser.add_argument(
        "--num-votes",
        type=int,
        default=1,
        help="Number of VLM queries per trajectory for majority voting (default: 1)",
    )
    parser.add_argument(
        "--vote-temperature",
        type=float,
        default=0.6,
        help="Temperature for VLM calls when num_votes > 1 (default: 0.6)",
    )
    parser.add_argument(
        "--vote-threshold",
        type=float,
        default=0.7,
        help="Average score threshold for yes/no decision. "
        "avg >= threshold → yes, avg <= 1-threshold → no, else unsure (default: 0.7)",
    )
    args = parser.parse_args()

    if args.output == "scripts/traj_eval/reward_eval_results.json":
        parts = []
        if args.prompt != "all":
            parts.append(args.prompt)
        if args.num_votes > 1:
            parts.append(f"votes{args.num_votes}")
        if parts:
            args.output = f"scripts/traj_eval/reward_eval_results_{'_'.join(parts)}.json"

    if args.video:
        if not args.instruction:
            parser.error("--instruction is required when using --video")
        _run_single(args)
    else:
        _run_batch(args)


if __name__ == "__main__":
    main()
