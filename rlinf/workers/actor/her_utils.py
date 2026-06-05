# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared utilities for HER (Hindsight Experience Replay) actor workers."""

import base64
import os
import tempfile
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------

# Max frames sent to the VLM per trajectory.  vllm's do_sample_frames may
# undersample and drop tail frames that contain the achieved outcome.
# We subsample ourselves so the VLM always sees the first 2 and last 4 frames.
HER_VLM_MAX_FRAMES = 32


def select_frame_indices_for_vlm(
    T: int,
    max_frames: int = HER_VLM_MAX_FRAMES,
) -> list[int]:
    """Return the indices into a T-frame sequence that select_frames_for_vlm would pick.

    Strategy:
      - Always keep the first 2 frames  (starting state)
      - Always keep the last 4 frames   (achieved outcome — most informative)
      - Fill the remaining slots with uniformly-spaced middle frames
    """
    if T <= max_frames:
        return list(range(T))

    n_head, n_tail = 2, 4
    n_middle = max_frames - n_head - n_tail
    if n_middle <= 0:
        return list(range(T - max_frames, T))

    span = T - n_head - n_tail
    step = span / (n_middle + 1)
    middle = [int(n_head + step * (i + 1)) for i in range(n_middle)]
    return sorted(set(list(range(n_head)) + middle + list(range(T - n_tail, T))))


def select_frames_for_vlm(
    frames: list,
    max_frames: int = HER_VLM_MAX_FRAMES,
) -> list:
    """Select up to *max_frames* from *frames*, guaranteeing first/last coverage.

    This prevents vllm's internal subsampling from accidentally dropping the
    end of the trajectory, which is the most important part for hindsight
    instruction generation.
    """
    indices = select_frame_indices_for_vlm(len(frames), max_frames)
    return [frames[i] for i in indices]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

HER_PROMPT_TEMPLATE = (
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
)

HER_PROMPT_TEMPLATE_V2 = (
    "You are labeling a robot manipulation trajectory with a hindsight "
    "instruction that accurately describes what the robot ACTUALLY DID, "
    "not what it was asked to do.\n\n"
    "Original instruction: \"{original_instruction}\"\n\n"
    "## Analysis steps\n"
    "1. **Identify objects**: Which objects does the gripper contact or move? "
    "Use the exact object names visible in the scene. Do NOT copy object names "
    "from the original instruction if the robot interacted with different objects.\n"
    "2. **Track gripper trajectory**: Where does the gripper start, what does "
    "it grasp (if anything), where does it transport the object, and where "
    "does it release?\n"
    "3. **Note the end-state**: What is the final spatial relationship between "
    "the manipulated object(s) and the surrounding landmarks?\n"
    "4. **Compare to original**: Did the robot complete the original task, "
    "partially complete it, or do something entirely different?\n\n"
    "## Instruction rules\n"
    "- Start with an imperative verb.\n"
    "- Must describe the FULL trajectory from start to end.\n"
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
)

HER_PROMPT_TEMPLATE_V3 = (
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
)

HER_PROMPT_TEMPLATE_V4 = (
    "Watch this video of a robot arm performing a manipulation task.\n"
    "The robot was originally instructed to: \"{original_instruction}\"\n"
    "However, the robot may have FAILED or done something DIFFERENT from "
    "the original instruction. Your job is to describe what the robot "
    "ACTUALLY DID based solely on visual evidence.\n\n"
    "Objects present in the scene: {scene_objects_text}\n\n"
    "Focus on three things:\n"
    "1. Which object(s) from the scene did the gripper contact or grasp?\n"
    "2. Where did the object(s) move to?\n"
    "3. What is the final spatial arrangement?\n\n"
    "Rules:\n"
    "- Write ONE imperative instruction (e.g., 'pick up the X and place "
    "it on the Y').\n"
    "- Use the exact object names listed above when referring to scene objects.\n"
    "- If the robot failed to grasp anything, describe its movement path "
    "(e.g., 'move the gripper above the table').\n"
    "- Do NOT copy the original instruction — describe what you SEE.\n"
    "- Under 20 words.\n\n"
    "Output:\n"
    "<thought>brief reasoning about what happened</thought>\n"
    "<final>your instruction</final>"
)

HER_PROMPT_TEMPLATE_V5 = (
    "Watch this video of a robot arm performing a manipulation task.\n"
    "The robot was originally instructed to: \"{original_instruction}\"\n"
    "However, the robot may have FAILED or done something DIFFERENT from "
    "the original instruction. Your job is to describe what the robot "
    "ACTUALLY DID based solely on visual evidence.\n\n"
    "Objects present in the scene: {scene_objects_text}\n\n"
    "First, determine whether the robot did anything INTERESTING.\n"
    "Interesting behavior means the robot touched, grasped, pushed, "
    "pulled, or otherwise physically interacted with any object.\n"
    "Uninteresting behavior means the robot merely hovered in the air, "
    "moved without contacting any object, or stayed idle.\n\n"
    "If the behavior is UNINTERESTING (no object contact or interaction), "
    "output exactly:\n"
    "<thought>brief reasoning about why nothing interesting happened</thought>\n"
    "<final>Nothing</final>\n\n"
    "If the behavior IS interesting, focus on:\n"
    "1. Which object(s) from the scene did the gripper contact or grasp?\n"
    "2. Where did the object(s) move to?\n"
    "3. What is the final spatial arrangement?\n\n"
    "Rules:\n"
    "- Write ONE imperative instruction (e.g., 'pick up the X and place "
    "it on the Y').\n"
    "- Use the exact object names listed above when referring to scene objects.\n"
    "- Do NOT copy the original instruction — describe what you SEE.\n"
    "- Under 20 words.\n\n"
    "Output:\n"
    "<thought>brief reasoning about what happened</thought>\n"
    "<final>your instruction</final>"
)

HER_PROMPT_TEMPLATE_V6 = (
    "Watch this video of a robot arm performing a manipulation task.\n"
    "The robot was originally instructed to: \"{original_instruction}\"\n"
    "However, the robot may have FAILED or done something DIFFERENT from "
    "the original instruction. Your job is to describe what the robot "
    "ACTUALLY DID based solely on visual evidence.\n\n"
    "First, determine whether the robot did anything INTERESTING.\n"
    "Interesting behavior means the robot touched, grasped, pushed, "
    "pulled, or otherwise physically interacted with any object.\n"
    "Uninteresting behavior means the robot merely hovered in the air, "
    "moved without contacting any object, or stayed idle.\n\n"
    "If the behavior is UNINTERESTING (no object contact or interaction), "
    "output exactly:\n"
    "<thought>brief reasoning about why nothing interesting happened</thought>\n"
    "<final>Nothing</final>\n\n"
    "If the behavior IS interesting, focus on:\n"
    "1. Which object(s) did the gripper contact or grasp?\n"
    "2. Where did the object(s) move to?\n"
    "3. What is the final spatial arrangement?\n\n"
    "Rules:\n"
    "- Write ONE imperative instruction (e.g., 'pick up the X and place "
    "it on the Y').\n"
    "- Do NOT copy the original instruction — describe what you SEE.\n"
    "- Under 20 words.\n\n"
    "Output:\n"
    "<thought>brief reasoning about what happened</thought>\n"
    "<final>your instruction</final>"
)

HER_PROMPT_TEMPLATE_V7 = (
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
)

HER_PROMPT_TEMPLATE_V7_NO_UNINTERESTING = (
    "Watch this video of a robot arm performing a manipulation task.\n"
    "The robot was originally instructed to: \"{original_instruction}\"\n"
    "However, the robot may have FAILED or done something DIFFERENT from "
    "the original instruction. Your job is to describe what the robot "
    "ACTUALLY DID based solely on visual evidence.\n\n"
    "Objects present in the scene: {scene_objects_text}\n\n"
    "Focus on:\n"
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
)

HER_PROMPT_TEMPLATE_V7_NO_OBJECTS = (
    "Watch this video of a robot arm performing a manipulation task.\n"
    "The robot was originally instructed to: \"{original_instruction}\"\n"
    "However, the robot may have FAILED or done something DIFFERENT from "
    "the original instruction. Your job is to describe what the robot "
    "ACTUALLY DID based solely on visual evidence.\n\n"
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
)

HER_PROMPT_TEMPLATE_V10 = (
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
)

HER_PROMPT_TEMPLATE_V11 = (
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
)

HER_PROMPT_TEMPLATE_V11_STRICT = (
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
)

HER_PROMPT_TEMPLATE_V12 = (
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
)

HER_PROMPT_TEMPLATE_SUBTRAJ = (
    "Watch this video of a robot arm performing a manipulation task.\n"
    "The robot was originally instructed to: \"{original_instruction}\"\n"
    "{scene_objects_line}"
    "All objects start on the table or ground. Use this fact to judge whether any "
    "object has actually been lifted.\n\n"
    "## Step 1 — Classify the trajectory\n\n"
    "Determine which of these three cases applies:\n\n"
    "CASE A — Meaningful manipulation: the robot grasped an object and it visibly "
    "lifted off the surface (pick), or transported and released it at a new location "
    "(place). This is the ONLY case that counts as a completed action.\n"
    "  Strict lift check: if the object is still resting on the table/ground at the "
    "end of the contact, the robot did NOT successfully pick it up.\n\n"
    "CASE B — Reaching only: the robot moved toward an object and its gripper made "
    "contact with or came very close to touching the object, but the object never "
    "left the surface. This includes tapping, bumping, or nearly grasping.\n\n"
    "CASE C — Nothing useful: the robot did not approach or contact any object — "
    "it merely hovered, moved randomly, or stayed idle throughout the video.\n\n"
    "## Step 2 — Find the useful prefix and write the instruction\n\n"
    "CASE A:\n"
    "  Find when the meaningful manipulation ends. If the robot drifts or idles after "
    "completing the action, cut off there; otherwise use the full video (cutoff=1.0).\n"
    "  Write an instruction describing what was actually picked/placed.\n\n"
    "CASE B:\n"
    "  Find the moment when the gripper is closest to or touching the target object "
    "— that is the cutoff point.\n"
    "  Write an instruction CLOSE TO THE ORIGINAL that describes the intended action "
    "on the object the robot was reaching for. For example: if the original says "
    "'pick up the milk' and the robot's gripper touched the milk box, the instruction "
    "is 'pick up the milk'. Use the original instruction as a guide — keep the same "
    "action verb and target object if the robot was reaching toward that object; "
    "adapt only if the robot was clearly going for a different object.\n\n"
    "CASE C:\n"
    "  Output <final>Nothing</final> with no <cutoff_fraction>.\n\n"
    "## Output format\n\n"
    "For CASE A or CASE B — ALWAYS include <cutoff_fraction>:\n"
    "<thought>which case applies and why; identify the cutoff moment</thought>\n"
    "<cutoff_fraction>decimal 0.0–1.0 (fraction of total video at the end of the "
    "useful prefix; 1.0 = full video)</cutoff_fraction>\n"
    "<final>one imperative instruction (under 20 words)</final>\n\n"
    "For CASE C:\n"
    "<thought>why the robot did nothing useful</thought>\n"
    "<final>Nothing</final>\n\n"
    "## Rules for <final>\n"
    "- Start with an imperative verb.\n"
    "- Use SHORT, GENERIC object names: strip numeric suffixes (_1, _2) and "
    "brand/material prefixes "
    "(e.g. 'akita_black_bowl_1' → 'bowl', 'flat_stove_1' → 'stove', "
    "'white_yellow_mug_1' → 'yellow and white mug').\n"
    "- Under 20 words.\n"
    "- For CASE B: keep the instruction as close to the original as possible while "
    "accurately describing the reaching behavior toward the target object."
)

HER_PROMPT_TEMPLATE_SUBTRAJ_V2 = (
    "Watch this video of a robot arm performing a manipulation task.\n"
    "The robot was originally instructed to: \"{original_instruction}\"\n"
    "The robot may or may not have completed the task. Your job is to find the most "
    "informative label for what actually happened.\n"
    "{scene_objects_line}"
    "NOTE: You are seeing {n_vlm_frames} frames total (frames are numbered 0 to "
    "{n_vlm_frames_minus1}).\n\n"
    "PRIORITY: Always prefer finding a useful prefix over outputting Nothing. "
    "Output Nothing ONLY as a last resort when the robot is completely idle "
    "(arm barely moves, no objects touched or approached) for the entire video.\n\n"
    "Step 1 — Check the LAST FRAME first (frame {n_vlm_frames_minus1}).\n\n"
    "Look ONLY at visual evidence — do NOT assume failure or success beforehand.\n\n"
    "- If the object is visibly at its destination in the last frame (e.g. placed on "
    "the stove, inside the box, in the caddy) → the task succeeded. "
    "You MUST output Form A. Do NOT cut off. Do NOT invent reasons for failure.\n"
    "- If the last frame shows the task is clearly incomplete → proceed to Step 2.\n\n"
    "Step 2 — Did the robot make any meaningful MOVEMENT toward an object?\n\n"
    "LOW BAR FOR CONTACT — any of the following counts:\n"
    "- Object shifts, tilts, or moves even slightly in response to the gripper\n"
    "- Gripper fingers visibly close around or press against the object\n"
    "- Robot clearly reaches toward and nearly touches an object\n\n"
    "DEPTH WARNING: These are monocular camera images — the gripper can appear "
    "close to an object while actually hovering above or beside it at a different "
    "depth. Do NOT assume contact from proximity alone. The most reliable signal "
    "for contact is that the object itself moves or tilts. If the object never "
    "moves and the gripper fingers never visibly wrap around it, treat it as "
    "hovering, not contact.\n\n"
    "Only output Nothing if the arm makes no directed movement toward any object "
    "across ALL frames.\n\n"
    "If NO meaningful movement, output exactly:\n"
    "<thought>brief reasoning</thought>\n"
    "<final>Nothing</final>\n\n"
    "Step 3 — Find the USEFUL PREFIX and decide its cutoff frame.\n\n"
    "The useful prefix is the longest initial segment of directed, purposeful behavior. "
    "It ends at the last frame where the robot is still making progress.\n\n"
    "LABEL THE PREFIX by what the object's state is AT THE CUTOFF FRAME — "
    "NOT by the original instruction, NOT by what you expect will happen next.\n\n"
    "STATE CHECK at the cutoff frame (apply depth awareness — confirm with object motion):\n"
    "- Is the object visibly at its destination? → \"place the X on/in Y\"\n"
    "- Is the object lifted off its original surface and in the gripper? → \"pick up the X\"\n"
    "- Did the gripper close around the object but object is still on the surface? → \"grasp the X\"\n"
    "- Did the gripper make brief confirmed contact (object twitched or shifted)? → \"touch the X\"\n"
    "- Did the gripper only reach toward the object without confirmed contact? → \"reach for the X\"\n\n"
    "Be conservative: use the WEAKEST label that is confirmed by the visual evidence "
    "at the cutoff frame. Do not infer contact that you cannot see confirmed.\n\n"
    "When in doubt about whether to cut off: prefer Form A (whole-video).\n\n"
    "Output format — choose exactly one:\n\n"
    "Form A (whole-video):\n"
    "<thought>step-by-step analysis</thought>\n"
    "<final>one imperative instruction describing the full video</final>\n\n"
    "Form B (prefix sub-trajectory):\n"
    "<thought>step-by-step analysis including when/why the behavior changes</thought>\n"
    "<cutoff_frame>integer 0 to {n_vlm_frames_minus1}: 0-indexed frame where "
    "the useful behavior ends</cutoff_frame>\n"
    "<final>one imperative instruction describing only the prefix</final>\n\n"
    "Rules for <final>:\n"
    "- Start with an imperative verb.\n"
    "- SHORT, GENERIC object names: strip numeric suffixes and brand/material prefixes "
    "(e.g. \"akita_black_bowl_1\" → \"bowl\", \"flat_stove_1\" → \"stove\").\n"
    "- Do NOT copy the original instruction — describe what you SEE.\n"
    "- Under 20 words."
)


def build_subtraj_prompt_v2(
    original_instruction: str,
    n_vlm_frames: int,
    scene_objects: Optional[list] = None,
) -> str:
    """Format HER_PROMPT_TEMPLATE_SUBTRAJ_V2 with the given parameters."""
    scene_objects_line = (
        f"Objects present in the scene: {', '.join(scene_objects)}\n"
        if scene_objects else ""
    )
    return HER_PROMPT_TEMPLATE_SUBTRAJ_V2.format(
        original_instruction=original_instruction,
        scene_objects_line=scene_objects_line,
        n_vlm_frames=n_vlm_frames,
        n_vlm_frames_minus1=max(0, n_vlm_frames - 1),
    )


def parse_subtraj_response(
    raw: str,
    vlm_indices: list[int],
    total_frames: int,
) -> tuple[Optional[str], Optional[float]]:
    """Parse a SUBTRAJ_V2 VLM response into (instruction, cutoff_fraction).

    SUBTRAJ_V2 uses ``<cutoff_frame>`` (integer VLM-space index) instead of
    ``<cutoff_fraction>``.  Converts the VLM frame index to an original-space
    cutoff fraction via *vlm_indices*.

    Returns:
        (instruction, cutoff_fraction) where:
          - instruction is None if parsing failed or the response was invalid
          - cutoff_fraction is None for Form A (whole-video) responses
          - cutoff_fraction is in (0, 1] for Form B (prefix) responses
    """
    import re

    # Strip thinking block; model sometimes omits opening <think> tag.
    if "</think>" in raw:
        clean = raw[raw.rindex("</think>") + len("</think>"):].strip()
    else:
        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    final_m = re.search(r"<final>\s*(.*?)\s*</final>", clean, flags=re.DOTALL)
    cutoff_m = re.search(r"<cutoff_frame>\s*(\d+)\s*</cutoff_frame>", clean)

    instruction = final_m.group(1).strip() if final_m else None
    if instruction is None:
        return None, None

    # Model sometimes puts reasoning inside <final>; real answer is the last line.
    if "\n" in instruction:
        last_line = instruction.rstrip().rsplit("\n", 1)[-1].strip()
        if last_line:
            instruction = last_line

    # Strip any residual XML tags the model nested inside <final>.
    instruction = re.sub(r"<[^>]+>", "", instruction).strip()

    if not instruction:
        return None, None

    instruction = instruction[0].upper() + instruction[1:]

    # Reject responses that echo back prompt rules instead of describing the video.
    _RULE_ECHO_PHRASES = (
        "imperative verb", "under 20 words", "generic name",
        "numeric suffix", "do not copy", "form a", "form b", "one imperative",
    )
    if any(phrase in instruction.lower() for phrase in _RULE_ECHO_PHRASES):
        return None, None

    # Reject over-long instructions (matches request_hindsight_instruction behaviour).
    if instruction.lower() != "nothing" and len(instruction.split()) > 20:
        return None, None

    # "Nothing" → no cutoff.
    if instruction.lower() == "nothing":
        return "Nothing", None

    # Form B: convert VLM frame index → original frame index → fraction.
    # Form A has no <cutoff_frame> tag; cutoff_fraction stays None (whole-video).
    cutoff_fraction: Optional[float] = None
    if cutoff_m is not None:
        n_vlm = len(vlm_indices)
        vlm_frame = max(0, min(int(cutoff_m.group(1)), n_vlm - 1))
        orig_frame = vlm_indices[vlm_frame]
        cutoff_fraction = (orig_frame + 1) / total_frames

    return instruction, cutoff_fraction


HER_NEGATIVE_PROMPT_TEMPLATE = (
    "Watch this video of a robot arm performing a manipulation task.\n"
    "The robot was originally instructed to: \"{original_instruction}\"\n"
    "Based on the video, the robot actually did: \"{positive_instruction}\"\n\n"
    "Your task is to generate ONE alternative task description that:\n"
    "1. Uses the SAME objects and workspace visible in the video.\n"
    "2. Sounds like a plausible robot manipulation task in this scene.\n"
    "3. Is DIFFERENT from what the robot actually did: \"{positive_instruction}\"\n"
    "4. Is specific enough that a robot following it would behave differently.\n\n"
    "Rules:\n"
    "- Start with a verb.\n"
    "- Reference objects actually visible in the scene.\n"
    "- Do NOT describe what the robot did — this must be a different action.\n"
    "- Under 20 words.\n\n"
    "Output format:\n"
    "<thought>brief reasoning about a plausible alternative task</thought>\n"
    "<final>your alternative task description</final>"
)

HER_PROMPT_TEMPLATE_LEAK_STUDY = (
    "Watch this video of a robot arm performing a manipulation task.\n"
    "The robot was originally instructed to: \"{original_instruction}\"\n\n"
    "Your job is to describe what the robot ACTUALLY DID based on visual evidence.\n\n"
    "## Reference task variants (PREFERRED)\n"
    "If the robot's behavior matches any of these task descriptions, "
    "output that description VERBATIM. These are the preferred outputs:\n"
    "{test_instructions_text}"
    "\n\n"
    "## Decision process\n"
    "1. Watch the full video and carefully observe the robot's behavior.\n"
    "2. For EACH reference variant above, ask: Does the robot's observed behavior "
    "match this description?\n"
    "3. If YES to any variant: Output that variant text EXACTLY as written (preferred).\n"
    "4. If NO to all variants: Generate a new instruction that describes what you saw.\n"
    "5. Output only ONE final instruction (never combine multiple variants).\n"
    "6. Keep under 20 words. Do NOT hallucinate or invent actions.\n"
    "7. The instruction MUST start with an imperative verb "
    "(e.g. pick, put, place, turn, open, close, push, pull).\n\n"
    "Output format (strict):\n"
    "<thought>which reference variant (if any) best describes the behavior? "
    "If none match, what did the robot actually do?</thought>\n"
    "<final>exact reference variant (if matched) or your own instruction</final>"
)

HER_REPHRASE_PROMPT_TEMPLATE = (
    "You are given a robot manipulation instruction. "
    "Rephrase it using different words while keeping exactly the same meaning.\n\n"
    "Original instruction: {original_instruction}\n\n"
    "Rules:\n"
    "1. The rephrased instruction must describe the same task with the same objects.\n"
    "2. Use different words or sentence structure.\n"
    "3. Keep it concise (under 15 words).\n"
    "4. Start with an imperative verb (e.g. pick, put, place, turn, open, close, push, pull).\n"
    "5. Do NOT add, remove, or change any objects or actions.\n\n"
    "Output your rephrased instruction inside <final> tags.\n"
    "Example: <final>Place the red cup onto the left side of the table</final>"
)


HER_REWARD_EVAL_PROMPT_TEMPLATE = (
    "Watch this video of a robot performing a task.\n"
    "The robot was asked to: \"{instruction}\"\n"
    "Did the robot successfully complete this instruction?\n"
    "Requirements:\n"
    "- Consider the full video from start to finish.\n"
    "- Answer based on whether the final state matches the instruction.\n"
    "Output format must be exactly:\n"
    "<thought>your reasoning</thought>\n"
    "<answer>yes</answer> or <answer>no</answer>"
)

HER_REWARD_EVAL_PROMPT_TEMPLATE_V3 = (
    "Watch this video of a robot arm performing a manipulation task.\n"
    "The robot was instructed to: \"{instruction}\"\n\n"
    "Determine whether the robot FULLY and SUCCESSFULLY completed the instruction.\n\n"
    "## Step 1 — Evaluate success\n\n"
    "Criteria (ALL must be met for YES):\n"
    "1. The robot makes physical contact with the correct object(s).\n"
    "2. The robot performs the exact action stated.\n"
    "3. The final state clearly matches the goal.\n"
    "4. The task reaches completion — not merely attempted or partially executed.\n\n"
    "## Step 2 — If not successful, find the useful prefix\n\n"
    "If the robot partially attempted the task (approached or touched the target object):\n"
    "- Find the frame where the robot was closest to completing the action "
    "(e.g., nearest to the target, or the moment of best contact).\n"
    "- That moment is the cutoff. Output it as a fraction of the total video.\n\n"
    "If the robot made no progress toward the instruction (no approach or contact), "
    "do not output a cutoff.\n\n"
    "## Output format\n\n"
    "If YES (fully successful):\n"
    "<thought>step-by-step reasoning; describe the final frame explicitly</thought>\n"
    "<answer>yes</answer>\n\n"
    "If NO with partial progress:\n"
    "<thought>what the robot did; identify the cutoff moment</thought>\n"
    "<cutoff_fraction>decimal 0.0–1.0</cutoff_fraction>\n"
    "<answer>no</answer>\n\n"
    "If NO with no useful progress:\n"
    "<thought>why the robot made no useful progress</thought>\n"
    "<answer>no</answer>\n\n"
    "Rules:\n"
    "- Do NOT give benefit of the doubt. If uncertain, answer NO.\n"
    "- Before answering, explicitly describe what you see in the LAST frame.\n"
    "- <cutoff_fraction> must always come BEFORE <answer>.\n"
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

HER_REWARD_EVAL_PROMPT_TEMPLATE_V2 = (
    "Watch this video of a robot arm performing a manipulation task.\n"
    "The robot was instructed to: \"{instruction}\"\n\n"
    "Determine whether the robot FULLY and SUCCESSFULLY completed the instruction.\n\n"
    "Evaluation criteria (all must be satisfied for YES):\n"
    "1. The robot makes physical contact with the correct object(s) named in the instruction.\n"
    "2. The robot performs the correct action as stated.\n"
    "3. The final state of the scene matches the goal described in the instruction.\n"
    "4. The task reaches completion — the robot does NOT merely attempt or partially execute it.\n\n"
    "Strict rules:\n"
    "- Answer NO if the robot touches the wrong object, drops the object before completing "
    "placement, or ends the episode without the target object reaching its goal state.\n"
    "- Answer NO if the final frame does not clearly show the task goal achieved.\n"
    "- Do NOT give benefit of the doubt. If you are uncertain, answer NO.\n\n"
    "CRITICAL — final frame verification:\n"
    "Before answering, explicitly describe what you see in the LAST frame of the video.\n\n"
    "Output format must be exactly:\n"
    "<thought>1) Describe what the robot did step by step. "
    "2) Describe exactly what you see in the final frame. "
    "3) State whether each criterion is satisfied based on the final frame.</thought>\n"
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
    "answer NO. Do NOT give benefit of the doubt.\n\n"
    "Output format must be exactly:\n"
    "<thought>your step-by-step analysis following steps 1-5 above</thought>\n"
    "<answer>yes</answer> or <answer>no</answer>"
)

# ---------------------------------------------------------------------------
# Video FPS
# ---------------------------------------------------------------------------

_HER_VIDEO_FPS = 2

# ---------------------------------------------------------------------------
# Frame / video helpers
# ---------------------------------------------------------------------------


def _to_vlm_frame_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert a preprocessed pixel tensor to a uint8 HWC numpy array."""
    if t.dim() == 4:
        t = t[0]
    img = t.detach().cpu()
    # OpenPI path: raw env image in HWC (uint8 or float).
    if img.dim() == 3 and img.shape[-1] == 3:
        if not torch.is_floating_point(img):
            return img.numpy()
        scale = 255.0 if img.max().item() <= 1.5 else 1.0
        img = (img * scale).clamp(0, 255)
        return img.to(dtype=torch.uint8).contiguous().numpy()
    # OpenVLA/OpenVLA-OFT: preprocessed CHW with channel-stacked streams.
    if img.dim() == 3 and img.shape[0] >= 6:
        img = img.float()
        img = img[3:6, :, :]
        img = (img * 0.5 + 0.5).clamp(0, 1)
        return (img.permute(1, 2, 0).contiguous().numpy() * 255).astype("uint8")
    if img.dim() == 3 and img.shape[0] == 3:
        if not torch.is_floating_point(img):
            img = img.float() / 255.0
        else:
            img = (
                img.clamp(0, 1)
                if img.min().item() >= 0
                else (img * 0.5 + 0.5).clamp(0, 1)
            )
        return (img.permute(1, 2, 0).contiguous().numpy() * 255).astype("uint8")
    raise ValueError(f"Unsupported frame tensor shape for VLM: {tuple(img.shape)}")


def _pixel_tensors_to_video_base64(
    tensors: list,
    fps: int = _HER_VIDEO_FPS,
    flip_horizontal: bool = False,
) -> str:
    """Encode pixel tensors as a base64 mp4 video for VLM input."""
    import imageio

    frames = []
    for t in tensors:
        frame = _to_vlm_frame_uint8(t)
        if flip_horizontal:
            frame = frame[:, ::-1, :]
        frames.append(frame)

    if not frames:
        raise ValueError("No valid frames to encode for HER video.")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        imageio.mimwrite(tmp_path, frames, fps=fps)
        with open(tmp_path, "rb") as f:
            video_bytes = f.read()
    finally:
        os.unlink(tmp_path)
    return base64.b64encode(video_bytes).decode("utf-8")


def _pixel_tensors_to_frame_base64s(
    tensors: list, flip_horizontal: bool = False
) -> list:
    """Encode pixel tensors as a list of base64 JPEG images."""
    import cv2

    b64_frames = []
    for t in tensors:
        frame = _to_vlm_frame_uint8(t)
        if flip_horizontal:
            frame = frame[:, ::-1, :]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", frame_bgr)
        b64_frames.append(base64.standard_b64encode(buffer).decode("utf-8"))
    return b64_frames


def _save_trajectory_video(
    frames: list,
    fps: int = _HER_VIDEO_FPS,
    flip_horizontal: bool = False,
    out_path: Optional[str] = None,
) -> str:
    """Save trajectory frames as mp4, return the file path."""
    import imageio

    out_frames = []
    for t in frames:
        frame = _to_vlm_frame_uint8(t)
        if flip_horizontal:
            frame = frame[:, ::-1, :]
        out_frames.append(frame)
    if out_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        out_path = tmp.name
        tmp.close()
    imageio.mimwrite(
        out_path,
        out_frames,
        fps=fps,
        output_params=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
    )
    return out_path
