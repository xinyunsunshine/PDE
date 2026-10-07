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

"""HERVLMClient: OpenAI-compatible VLM client for HER instruction and reward queries."""

import random
import re
from typing import Callable, Optional

import backoff
import openai
from openai import OpenAI

from pde.rlinf.prompting import (
    _HER_VIDEO_FPS,
    HER_PROMPT_TEMPLATE,
    HER_PROMPT_TEMPLATE_SUBTRAJ,
    HER_PROMPT_TEMPLATE_V2,
    HER_PROMPT_TEMPLATE_V3,
    HER_PROMPT_TEMPLATE_V4,
    HER_PROMPT_TEMPLATE_V5,
    HER_PROMPT_TEMPLATE_V6,
    HER_PROMPT_TEMPLATE_V7,
    HER_PROMPT_TEMPLATE_V10,
    HER_PROMPT_TEMPLATE_V11,
    HER_PROMPT_TEMPLATE_V11_STRICT,
    HER_PROMPT_TEMPLATE_V12,
    HER_PROMPT_TEMPLATE_V7_NO_OBJECTS,
    HER_PROMPT_TEMPLATE_V7_NO_UNINTERESTING,
    HER_REPHRASE_PROMPT_TEMPLATE,
    HER_REWARD_EVAL_PROMPT_TEMPLATE,
    HER_REWARD_EVAL_PROMPT_TEMPLATE_V2,
    HER_REWARD_EVAL_PROMPT_TEMPLATE_V3,
    HER_REWARD_EVAL_PROMPT_TEMPLATE_V4,
    HER_REWARD_EVAL_PROMPT_TEMPLATE_V6,
    HER_REWARD_EVAL_PROMPT_TEMPLATE_V7,
    HER_VLM_MAX_FRAMES,
    _pixel_tensors_to_frame_base64s,
    _pixel_tensors_to_video_base64,
    build_subtraj_prompt_v2,
    parse_subtraj_response,
    select_frame_indices_for_vlm,
    select_frames_for_vlm,
)


class HERVLMClient:
    """OpenAI-compatible VLM client for HER relabeling and reward evaluation.

    Handles three concerns:
    - Transport: encoding frames as mp4/jpeg, building message content, calling
      the API with the right ``extra_body`` for each backend.
    - Prompt selection: picking and formatting the right prompt template for
      a given ``her_prompt_version``.
    - Retry / parsing: backoff on transient API errors; retry loop until a
      well-formed response tag (``<final>`` or ``<answer>``) is found.

    Supported backends (``video_mode``):
      ``"mp4"``               — single mp4 video, thinking enabled
      ``"frames"``            — JPEG frames, thinking enabled
      ``"frames_no_thinking"``— JPEG frames, thinking disabled (instruct model)
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        video_mode: str,
        api_key: str,
        max_frames: int = HER_VLM_MAX_FRAMES,
        flip_horizontal: bool = False,
        reward_eval_prompt_version: int = 1,
        log_warning_fn: Optional[Callable[[str], None]] = None,
        endpoint_file: Optional[str] = None,
    ) -> None:
        self.endpoint = endpoint
        self.endpoint_file = endpoint_file
        self.model = model
        self.video_mode = video_mode
        self.api_key = api_key
        self.max_frames = max_frames
        self.flip_horizontal = flip_horizontal
        self.reward_eval_prompt_version = reward_eval_prompt_version
        self._log_warning_fn = log_warning_fn or (lambda msg: None)
        self._last_endpoint_from_file: Optional[str] = None

    def log_warning(self, msg: str) -> None:
        self._log_warning_fn(msg)

    # ── Low-level API call ────────────────────────────────────────────────────

    def _resolve_endpoint(self) -> str:
        """Return the endpoint to use for the next request.

        If ``endpoint_file`` is configured, read it fresh each call so the
        router URL can be updated mid-run by editing the file. Raises on
        missing/empty file (no silent fallback to ``self.endpoint``).
        """
        if self.endpoint_file is None:
            return self.endpoint
        try:
            with open(self.endpoint_file) as f:
                endpoint = f.read().strip()
        except OSError as e:
            raise RuntimeError(
                f"[her] failed to read endpoint file {self.endpoint_file!r}: {e}"
            ) from None
        if not endpoint:
            raise RuntimeError(
                f"[her] endpoint file {self.endpoint_file!r} is empty"
            )
        if endpoint != self._last_endpoint_from_file:
            self.log_warning(
                f"[her] endpoint updated from file {self.endpoint_file!r}: "
                f"{self._last_endpoint_from_file!r} -> {endpoint!r}"
            )
            self._last_endpoint_from_file = endpoint
        return endpoint

    def _make_client(self, timeout: int = 600) -> OpenAI:
        endpoint = self._resolve_endpoint()
        return OpenAI(
            api_key=self.api_key,
            base_url=endpoint or None,
            timeout=timeout,
        )

    def _build_message_content(self, frame_tensors: list, prompt_text: str) -> list:
        """Encode frames and assemble the multimodal message content list."""
        if self.video_mode == "mp4":
            video_b64 = _pixel_tensors_to_video_base64(
                frame_tensors, flip_horizontal=self.flip_horizontal
            )
            return [
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
                },
                {"type": "text", "text": prompt_text},
            ]
        else:
            frame_b64s = _pixel_tensors_to_frame_base64s(
                frame_tensors, flip_horizontal=self.flip_horizontal
            )
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{fb64}"},
                }
                for fb64 in frame_b64s
            ]
            content.append({"type": "text", "text": prompt_text})
            return content

    def _build_extra_body(self) -> dict:
        """Backend-specific extra_body for the completions request."""
        if self.video_mode == "mp4":
            return {
                "mm_processor_kwargs": {
                    "fps": _HER_VIDEO_FPS,
                    "do_sample_frames": True,
                },
                "chat_template_kwargs": {"enable_thinking": True},
            }
        if self.video_mode in ("frames", "frames_no_thinking"):
            return {
                "chat_template_kwargs": {
                    "enable_thinking": self.video_mode == "frames",
                },
            }
        return {}

    @backoff.on_exception(
        backoff.expo,
        (openai.APIConnectionError, openai.InternalServerError, openai.RateLimitError),
        max_tries=None,
        on_backoff=lambda d: d["args"][0].log_warning(
            f"[her] API error on attempt {d['tries']}, retrying: {d['exception']}"
        ),
    )
    def call(self, frame_tensors: list, prompt_text: str, max_tokens: int) -> str:
        """Send frames + prompt to the VLM and return the raw response text.

        Selects up to ``max_frames`` frames before encoding.
        """
        frame_tensors = select_frames_for_vlm(frame_tensors, self.max_frames)
        content = self._build_message_content(frame_tensors, prompt_text)
        extra_body = self._build_extra_body()

        client = self._make_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            **({"extra_body": extra_body} if extra_body else {}),
        )
        if not response.choices:
            raise RuntimeError(
                f"VLM returned empty choices for model={self.model!r}. "
                f"Response: {response!r}"
            )
        raw = response.choices[0].message.content
        if isinstance(raw, list):
            raw = "\n".join(
                str(item.get("text", ""))
                for item in raw
                if isinstance(item, dict) and item.get("type") == "text"
            )
        return str(raw).strip()

    @backoff.on_exception(
        backoff.expo,
        (openai.APIConnectionError, openai.InternalServerError, openai.RateLimitError),
        max_tries=None,
        on_backoff=lambda d: d["args"][0].log_warning(
            f"[her] API error on attempt {d['tries']}, retrying: {d['exception']}"
        ),
    )
    def call_n(
        self,
        frame_tensors: list,
        prompt_text: str,
        max_tokens: int,
        n: int,
        temperature: float,
    ) -> list[str]:
        """Send frames + prompt and return *n* sampled completions."""
        frame_tensors = select_frames_for_vlm(frame_tensors, self.max_frames)
        content = self._build_message_content(frame_tensors, prompt_text)
        extra_body = self._build_extra_body()

        client = self._make_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            n=n,
            temperature=temperature,
            **({"extra_body": extra_body} if extra_body else {}),
        )
        if not response.choices:
            raise RuntimeError(
                f"VLM returned empty choices for model={self.model!r}. "
                f"Response: {response!r}"
            )
        results: list[str] = []
        for choice in response.choices:
            raw = choice.message.content
            if isinstance(raw, list):
                raw = "\n".join(
                    str(item.get("text", ""))
                    for item in raw
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            results.append(str(raw).strip())
        return results

    # ── Prompt building ───────────────────────────────────────────────────────

    def _build_relabel_prompt(
        self,
        original_instruction: str,
        scene_objects: Optional[list],
        prompt_version: str,
        n_vlm_frames: Optional[int] = None,
    ) -> str:
        """Select and format the HER relabeling prompt template."""
        scene_text = ", ".join(scene_objects) if scene_objects else "not specified"
        scene_line = (
            f"Objects present in the scene: {', '.join(scene_objects)}\n"
            if scene_objects
            else ""
        )
        if prompt_version == "subtraj_v2":
            if n_vlm_frames is None:
                raise ValueError("n_vlm_frames is required for prompt_version='subtraj_v2'")
            return build_subtraj_prompt_v2(original_instruction, n_vlm_frames, scene_objects)
        if prompt_version == "subtraj":
            return HER_PROMPT_TEMPLATE_SUBTRAJ.format(
                original_instruction=original_instruction,
                scene_objects_line=scene_line,
            )
        if prompt_version == "v12":
            return HER_PROMPT_TEMPLATE_V12.format(
                original_instruction=original_instruction,
                scene_objects_text=scene_text,
            )
        if prompt_version == "v11_strict":
            return HER_PROMPT_TEMPLATE_V11_STRICT.format(
                original_instruction=original_instruction,
                scene_objects_text=scene_text,
            )
        if prompt_version == "v11":
            return HER_PROMPT_TEMPLATE_V11.format(
                original_instruction=original_instruction,
                scene_objects_text=scene_text,
            )
        if prompt_version == "v10":
            return HER_PROMPT_TEMPLATE_V10.format(
                original_instruction=original_instruction,
                scene_objects_text=scene_text,
            )
        if prompt_version == "v7_no_uninteresting":
            return HER_PROMPT_TEMPLATE_V7_NO_UNINTERESTING.format(
                original_instruction=original_instruction,
                scene_objects_text=scene_text,
            )
        if prompt_version == "v7_no_objects":
            return HER_PROMPT_TEMPLATE_V7_NO_OBJECTS.format(
                original_instruction=original_instruction,
            )
        if prompt_version == "v7":
            return HER_PROMPT_TEMPLATE_V7.format(
                original_instruction=original_instruction,
                scene_objects_text=scene_text,
            )
        if prompt_version == "v6":
            return HER_PROMPT_TEMPLATE_V6.format(
                original_instruction=original_instruction,
            )
        if prompt_version == "v5":
            return HER_PROMPT_TEMPLATE_V5.format(
                original_instruction=original_instruction,
                scene_objects_text=scene_text,
            )
        if prompt_version == "v4":
            return HER_PROMPT_TEMPLATE_V4.format(
                original_instruction=original_instruction,
                scene_objects_text=scene_text,
            )
        if prompt_version == "v3":
            return HER_PROMPT_TEMPLATE_V3.format(
                original_instruction=original_instruction,
            )
        if prompt_version == "v2":
            return HER_PROMPT_TEMPLATE_V2.format(
                original_instruction=original_instruction,
            )
        # v1 / default
        return HER_PROMPT_TEMPLATE.format(
            original_instruction=original_instruction,
        )

    # ── High-level requests ───────────────────────────────────────────────────

    def request_hindsight_instruction(
        self,
        frame_tensors: list,
        original_instruction: str,
        scene_objects: Optional[list] = None,
        prompt_version: str = "v1",
        _raw_out: Optional[dict] = None,
    ) -> tuple:
        """Query the VLM for a hindsight instruction describing what the robot did.

        Retries until the response contains a ``<final>`` tag with a valid
        instruction (≤20 words, or the special token "Nothing").

        Returns:
            (instruction, cutoff_fraction) — ``cutoff_fraction`` is a float in
            [0, 1] only for ``prompt_version="subtraj"`` / ``"subtraj_v2"``
            Form B responses; ``None`` otherwise.
        """
        is_subtraj_v2 = prompt_version == "subtraj_v2"
        is_subtraj = prompt_version == "subtraj" or is_subtraj_v2

        # For subtraj_v2 the prompt embeds the VLM frame count; compute it once
        # from the full frame list before call() subsamples it.
        if is_subtraj_v2:
            vlm_indices = select_frame_indices_for_vlm(len(frame_tensors), self.max_frames)
            n_vlm_frames = len(vlm_indices)
            total_frames = len(frame_tensors)
        else:
            vlm_indices = None
            n_vlm_frames = None
            total_frames = None

        prompt_text = self._build_relabel_prompt(
            original_instruction or "unknown", scene_objects, prompt_version,
            n_vlm_frames=n_vlm_frames,
        )
        if _raw_out is not None:
            _raw_out["prompt_text"] = prompt_text

        max_tokens = 8192 if is_subtraj else 1024
        attempt = 0
        while True:
            attempt += 1
            try:
                raw = self.call(frame_tensors, prompt_text, max_tokens=max_tokens)
            except RuntimeError as e:
                self.log_warning(
                    f"[her] attempt {attempt}: VLM call failed ({e}), retrying"
                )
                continue
            except openai.APIStatusError as e:
                if isinstance(e, (openai.RateLimitError, openai.InternalServerError)):
                    raise
                raise RuntimeError(
                    f"HER VLM request failed: "
                    f"{type(e).__name__}(status={e.status_code}): {e.message}"
                ) from None

            if _raw_out is not None:
                _raw_out["response"] = raw

            # subtraj_v2: parse <cutoff_frame> integer via parse_subtraj_response.
            if is_subtraj_v2:
                assert vlm_indices is not None and total_frames is not None
                instruction, cutoff_fraction = parse_subtraj_response(
                    raw, vlm_indices, total_frames
                )
                if instruction is None:
                    self.log_warning(
                        f"[her] attempt {attempt}: subtraj_v2 parse failed, retrying"
                    )
                    continue
                return instruction, cutoff_fraction

            matched = re.search(r"<final>\s*(.*?)\s*</final>", raw, flags=re.DOTALL)
            if not matched:
                self.log_warning(
                    f"[her] attempt {attempt}: no <final> tag in response, retrying"
                )
                continue
            instruction = matched.group(1).strip()
            if instruction:
                instruction = instruction[0].upper() + instruction[1:]
            if instruction.lower() != "nothing" and len(instruction.split()) > 20:
                self.log_warning(
                    f"[her] attempt {attempt}: instruction too long "
                    f"({len(instruction.split())} words), retrying"
                )
                continue

            if is_subtraj:
                cutoff_fraction: Optional[float] = None
                m = re.search(
                    r"<cutoff_fraction>\s*([\d.]+)\s*</cutoff_fraction>", raw
                )
                if m:
                    try:
                        cutoff_fraction = max(0.0, min(1.0, float(m.group(1))))
                    except ValueError:
                        pass
                if instruction.lower() != "nothing" and cutoff_fraction is None:
                    self.log_warning(
                        f"[her] attempt {attempt}: subtraj response has instruction "
                        f"but no <cutoff_fraction> tag, retrying"
                    )
                    continue
                return instruction, cutoff_fraction

            return instruction, None

    @backoff.on_exception(
        backoff.expo,
        (openai.APIConnectionError, openai.InternalServerError, openai.RateLimitError),
        max_tries=None,
        on_backoff=lambda d: d["args"][0].log_warning(
            f"[her] rephrase API error on attempt {d['tries']}, retrying: {d['exception']}"
        ),
    )
    def request_rephrase_instruction(
        self,
        original_instruction: str,
        temperature: float = 0.8,
    ) -> str:
        """Ask the VLM to rephrase an instruction (text-only, no video)."""
        prompt_text = HER_REPHRASE_PROMPT_TEMPLATE.format(
            original_instruction=original_instruction,
        )
        client = self._make_client()
        extra_body = self._build_extra_body()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=1024,
            temperature=temperature,
            **({"extra_body": extra_body} if extra_body else {}),
        )
        if not response.choices:
            raise RuntimeError("VLM returned empty choices for rephrase request.")
        raw = response.choices[0].message.content
        if isinstance(raw, list):
            raw = "\n".join(
                str(item.get("text", ""))
                for item in raw
                if isinstance(item, dict) and item.get("type") == "text"
            )
        raw = str(raw).strip()
        matched = re.search(r"<final>\s*(.*?)\s*</final>", raw, flags=re.DOTALL)
        if not matched:
            raise RuntimeError(
                f"Rephrase response missing <final> tag: {raw[:200]}"
            )
        instruction = matched.group(1).strip()
        if instruction:
            instruction = instruction[0].upper() + instruction[1:]
        return instruction

    def _reward_eval_config(self) -> tuple[str, re.Pattern, int]:
        """Return (prompt_text_template, answer_pattern, max_tokens) for the
        current ``reward_eval_prompt_version``."""
        template = {
            2: HER_REWARD_EVAL_PROMPT_TEMPLATE_V2,
            3: HER_REWARD_EVAL_PROMPT_TEMPLATE_V3,
            4: HER_REWARD_EVAL_PROMPT_TEMPLATE_V4,
            6: HER_REWARD_EVAL_PROMPT_TEMPLATE_V6,
            7: HER_REWARD_EVAL_PROMPT_TEMPLATE_V7,
        }.get(self.reward_eval_prompt_version, HER_REWARD_EVAL_PROMPT_TEMPLATE)
        is_v6 = self.reward_eval_prompt_version == 6
        pattern = re.compile(
            r"<answer>\s*(yes|no|unsure)\s*</answer>"
            if is_v6
            else r"<answer>\s*(yes|no)\s*</answer>",
            flags=re.DOTALL,
        )
        max_tokens = {3: 2048, 4: 16384, 6: 16384, 7: 16384}.get(
            self.reward_eval_prompt_version, 512
        )
        return template, pattern, max_tokens

    def request_reward_eval(
        self,
        frame_tensors: list,
        instruction: str,
        num_samples: int = 1,
        vote_temperature: float = 0.6,
    ) -> tuple[float, Optional[float]]:
        """Query the VLM to check whether the robot completed the instruction.

        When ``num_samples == 1`` (default), makes a single deterministic call
        and retries until the response contains a parseable ``<answer>`` tag.

        When ``num_samples > 1``, uses majority voting: samples *n*
        completions at ``vote_temperature``, parses each, and returns the
        answer with the most votes.  If some samples fail to parse, only the
        remaining needed samples are re-requested.

        Returns:
            (reward, cutoff_fraction) where reward is 1.0 (yes), 0.5 (unsure,
            v6 only), or 0.0 (no).  cutoff_fraction is only set for prompt
            version 3 in single-sample mode.
        """
        template, answer_pattern, max_tokens = self._reward_eval_config()
        prompt_text = template.format(instruction=instruction)

        if num_samples <= 1:
            return self._reward_eval_single(
                frame_tensors, prompt_text, answer_pattern, max_tokens
            )
        return self._reward_eval_majority(
            frame_tensors,
            prompt_text,
            answer_pattern,
            max_tokens,
            num_samples=num_samples,
            temperature=vote_temperature,
        )

    def _reward_eval_single(
        self,
        frame_tensors: list,
        prompt_text: str,
        answer_pattern: re.Pattern,
        max_tokens: int,
    ) -> tuple[float, Optional[float]]:
        """Single-sample reward eval (original behaviour)."""
        is_v3 = self.reward_eval_prompt_version == 3
        attempt = 0
        while True:
            attempt += 1
            try:
                raw = self.call(frame_tensors, prompt_text, max_tokens=max_tokens)
            except RuntimeError as e:
                self.log_warning(
                    f"[her_eval] attempt {attempt}: VLM call failed ({e}), retrying"
                )
                continue
            except openai.APIStatusError as e:
                if isinstance(e, (openai.RateLimitError, openai.InternalServerError)):
                    raise
                raise RuntimeError(
                    f"VLM reward eval request failed: "
                    f"{type(e).__name__}(status={e.status_code}): {e.message}"
                ) from None

            matched = answer_pattern.search(raw.lower())
            if not matched:
                self.log_warning(
                    f"[her_eval] attempt {attempt}: no <answer> tag in response, retrying"
                )
                continue
            answer = matched.group(1).strip()
            if answer == "unsure":
                reward = 0.5
            else:
                reward = 1.0 if answer == "yes" else 0.0

            if not is_v3 or reward == 1.0:
                return reward, None

            cutoff_fraction: Optional[float] = None
            m = re.search(r"<cutoff_fraction>\s*([\d.]+)\s*</cutoff_fraction>", raw)
            if m:
                try:
                    cutoff_fraction = max(0.0, min(1.0, float(m.group(1))))
                except ValueError:
                    pass
            return reward, cutoff_fraction

    def _reward_eval_majority(
        self,
        frame_tensors: list,
        prompt_text: str,
        answer_pattern: re.Pattern,
        max_tokens: int,
        num_samples: int,
        temperature: float,
    ) -> tuple[float, Optional[float]]:
        """Majority-voting reward eval: sample *num_samples* completions and
        return the most frequent answer.

        Retries indefinitely (matching ``_reward_eval_single`` behaviour) until
        all *num_samples* valid answers are collected.  Each round requests only
        the remaining needed completions via ``call_n``.
        """
        valid_answers: list[str] = []
        round_idx = 0
        while len(valid_answers) < num_samples:
            round_idx += 1
            remaining = num_samples - len(valid_answers)
            try:
                responses = self.call_n(
                    frame_tensors,
                    prompt_text,
                    max_tokens=max_tokens,
                    n=remaining,
                    temperature=temperature,
                )
            except RuntimeError as e:
                self.log_warning(
                    f"[her_eval_mv] round {round_idx}: VLM call failed ({e}), retrying"
                )
                continue
            except openai.APIStatusError as e:
                if isinstance(e, (openai.RateLimitError, openai.InternalServerError)):
                    raise
                raise RuntimeError(
                    f"VLM reward eval (majority) request failed: "
                    f"{type(e).__name__}(status={e.status_code}): {e.message}"
                ) from None

            for raw in responses:
                matched = answer_pattern.search(raw.lower())
                if matched:
                    valid_answers.append(matched.group(1).strip())

            if len(valid_answers) < num_samples:
                self.log_warning(
                    f"[her_eval_mv] round {round_idx}: {len(valid_answers)}/{num_samples} "
                    f"valid answers, retrying remaining {num_samples - len(valid_answers)}"
                )

        counts: dict[str, int] = {}
        for v in valid_answers:
            counts[v] = counts.get(v, 0) + 1
        max_count = max(counts.values())
        tied = [k for k, c in counts.items() if c == max_count]
        winner = random.choice(tied)
        reward = {"yes": 1.0, "unsure": 0.5, "no": 0.0}[winner]
        print(
            f"[her_eval_mv] votes={valid_answers} counts={counts} "
            f"winner={winner} reward={reward}"
        )
        return reward, None
