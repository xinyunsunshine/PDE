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

"""MI reward via InfoNCE: rewards trajectories identifiable by their instruction.

Given N rollouts {(τ_i, l_i)}, compute:
    r[i] = sim(l_i, τ_i) - log Σ_j exp(sim(l_j, τ_i))

sim(l, τ) is computed by prompting Qwen3-VL (thinking) with the full video
and all M candidate instructions simultaneously, asking it to score each one
from 1 to 10.  One API call per rollout covers the entire score matrix row.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import backoff
import numpy as np

# Max concurrent VLM API calls (one per rollout).
_CONCURRENCY = 16


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SCORE_PROMPT_TEMPLATE = """\
This is a video of a robot manipulation trajectory.

Below are {n} candidate instructions. For each one, output an integer score \
from 0 to 5 indicating how well the robot follows that instruction:
0 = the trajectory clearly does NOT follow the instruction
5 = the trajectory clearly DOES follow the instruction

Instructions:
{instruction_list}

Think carefully, then output a JSON array of {n} integers in the same order \
as the instructions above. Example for 4 instructions: [0, 5, 1, 2]
Output only the JSON array, nothing else."""


def _build_prompt(instructions: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {instr}" for i, instr in enumerate(instructions))
    return _SCORE_PROMPT_TEMPLATE.format(n=len(instructions), instruction_list=numbered)


# ---------------------------------------------------------------------------
# VLM scoring
# ---------------------------------------------------------------------------


def score_video_all_instructions(
    client: "openai.OpenAI",
    video_b64: str,
    instructions: list[str],
    model: str,
    thinking_budget: int = 512,
) -> list[float]:
    """Score a trajectory video against all instructions in a single API call.

    Sends the MP4 video + all candidate instructions to the thinking model and
    asks it to rank the instructions from best to worst match.

    Args:
        client: OpenAI client pointed at the vLLM server.
        video_b64: Base64-encoded MP4 video of the trajectory.
        instructions: List of M instruction strings to score.
        model: Model name as registered in vLLM.
        thinking_budget: Max tokens for the thinking chain.

    Returns:
        List of M floats (higher = trajectory matches better).
    """
    n = len(instructions)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
                },
                {
                    "type": "text",
                    "text": _build_prompt(instructions),
                },
            ],
        }
    ]

    @backoff.on_exception(
        backoff.expo,
        Exception,
        on_backoff=lambda details: print(
            f"[mi_reward] API call failed (attempt {details['tries']}), "
            f"retrying in {details['wait']:.1f}s: "
            f"{type(details['exception']).__name__}: {details['exception']}"
        ),
    )
    def _call_api():
        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=thinking_budget + 4096,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "thinking_budget": thinking_budget,
                },
            },
        )

    @backoff.on_exception(
        backoff.expo,
        (ValueError, RuntimeError),
        on_backoff=lambda details: print(
            f"[mi_reward] parse failed (attempt {details['tries']}), "
            f"retrying in {details['wait']:.1f}s: "
            f"{type(details['exception']).__name__}: {details['exception']}"
        ),
    )
    def _call_and_parse():
        response = _call_api()
        msg = response.choices[0].message
        raw = msg.content or ""
        thinking = getattr(msg, "reasoning_content", None) or ""

        print(
            f"[mi_reward] --- thinking ---\n{thinking}\n"
            f"[mi_reward] --- answer ---\n{raw}\n"
            f"[mi_reward] ---"
        )

        match = re.search(r"\[[\d,\s]+\]", raw)
        if not match:
            raise ValueError(
                f"Could not parse JSON score array from response: {repr(raw[:300])}"
            )
        scores = json.loads(match.group())
        if len(scores) != n:
            raise ValueError(
                f"Expected {n} scores, got {len(scores)}: {scores}"
            )
        return [float(max(0, min(5, s))) for s in scores]

    return _call_and_parse()


# ---------------------------------------------------------------------------
# InfoNCE reward
# ---------------------------------------------------------------------------


def infonce_rewards(
    score_matrix: np.ndarray,
    intended: list[int],
    temperature: float = 1.0,
) -> np.ndarray:
    """Compute InfoNCE rewards from a score matrix.

    Args:
        score_matrix: (N, M) array where S[i, j] = sim(instructions[j], rollouts[i]).
        intended: Length-N list where intended[i] is the instruction index for rollout i.
        temperature: Softmax temperature.

    Returns:
        rewards: (N,) array of InfoNCE rewards.
    """
    S = score_matrix / temperature
    rewards = np.zeros(len(intended), dtype=np.float32)
    for i, j in enumerate(intended):
        row = S[i]
        max_s = np.max(row)
        # Numerically stable log-sum-exp.
        lse = max_s + np.log(np.sum(np.exp(row - max_s)))
        rewards[i] = S[i, j] - lse
    return rewards


# ---------------------------------------------------------------------------
# MIReward: main class
# ---------------------------------------------------------------------------


class MIReward:
    """Mutual Information reward via InfoNCE with VLM scoring.

    Makes one API call per rollout: the full video + all candidate instructions
    are sent together, and the model scores all instructions in a single pass.
    """

    def __init__(
        self,
        vllm_url: str = "http://localhost:8000",
        model: str = "qwen3-vl",
        temperature: float = 1.0,
        concurrency: int = _CONCURRENCY,
        backend: str = "vllm",
        api_key: str = "",
        thinking_budget: int = 512,
    ):
        self.vllm_url = vllm_url
        self.model = model
        self.temperature = temperature
        self.concurrency = concurrency
        self.backend = backend
        self.api_key = api_key
        self.thinking_budget = thinking_budget

    def _make_client(self) -> "openai.OpenAI":
        import openai

        if self.backend == "rits":
            return openai.OpenAI(
                api_key=self.api_key,
                base_url=self.vllm_url,
                default_headers={"RITS_API_KEY": self.api_key},
                timeout=120,
            )
        return openai.OpenAI(
            api_key=self.api_key or "EMPTY",
            base_url=self.vllm_url,
            timeout=120,
        )

    def _score_rollout(
        self,
        client: "openai.OpenAI",
        video_b64: str,
        instructions: list[str],
    ) -> list[float]:
        """Score one rollout against all instructions."""
        return score_video_all_instructions(
            client, video_b64, instructions, self.model, self.thinking_budget
        )

    def compute_rewards_sync(
        self,
        videos_b64: list[str],
        instructions_per_rollout: list[str],
    ) -> np.ndarray:
        """Compute InfoNCE rewards for a batch of GRPO rollouts.

        Accepts one instruction string per rollout (repeats allowed — GRPO
        groups share the same instruction).  Deduplicates internally so each
        unique instruction appears exactly once in the negative set seen by
        every rollout.  Makes one API call per rollout.

        Args:
            videos_b64: N base64-encoded MP4 videos, one per rollout.
            instructions_per_rollout: Length-N list of instruction strings,
                one per rollout.  Rollouts in the same GRPO group share the
                same string.

        Returns:
            rewards: (N,) InfoNCE reward array.
        """
        from tqdm import tqdm

        N = len(videos_b64)

        # Deduplicate instructions, preserving first-seen order.
        seen: dict[str, int] = {}
        unique_instructions: list[str] = []
        for instr in instructions_per_rollout:
            if instr not in seen:
                seen[instr] = len(unique_instructions)
                unique_instructions.append(instr)
        intended = [seen[instr] for instr in instructions_per_rollout]
        M = len(unique_instructions)

        print(
            f"[mi_reward] N={N} rollouts, M={M} unique instructions "
            f"(group_size={N // M})"
        )
        for idx, instr in enumerate(unique_instructions):
            print(f"[mi_reward]   [{idx}] {instr}")

        score_matrix = np.zeros((N, M), dtype=np.float32)
        client = self._make_client()

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            future_to_idx = {
                pool.submit(self._score_rollout, client, videos_b64[i], unique_instructions): i
                for i in range(N)
            }
            with tqdm(
                total=N,
                desc=f"[mi_reward] scoring {N} rollouts",
                leave=False,
            ) as pbar:
                for future in as_completed(future_to_idx):
                    i = future_to_idx[future]
                    score_matrix[i] = future.result()
                    pbar.update(1)

        score_matrix = np.where(np.isfinite(score_matrix), score_matrix, 0.0)
        return infonce_rewards(score_matrix, intended, temperature=self.temperature)

