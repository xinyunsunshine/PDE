"""Sanity check for the MI reward module.

Runs two sub-tests against the same trajectory:

  Test A — semantic distractors
    Score correct instruction vs. 10 unrelated distractors.
    Pass: correct instruction ranks #1.

  Test B — paraphrase denominator (the hard case)
    Score correct instruction vs. 5 paraphrases of itself + 5 unrelated
    distractors.  The denominator now contains instructions that are
    semantically near-identical to the correct one, so the model must
    distinguish the *exact* phrasing from re-wordings that still describe
    the same action.  Pass: correct instruction ranks #1.

Usage:
    # vLLM backend (default):
    python scripts/test_mi.py [video_path] [correct_instruction] \
        [--url http://host:8000/v1] [--model qwen3-vl]

    # RITS backend:
    RITS_API_KEY=<key> python scripts/test_mi.py [video_path] [correct_instruction] \
        --backend rits \
        --url https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/qwen3-vl-235b-a22b-instruct/v1 \
        --model Qwen/Qwen3-VL-235B-A22B-Instruct

Defaults:
    video_path:          scripts/ibm_rits_api/her_video.mp4
    correct_instruction: put hamburger on plate
    backend:             vllm
    url:                 http://VLM_ENDPOINT_HOST:PORT/v1
    model:               Qwen/Qwen3-VL-235B-A22B-Thinking-FP8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root without installing the package.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from rlinf.algorithms.rewards.mi import (
    MIReward,
    infonce_rewards,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_VLLM_URL = "http://VLM_ENDPOINT_HOST:PORT/v1"
_DEFAULT_VLLM_MODEL = "Qwen/Qwen3-VL-235B-A22B-Thinking-FP8"
_DEFAULT_RITS_URL = (
    "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com"
    "/qwen3-vl-235b-a22b-thinking/v1"
)
_DEFAULT_RITS_MODEL = "Qwen/Qwen3-VL-235B-A22B-Thinking"

# ---------------------------------------------------------------------------
# Candidate instructions
# ---------------------------------------------------------------------------

# Test A: unrelated distractors (easy — different objects and actions).
_DISTRACTOR_INSTRUCTIONS = [
    "pick up the cream cheese and place it in the basket",
    "pick up the ketchup and place it in the basket",
    "pick up the tomato sauce and place it in the bowl",
    "pick up the butter and place it on the plate",
    "open the top drawer of the wooden cabinet",
    "push the plate to the front of the stove",
    "stack the bowls together",
    "close the bottom drawer of the wooden cabinet",
    "put the wine bottle in the wine rack",
    "turn on the stove",
]

# Test B paraphrases: same meaning as the correct instruction, different words.
_PARAPHRASE_INSTRUCTIONS: dict[str, list[str]] = {
    "pick up the tomato sauce and place it in the basket": [
        "grasp the tomato sauce and put it in the basket",
        "move the tomato sauce into the basket",
        "transfer the tomato sauce to the basket",
        "place the tomato sauce bottle in the basket",
        "take the tomato sauce and drop it in the basket",
    ],
}


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------# ---------------------------------------------------------------------------
# Sub-test runner
# ---------------------------------------------------------------------------


def _run_subtest(
    label: str,
    video_b64: str,
    correct_instruction: str,
    pool: list[str],
    mi: MIReward,
    note: str = "",
) -> tuple[bool, float]:
    """Score correct_instruction against pool and report results.

    Returns (passed, infonce_reward).
    """
    all_instructions = [correct_instruction] + pool

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    if note:
        print(f"  {note}")
    print(f"{'=' * 60}")
    print(f"  correct: {correct_instruction}")
    print(f"  pool size: {len(pool)}")

    client = mi._make_client()
    scores = mi._score_rollout(client, video_b64, all_instructions)

    ranked = sorted(enumerate(zip(all_instructions, scores)), key=lambda x: -x[1][1])

    print()
    print(f"  {'Rank':<5} {'Score':>6}  Instruction")
    print(f"  {'-' * 56}")
    for rank, (orig_idx, (instr, sc)) in enumerate(ranked, 1):
        marker = " <-- CORRECT" if orig_idx == 0 else ""
        print(f"  {rank:<5} {sc:>6.2f}  {instr}{marker}")

    correct_rank = next(rank for rank, (idx, _) in enumerate(ranked, 1) if idx == 0)

    score_matrix = np.array([scores], dtype=np.float32)
    reward = infonce_rewards(score_matrix, intended=[0])[0]

    print()
    print(f"  Correct rank: {correct_rank} / {len(all_instructions)}")
    print(f"  Correct score: {scores[0]:.2f}")
    print(f"  InfoNCE reward (temp=1.0): {reward:.4f}")

    passed = correct_rank == 1
    status = "PASS" if passed else f"FAIL (ranked #{correct_rank})"
    print(f"  -> {status}")
    return passed, float(reward)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="MI reward sanity check")
    parser.add_argument("video_path", nargs="?", default="scripts/ibm_rits_api/libero_video.mp4")
    parser.add_argument("correct_instruction", nargs="?", default="pick up the tomato sauce and place it in the basket")
    parser.add_argument("--backend", choices=["vllm", "rits"], default="vllm")
    parser.add_argument("--url", default=None, help="API base URL (overrides default for backend)")
    parser.add_argument("--model", default=None, help="Model name (overrides default for backend)")
    args = parser.parse_args()

    # Resolve defaults by backend.
    if args.backend == "rits":
        url = args.url or _DEFAULT_RITS_URL
        model = args.model or _DEFAULT_RITS_MODEL
        api_key = os.environ.get("RITS_API_KEY", "")
        if not api_key:
            print("ERROR: RITS_API_KEY environment variable not set.")
            sys.exit(1)
    else:
        url = args.url or _DEFAULT_VLLM_URL
        model = args.model or _DEFAULT_VLLM_MODEL
        api_key = ""

    if not Path(args.video_path).exists():
        print(f"ERROR: Video file not found: {args.video_path}")
        sys.exit(1)

    mi = MIReward(
        vllm_url=url,
        model=model,
        backend=args.backend,
        api_key=api_key,
    )

    print("=" * 60)
    print("MI Reward Sanity Check")
    print("=" * 60)
    print(f"Video:   {args.video_path}")
    print(f"Backend: {args.backend}")
    print(f"URL:     {url}")
    print(f"Model:   {model}")

    import base64
    print("\nLoading video...")
    with open(args.video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")
    print(f"  -> {len(video_b64) // 1024} KB encoded.")

    paraphrases = _PARAPHRASE_INSTRUCTIONS.get(args.correct_instruction, [])
    results: list[tuple[str, bool, float]] = []

    # --- Test A: semantic distractors ---
    passed_a, reward_a = _run_subtest(
        label="Test A — semantic distractors",
        video_b64=video_b64,
        correct_instruction=args.correct_instruction,
        pool=_DISTRACTOR_INSTRUCTIONS,
        mi=mi,
        note="Pool: 10 unrelated instructions.",
    )
    results.append(("A", passed_a, reward_a))

    # --- Test B: paraphrases in the denominator ---
    if paraphrases:
        mixed_pool = paraphrases + _DISTRACTOR_INSTRUCTIONS[:5]
        passed_b, reward_b = _run_subtest(
            label="Test B — paraphrases in denominator",
            video_b64=video_b64,
            correct_instruction=args.correct_instruction,
            pool=mixed_pool,
            mi=mi,
            note=(
                f"Pool: {len(paraphrases)} paraphrases of the correct instruction "
                f"+ {len(_DISTRACTOR_INSTRUCTIONS[:5])} unrelated distractors.\n"
                "  Expected: InfoNCE reward lower than Test A (paraphrases compete in denominator)."
            ),
        )
        results.append(("B", passed_b, reward_b))

        reward_drops = reward_b < reward_a
        print(
            f"\n  Reward comparison: A={reward_a:.4f}  B={reward_b:.4f}  "
            f"{'(B < A as expected)' if reward_drops else '(B >= A — unexpected)'}"
        )

    # --- Test C: duplicated instructions (GRPO simulation) ---
    # Simulate a GRPO batch: 4 rollouts sharing the correct instruction,
    # 4 rollouts sharing the first distractor.  Deduplication must reduce
    # this to 2 unique instructions and produce correct intended mapping.
    group_size = 4
    instructions_per_rollout = (
        [args.correct_instruction] * group_size
        + [_DISTRACTOR_INSTRUCTIONS[0]] * group_size
    )
    videos_per_rollout = [video_b64] * (group_size * 2)

    print(f"\n{'=' * 60}")
    print("  Test C — duplicated instructions (GRPO simulation)")
    print(f"{'=' * 60}")
    print(f"  {group_size} rollouts x correct + {group_size} rollouts x distractor")
    print(f"  instructions_per_rollout: {instructions_per_rollout}")

    rewards_c = mi.compute_rewards_sync(videos_per_rollout, instructions_per_rollout)

    print(f"\n  rewards (rollouts 0-{group_size-1} = correct group): "
          f"{[f'{r:.3f}' for r in rewards_c[:group_size]]}")
    print(f"  rewards (rollouts {group_size}-{group_size*2-1} = distractor group): "
          f"{[f'{r:.3f}' for r in rewards_c[group_size:]]}")

    # Correct group should have higher mean reward than distractor group.
    mean_correct = float(np.mean(rewards_c[:group_size]))
    mean_distractor = float(np.mean(rewards_c[group_size:]))
    passed_c = mean_correct > mean_distractor
    print(f"\n  mean correct={mean_correct:.4f}  mean distractor={mean_distractor:.4f}")
    print(f"  -> {'PASS' if passed_c else 'FAIL (distractor group scored higher)'}")
    results.append(("C", passed_c, mean_correct))

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    all_passed = True
    for name, passed, reward in results:
        status = "PASS" if passed else "FAIL"
        print(f"  Test {name}: {status}  InfoNCE={reward:.4f}")
        all_passed = all_passed and passed

    print()
    if all_passed:
        print("ALL TESTS PASSED.")
    else:
        print("SOME TESTS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
