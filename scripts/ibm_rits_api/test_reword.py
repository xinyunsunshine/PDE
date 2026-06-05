"""Test rewording via RITS (Qwen2.5-72B-Instruct).

Usage:
    python scripts/ibm_rits_api/test_reword.py [instruction] [--variant 1|2] [--n N]

Defaults: instruction="put hamburger on plate", variant=2, n=3
Requires RITS_API_KEY env var.
"""

import argparse
import os
import re

from openai import OpenAI

RITS_BASE_URL = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/qwen2-5-72b-instruct/v1"
RITS_MODEL = "Qwen/Qwen2.5-72B-Instruct"

REWORDING_PROMPT_TEMPLATE = (
    "Rephrase the following robot instruction using different words and "
    "sentence structure while preserving its meaning. Be creative with "
    "the wording — do not reuse the same phrases. Keep it under 20 words.\n\n"
    "Original instruction: {instruction}\n\n"
    "Output format must be exactly:\n"
    "your reasoning\n"
    "<final>one concise instruction</final>"
)

REWORDING_PROMPT_TEMPLATE_2 = (
    "You are given a robot instruction. Your task is to rewrite it so that it "
    "uses completely different words and phrasing while keeping the same meaning.\n\n"
    "Rules:\n"
    "- You MUST NOT reproduce the original instruction verbatim.\n"
    "- You MUST NOT reuse the key action verbs or object names from the original "
    "— replace them with synonyms or alternative descriptions.\n"
    "- The rewritten instruction must be clearly different in wording from the original.\n"
    "- Keep it concise (under 20 words).\n\n"
    "Original instruction: {instruction}\n\n"
    "First, identify the key words in the original and list their synonyms. "
    "Then write the reworded instruction using those synonyms.\n\n"
    "Output format must be exactly:\n"
    "your reasoning\n"
    "<final>one concise instruction</final>"
)

TEMPLATES = {1: REWORDING_PROMPT_TEMPLATE, 2: REWORDING_PROMPT_TEMPLATE_2}


def parse_instruction(raw: str) -> str:
    matched = re.search(r"<final>\s*(.*?)\s*</final>", raw, flags=re.DOTALL)
    instruction = matched.group(1).strip() if matched else raw.strip()
    if len(instruction.split()) > 20:
        instruction = " ".join(instruction.split()[:20])
    return instruction


def reword(client: OpenAI, instruction: str, variant: int = 2) -> tuple[str, str]:
    """Returns (raw_response, parsed_instruction)."""
    template = TEMPLATES[variant]
    response = client.chat.completions.create(
        model=RITS_MODEL,
        messages=[{"role": "user", "content": template.format(instruction=instruction)}],
        max_tokens=256,
    )
    raw = str(response.choices[0].message.content).strip()
    return raw, parse_instruction(raw)


def main():
    parser = argparse.ArgumentParser(description="Test rewording via RITS")
    parser.add_argument("instruction", nargs="?", default="put hamburger on plate")
    parser.add_argument("--variant", type=int, choices=[1, 2], default=2)
    parser.add_argument("--n", type=int, default=3, help="Number of rewording attempts")
    args = parser.parse_args()

    api_key = os.environ["RITS_API_KEY"]
    client = OpenAI(
        api_key=api_key,
        base_url=RITS_BASE_URL,
        default_headers={"RITS_API_KEY": api_key},
        timeout=120,
    )

    print(f"Model:       {RITS_MODEL}")
    print(f"Endpoint:    {RITS_BASE_URL}")
    print(f"Variant:     {args.variant}")
    print(f"Instruction: {args.instruction}")
    print(f"Attempts:    {args.n}")

    for i in range(args.n):
        print(f"\n{'='*40}")
        print(f"Attempt {i+1}/{args.n}")
        print("=" * 40)
        raw, parsed = reword(client, args.instruction, args.variant)
        print(f"Raw:\n{raw}")
        print(f"\nParsed: {parsed}")
        same = parsed.lower() == args.instruction.lower()
        print(f"Different from original: {'NO' if same else 'yes'}")


if __name__ == "__main__":
    main()
