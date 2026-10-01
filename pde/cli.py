"""Small release utilities for prompt-pool artifacts."""

from __future__ import annotations

import argparse
import json

from pde.prompt_pool import load_prompt_pool


def main() -> None:
    """Validate and summarize a prompt-pool JSON file."""
    parser = argparse.ArgumentParser(prog="pde")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-pool")
    validate.add_argument("path")
    args = parser.parse_args()

    pool = load_prompt_pool(args.path)
    print(
        json.dumps(
            {
                "task_id": pool.task_id,
                "candidates": len(pool.candidates),
                "admitted": len(pool.admitted),
                "rollouts": sum(item.rollouts for item in pool.candidates),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
