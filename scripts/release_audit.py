"""Fail on common private artifacts before publishing the repository."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

PRIVATE_PATTERNS = {
    "absolute project path": re.compile(r"/(?:proj|gpfs)/[^\s\"']+"),
    "cluster hostname": re.compile(r"\bp\d+-r\d+-n\d+\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\bgh[ps]_[A-Za-z0-9]{30,}\b"),
    "OpenAI token": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "unresolved repository owner": re.compile("_".join(("REPLACE", "WITH", "OWNER"))),
}

TEXT_SUFFIXES = {
    ".bib",
    ".cfg",
    ".cff",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def tracked_files(root: Path) -> list[Path]:
    """Return paths tracked by the repository at ``root``."""
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [root / item.decode() for item in result.stdout.split(b"\0") if item]


def main() -> int:
    """Scan tracked text files and return nonzero when findings exist."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    paths = tracked_files(root)
    findings = []
    for path in paths:
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        try:
            content = path.read_text(errors="strict")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(content.splitlines(), start=1):
            for label, pattern in PRIVATE_PATTERNS.items():
                if pattern.search(line):
                    findings.append(f"{path.relative_to(root)}:{line_number}: {label}")
    if findings:
        print("Release audit failed:")
        print("\n".join(findings))
        return 1
    print(f"Release audit passed ({len(paths)} tracked files scanned).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
