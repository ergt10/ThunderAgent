#!/usr/bin/env python3
"""Rewrite a source-node freeze file into a workspace-local uv install file."""

from __future__ import annotations

import argparse
from pathlib import Path


def rewrite_line(line: str, repo_root: Path) -> str:
    stripped = line.strip()
    if stripped == "-e file:///home/hkang/zthunder_yagent/SkyRL":
        return f"-e file://{repo_root}"
    if stripped == "-e file:///home/hkang/zthunder_yagent/SkyRL/skyrl-gym":
        return f"-e file://{repo_root / 'skyrl-gym'}"
    return line.rstrip("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, dest="input_path")
    parser.add_argument("--output", required=True, dest="output_path")
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()
    output_path = Path(args.output_path).resolve()
    repo_root = Path(args.repo_root).resolve()

    rewritten = [rewrite_line(line, repo_root) for line in input_path.read_text().splitlines()]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rewritten) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
