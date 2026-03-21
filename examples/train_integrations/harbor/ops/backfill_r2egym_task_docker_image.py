#!/usr/bin/env python3
"""Backfill Harbor task.toml [environment].docker_image from R2EGYM metadata.json."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path


def _update_task_toml(task_toml: Path, docker_image: str) -> bool:
    original = task_toml.read_text(encoding="utf-8")
    parsed = tomllib.loads(original)
    current = ((parsed.get("environment") or {}).get("docker_image")) if isinstance(parsed, dict) else None
    if current == docker_image:
        return False

    if current:
        updated = re.sub(
            r'(?m)^(\s*docker_image\s*=\s*)".*"$',
            rf'\1"{docker_image}"',
            original,
            count=1,
        )
        if updated == original:
            raise RuntimeError(f"Failed to replace existing docker_image in {task_toml}")
        task_toml.write_text(updated, encoding="utf-8")
        return True

    if re.search(r"(?m)^\[environment\]\s*$", original):
        updated = re.sub(
            r"(?m)^(\[environment\]\s*)$",
            rf'\1\ndocker_image = "{docker_image}"',
            original,
            count=1,
        )
        task_toml.write_text(updated, encoding="utf-8")
        return True

    updated = original.rstrip() + f'\n\n[environment]\ndocker_image = "{docker_image}"\n'
    task_toml.write_text(updated, encoding="utf-8")
    return True


def _process_task_dir(task_dir: Path) -> tuple[bool, str | None]:
    task_toml = task_dir / "task.toml"
    metadata_path = task_dir / "environment" / "workspace" / "metadata.json"
    if not task_toml.is_file():
        raise RuntimeError(f"Missing task.toml: {task_toml}")
    if not metadata_path.is_file():
        raise RuntimeError(f"Missing metadata.json: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    docker_image = metadata.get("docker_image")
    if not docker_image:
        raise RuntimeError(f"metadata.json missing docker_image: {metadata_path}")
    changed = _update_task_toml(task_toml, docker_image)
    return changed, docker_image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="+",
        help="One or more Harbor dataset roots, e.g. data/harbor/r2egym-trivial",
    )
    args = parser.parse_args()

    updated = 0
    skipped = 0
    for root_arg in args.roots:
        root = Path(root_arg).expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"Dataset root not found: {root}")
        for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            changed, _ = _process_task_dir(task_dir)
            if changed:
                updated += 1
            else:
                skipped += 1

    print(f"updated={updated}")
    print(f"skipped={skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
