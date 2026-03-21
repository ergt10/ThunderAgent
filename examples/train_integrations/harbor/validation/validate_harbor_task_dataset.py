#!/usr/bin/env python3
"""Validate Harbor task dataset directories and compare them with stage requirements."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from harbor.models.task.paths import TaskPaths


def _parse_dataset_spec(raw: str) -> list[Path]:
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        parsed = raw

    if isinstance(parsed, str):
        items = [parsed]
    elif isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        items = parsed
    else:
        raise ValueError(f"Dataset spec must be a path string or Python list of path strings, got: {raw!r}")

    return [Path(item).expanduser() for item in items]


def inspect_dataset_path(dataset_path: Path) -> dict:
    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.exists():
        return {
            "path": str(dataset_path),
            "exists": False,
            "valid_tasks": 0,
            "invalid_entries": [],
        }

    if TaskPaths(dataset_path).is_valid():
        candidates = [dataset_path]
    else:
        candidates = sorted(p for p in dataset_path.iterdir() if p.is_dir())

    valid_tasks = []
    invalid_entries = []
    for candidate in candidates:
        paths = TaskPaths(candidate)
        missing = []
        if not paths.instruction_path.exists():
            missing.append("instruction.md")
        if not paths.config_path.exists():
            missing.append("task.toml")
        if not paths.environment_dir.exists():
            missing.append("environment/")
        if not paths.test_path.exists():
            missing.append("tests/test.sh")

        if missing:
            invalid_entries.append({"path": str(candidate), "missing": missing})
        else:
            valid_tasks.append(str(candidate))

    return {
        "path": str(dataset_path),
        "exists": True,
        "valid_tasks": len(valid_tasks),
        "sample_valid_tasks": valid_tasks[:5],
        "invalid_entries": invalid_entries[:20],
    }


def inspect_dataset_spec(raw: str) -> dict:
    sources = [inspect_dataset_path(path) for path in _parse_dataset_spec(raw)]
    combined_sample_valid_tasks = []
    combined_invalid_entries = []
    for source in sources:
        combined_sample_valid_tasks.extend(source.get("sample_valid_tasks", []))
        combined_invalid_entries.extend(source.get("invalid_entries", []))

    return {
        "spec": raw,
        "exists": all(source["exists"] for source in sources),
        "valid_tasks": sum(source["valid_tasks"] for source in sources),
        "sample_valid_tasks": combined_sample_valid_tasks[:5],
        "invalid_entries": combined_invalid_entries[:20],
        "sources": sources,
    }


def stage_requirements(
    stage: str,
    train_num_nodes: int,
    train_gpus_per_node: int,
    full_train_batch_size: int,
) -> tuple[int, int]:
    world_size = train_num_nodes * train_gpus_per_node
    if stage == "smoke":
        return max(world_size // 2, 1), 4
    if stage == "pilot":
        return 64, 10
    if stage == "full":
        return full_train_batch_size, 20
    raise ValueError(f"Unsupported stage: {stage}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--stage", choices=("smoke", "pilot", "full"), default="full")
    parser.add_argument("--train-num-nodes", type=int, default=4)
    parser.add_argument("--train-gpus-per-node", type=int, default=8)
    parser.add_argument("--full-train-batch-size", type=int, default=64)
    args = parser.parse_args()

    train_summary = inspect_dataset_spec(args.train_data)
    eval_summary = inspect_dataset_spec(args.eval_data)
    min_train, min_eval = stage_requirements(
        args.stage,
        train_num_nodes=args.train_num_nodes,
        train_gpus_per_node=args.train_gpus_per_node,
        full_train_batch_size=args.full_train_batch_size,
    )

    summary = {
        "stage": args.stage,
        "requirements": {
            "min_train_tasks": min_train,
            "min_eval_tasks": min_eval,
        },
        "train": train_summary,
        "eval": eval_summary,
    }

    print(json.dumps(summary, indent=2, sort_keys=True))

    if not train_summary["exists"]:
        raise SystemExit("train dataset directory is missing")
    if not eval_summary["exists"]:
        raise SystemExit("eval dataset directory is missing")
    if train_summary["valid_tasks"] < min_train:
        raise SystemExit(
            f"train dataset has only {train_summary['valid_tasks']} valid tasks, need at least {min_train}"
        )
    if eval_summary["valid_tasks"] < min_eval:
        raise SystemExit(
            f"eval dataset has only {eval_summary['valid_tasks']} valid tasks, need at least {min_eval}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
