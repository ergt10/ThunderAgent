#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


PRIMARY_PATTERNS = (
    "ThunderAgent",
    "thunderagent.log",
    "HTTP Inference (ThunderAgent)",
)

FALLBACK_PATTERNS = (
    "Mesh Ranks:",
    "Initializing process group for RayActorGroup",
    "Initialized process group for RayActorGroup",
    "resource_tracker: There appear to be",
)

PID_RE = re.compile(r"-(\d+)\.(?:out|err)$")


@dataclass(order=True)
class Candidate:
    mtime: float
    pid: int
    path: str
    pattern_group: str


def _matches(path: str, patterns: tuple[str, ...]) -> bool:
    try:
        with open(path, "r", errors="ignore") as fh:
            for line in fh:
                for pattern in patterns:
                    if pattern in line:
                        return True
    except OSError:
        return False
    return False


def _collect_candidates(ray_log_dir: Path, patterns: tuple[str, ...], pattern_group: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    for path in glob.glob(str(ray_log_dir / "worker-*.out")) + glob.glob(str(ray_log_dir / "worker-*.err")):
        match = PID_RE.search(path)
        if not match:
            continue
        if not _matches(path, patterns):
            continue
        try:
            stat = os.stat(path)
        except OSError:
            continue
        candidates.append(
            Candidate(
                mtime=stat.st_mtime,
                pid=int(match.group(1)),
                path=path,
                pattern_group=pattern_group,
            )
        )
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-log-dir", default="/tmp/ray/session_latest/logs")
    parser.add_argument("--pid-only", action="store_true")
    args = parser.parse_args()

    ray_log_dir = Path(args.ray_log_dir)
    primary = _collect_candidates(ray_log_dir, PRIMARY_PATTERNS, "primary")
    fallback = _collect_candidates(ray_log_dir, FALLBACK_PATTERNS, "fallback")
    candidates = primary or fallback
    if not candidates:
        return 1

    best = max(candidates)
    if args.pid_only:
        print(best.pid)
    else:
        print(f"pid={best.pid} path={best.path} source={best.pattern_group} mtime={best.mtime:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
