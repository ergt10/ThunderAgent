#!/usr/bin/env python3
"""Stress Harbor Docker trial startup without rollout or trainer dependencies."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Any

import harbor
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)


def create_smoke_task(task_dir: Path) -> None:
    (task_dir / "environment").mkdir(parents=True, exist_ok=True)
    (task_dir / "tests").mkdir(parents=True, exist_ok=True)

    (task_dir / "instruction.md").write_text("Do nothing.\n")
    (task_dir / "task.toml").write_text(
        textwrap.dedent(
            """
            version = "1.0"

            [agent]
            timeout_sec = 60

            [verifier]
            timeout_sec = 60

            [environment]
            build_timeout_sec = 600
            cpus = 1
            memory_mb = 1024
            storage_mb = 1024
            gpus = 0
            allow_internet = true
            """
        ).strip()
        + "\n"
    )
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (task_dir / "tests" / "test.sh").write_text(
        textwrap.dedent(
            """#!/usr/bin/env bash
            set -euo pipefail

            echo 0.0 > /logs/verifier/reward.txt
            """
        ).strip()
        + "\n"
    )
    (task_dir / "tests" / "test.sh").chmod(0o755)


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def check_docker_compose() -> None:
    result = run_command(["docker", "compose", "version"])
    if result.returncode != 0:
        raise RuntimeError(
            "docker compose plugin is unavailable: "
            + ((result.stdout or "") + (result.stderr or ""))
        )


def classify_failure(exception_type: str | None, message: str | None, traceback_text: str | None) -> str:
    haystack = "\n".join(part for part in [exception_type or "", message or "", traceback_text or ""]).lower()
    if "all predefined address pools have been fully subnetted" in haystack:
        return "docker_address_pool_exhausted"
    if "no available ipv4 addresses on this network's address pools" in haystack:
        return "docker_address_pool_exhausted"
    if "unable to create session key" in haystack or "unable to join session keyring" in haystack:
        return "session_key_disk_quota"
    if "docker compose command failed" in haystack:
        return "docker_compose_failed"
    if "timed out after" in haystack:
        return "timeout"
    if exception_type:
        return exception_type
    return "unknown"


def list_compose_resources(kind: str) -> list[dict[str, str]]:
    if kind == "containers":
        result = run_command(
            [
                "docker",
                "ps",
                "-a",
                "--format",
                "{{.ID}}\t{{.Label \"com.docker.compose.project\"}}\t{{.Names}}",
            ]
        )
    elif kind == "networks":
        result = run_command(
            [
                "docker",
                "network",
                "ls",
                "--format",
                "{{.ID}}\t{{.Label \"com.docker.compose.project\"}}\t{{.Name}}",
            ]
        )
    else:
        raise ValueError(f"Unsupported resource kind: {kind}")

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to list docker {kind}: "
            + ((result.stdout or "") + (result.stderr or ""))
        )

    rows: list[dict[str, str]] = []
    for line in (result.stdout or "").splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        rows.append(
            {
                "id": parts[0].strip(),
                "project": parts[1].strip(),
                "name": parts[2].strip(),
            }
        )
    return rows


def capture_docker_state(trial_names: set[str]) -> dict[str, Any]:
    containers = list_compose_resources("containers")
    networks = list_compose_resources("networks")
    matching_containers = [row for row in containers if row["project"] in trial_names]
    matching_networks = [row for row in networks if row["project"] in trial_names]
    return {
        "total_container_count": len(containers),
        "total_network_count": len(networks),
        "matching_containers": matching_containers,
        "matching_networks": matching_networks,
    }


def cleanup_leftovers(trial_names: set[str]) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    for trial_name in sorted(trial_names):
        container_result = run_command(
            [
                "docker",
                "ps",
                "-aq",
                "--filter",
                f"label=com.docker.compose.project={trial_name}",
            ]
        )
        container_ids = [line.strip() for line in (container_result.stdout or "").splitlines() if line.strip()]
        if container_ids:
            rm_result = run_command(["docker", "rm", "-f", *container_ids])
            actions.append(
                {
                    "trial_name": trial_name,
                    "resource": "containers",
                    "ids": container_ids,
                    "returncode": rm_result.returncode,
                    "stdout": rm_result.stdout,
                    "stderr": rm_result.stderr,
                }
            )

        network_result = run_command(
            [
                "docker",
                "network",
                "ls",
                "-q",
                "--filter",
                f"label=com.docker.compose.project={trial_name}",
            ]
        )
        network_ids = [line.strip() for line in (network_result.stdout or "").splitlines() if line.strip()]
        if network_ids:
            rm_network_result = run_command(["docker", "network", "rm", *network_ids])
            actions.append(
                {
                    "trial_name": trial_name,
                    "resource": "networks",
                    "ids": network_ids,
                    "returncode": rm_network_result.returncode,
                    "stdout": rm_network_result.stdout,
                    "stderr": rm_network_result.stderr,
                }
            )
    return {"actions": actions}


async def run_single_trial(
    *,
    task_dir: Path,
    trials_dir: Path,
    trial_name: str,
    force_build: bool,
) -> dict[str, Any]:
    config = TrialConfig(
        task=TaskConfig(path=task_dir),
        trials_dir=trials_dir,
        trial_name=trial_name,
        agent=AgentConfig(name="nop"),
        environment=EnvironmentConfig(force_build=force_build, delete=True),
        verifier=VerifierConfig(disable=False),
    )
    trial = harbor.Trial(config)

    started = time.perf_counter()
    result = await trial.run()
    duration_sec = time.perf_counter() - started

    verifier_reward = None
    if result.verifier_result is not None and result.verifier_result.rewards is not None:
        verifier_reward = result.verifier_result.rewards.get("reward")

    exception_info = None
    failure_class = None
    if result.exception_info is not None:
        exception_info = {
            "exception_type": result.exception_info.exception_type,
            "exception_message": result.exception_info.exception_message,
            "exception_traceback": result.exception_info.exception_traceback,
        }
        failure_class = classify_failure(
            result.exception_info.exception_type,
            result.exception_info.exception_message,
            result.exception_info.exception_traceback,
        )

    return {
        "trial_name": trial_name,
        "status": "pass" if result.exception_info is None else "fail",
        "duration_sec": round(duration_sec, 3),
        "verifier_reward": verifier_reward,
        "trial_dir": str(trial.trial_dir),
        "failure_class": failure_class,
        "exception_info": exception_info,
    }


async def run_trials(
    *,
    task_dir: Path,
    trials_dir: Path,
    trial_count: int,
    max_concurrency: int,
    trial_prefix: str,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max_concurrency)
    results: list[dict[str, Any]] = []

    async def worker(index: int) -> None:
        trial_name = f"{trial_prefix}-{index:04d}"
        async with semaphore:
            results.append(
                await run_single_trial(
                    task_dir=task_dir,
                    trials_dir=trials_dir,
                    trial_name=trial_name,
                    force_build=False,
                )
            )

    await asyncio.gather(*(worker(index) for index in range(trial_count)))
    results.sort(key=lambda item: item["trial_name"])
    return results


async def async_main(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).expanduser().resolve()
    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.disable_project_network:
        os.environ["HARBOR_DOCKER_DISABLE_PROJECT_NETWORK"] = "1"
    else:
        os.environ.pop("HARBOR_DOCKER_DISABLE_PROJECT_NETWORK", None)

    check_docker_compose()

    task_dir = output_root / "task"
    trials_dir = output_root / "trials"
    create_smoke_task(task_dir)
    trials_dir.mkdir(parents=True, exist_ok=True)

    trial_names = {f"{args.trial_prefix}-{index:04d}" for index in range(args.trial_count)}
    all_trial_names = set(trial_names)
    all_trial_names.add(f"{args.trial_prefix}-warmup")
    summary: dict[str, Any] = {
        "docker_host": args.docker_host,
        "disable_project_network": args.disable_project_network,
        "output_root": str(output_root),
        "task_dir": str(task_dir),
        "trials_dir": str(trials_dir),
        "trial_count": args.trial_count,
        "max_concurrency": args.max_concurrency,
        "trial_prefix": args.trial_prefix,
        "docker_state_before": capture_docker_state(all_trial_names),
    }

    warmup_result = await run_single_trial(
        task_dir=task_dir,
        trials_dir=trials_dir,
        trial_name=f"{args.trial_prefix}-warmup",
        force_build=True,
    )
    summary["warmup"] = warmup_result
    if warmup_result["status"] != "pass":
        summary_path = output_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1

    started = time.perf_counter()
    trial_results = await run_trials(
        task_dir=task_dir,
        trials_dir=trials_dir,
        trial_count=args.trial_count,
        max_concurrency=args.max_concurrency,
        trial_prefix=args.trial_prefix,
    )
    duration_sec = time.perf_counter() - started

    failure_counts = Counter(
        result["failure_class"] for result in trial_results if result["failure_class"] is not None
    )
    passed = sum(1 for result in trial_results if result["status"] == "pass")
    failed = len(trial_results) - passed

    trial_results_path = output_root / "trial_results.json"
    trial_results_path.write_text(json.dumps(trial_results, indent=2, sort_keys=True) + "\n")

    summary["results"] = {
        "duration_sec": round(duration_sec, 3),
        "passed": passed,
        "failed": failed,
        "failure_counts": dict(sorted(failure_counts.items())),
        "max_failures": args.max_failures,
        "sample_failures": [result for result in trial_results if result["status"] == "fail"][:10],
        "trial_results_path": str(trial_results_path),
    }

    summary["docker_state_after_run"] = capture_docker_state(all_trial_names)
    summary["cleanup"] = cleanup_leftovers(all_trial_names)
    summary["docker_state_after_cleanup"] = capture_docker_state(all_trial_names)

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))

    return 0 if failed <= args.max_failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="~/harbor_trial_smokes/docker-concurrency",
        help="Directory for task, trial artifacts, and summary output",
    )
    parser.add_argument(
        "--trial-count",
        type=int,
        default=64,
        help="Total number of Harbor trials to run after the warmup build",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=64,
        help="Maximum number of Harbor trials to run concurrently",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=0,
        help="Maximum number of failed trials allowed before the script exits non-zero",
    )
    parser.add_argument(
        "--trial-prefix",
        default="docker-stress",
        help="Prefix used to name generated Harbor trials",
    )
    parser.add_argument(
        "--disable-project-network",
        action="store_true",
        help="Explicitly enable Harbor's no-network compose fallback for comparison",
    )
    parser.add_argument(
        "--docker-host",
        default="",
        help="Recorded in the summary for debugging",
    )
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.trial_count <= 0:
        raise SystemExit("--trial-count must be > 0")
    if args.max_concurrency <= 0:
        raise SystemExit("--max-concurrency must be > 0")
    if args.max_concurrency > args.trial_count:
        args.max_concurrency = args.trial_count

    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
