#!/usr/bin/env python3
"""Run a Harbor rootless Docker smoke test that exercises the historical verifier failure path."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import textwrap
from pathlib import Path

import harbor
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import EnvironmentConfig as TaskEnvironmentConfig
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)
from harbor.models.trial.paths import TrialPaths
from harbor.models.task.task import Task


def create_smoke_task(task_dir: Path) -> None:
    (task_dir / "environment").mkdir(parents=True, exist_ok=True)
    (task_dir / "tests").mkdir(parents=True, exist_ok=True)
    (task_dir / "solution").mkdir(parents=True, exist_ok=True)

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
    (task_dir / "environment" / "Dockerfile").write_text(
        textwrap.dedent(
            """
            FROM ubuntu:22.04
            """
        ).strip()
        + "\n"
    )
    (task_dir / "tests" / "test.sh").write_text(
        textwrap.dedent(
            """#!/usr/bin/env bash
            set -euo pipefail

            if [ -f /solution/renamed.txt ]; then
              stat -c '%u:%g %n' /solution/renamed.txt > /logs/verifier/uploaded-file.txt
            fi

            if [ -f /solution.py ]; then
              echo 1.0 > /logs/verifier/reward.txt
            else
              echo 0.0 > /logs/verifier/reward.txt
            fi
            """
        ).strip()
        + "\n"
    )
    (task_dir / "tests" / "test.sh").chmod(0o755)


async def run_direct_upload_smoke(task_dir: Path, output_root: Path) -> dict:
    task = Task(task_dir)
    trial_paths = TrialPaths(output_root / "direct-upload-smoke")
    trial_paths.mkdir()
    env = DockerEnvironment(
        environment_dir=task.paths.environment_dir,
        environment_name=f"{task.name}-direct-upload",
        session_id="rootless-upload-smoke",
        trial_paths=trial_paths,
        task_env_config=TaskEnvironmentConfig.model_validate(
            task.config.environment.model_dump()
        ),
    )

    source_file = output_root / "source.txt"
    source_file.write_text("rootless upload smoke\n")

    try:
        await env.start(force_build=True)
        await env.upload_dir(task.paths.tests_dir, "/tests")
        await env.upload_file(source_file, "/solution/renamed.txt")
        result = await env.exec(
            "stat -c '%u:%g %n' /tests/test.sh /solution/renamed.txt && cat /solution/renamed.txt"
        )
        stdout = result.stdout or ""
        if "0:0 /tests/test.sh" not in stdout or "0:0 /solution/renamed.txt" not in stdout:
            raise RuntimeError(f"Unexpected ownership after upload: {stdout}")
        return {
            "status": "pass",
            "stdout": stdout,
        }
    finally:
        await env.stop(delete=True)


async def run_trial_smoke(task_dir: Path, output_root: Path) -> dict:
    trials_dir = output_root / "trial-runs"
    config = TrialConfig(
        task=TaskConfig(path=task_dir),
        trials_dir=trials_dir,
        trial_name="rootlessVerifierFix",
        agent=AgentConfig(name="nop"),
        environment=EnvironmentConfig(force_build=True, delete=True),
        verifier=VerifierConfig(disable=False),
    )
    trial = harbor.Trial(config)
    result = await trial.run()
    if result.exception_info is not None:
        raise RuntimeError(f"Trial failed: {result.exception_info}")
    if not result.verifier_result or "reward" not in result.verifier_result.rewards:
        raise RuntimeError("Verifier did not write a reward")
    return {
        "status": "pass",
        "trial_dir": str(trial.trial_dir),
        "reward": result.verifier_result.rewards["reward"],
        "result_path": str(trial._trial_paths.result_path),
    }


async def async_main(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).expanduser().resolve()
    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    compose_probe = await asyncio.create_subprocess_exec(
        "docker",
        "compose",
        "version",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    compose_stdout, _ = await compose_probe.communicate()
    if compose_probe.returncode != 0:
        raise RuntimeError(
            "docker compose plugin is unavailable: "
            + (compose_stdout.decode(errors="replace") if compose_stdout else "")
        )

    task_dir = output_root / "task"
    create_smoke_task(task_dir)

    summary: dict[str, object] = {
        "task_dir": str(task_dir),
        "docker_host": str(args.docker_host),
    }

    if not args.skip_direct_upload_smoke:
        summary["direct_upload"] = await run_direct_upload_smoke(task_dir, output_root)

    if not args.skip_trial_smoke:
        summary["trial"] = await run_trial_smoke(task_dir, output_root)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="~/harbor_trial_smokes/rootless-verifier-fix",
        help="Directory for smoke task and artifacts",
    )
    parser.add_argument("--docker-host", default="", help="Recorded in output for debugging")
    parser.add_argument("--skip-direct-upload-smoke", action="store_true")
    parser.add_argument("--skip-trial-smoke", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
