#!/usr/bin/env python3
"""Preflight checks for the Qwen3-32B 6-node Harbor + ThunderAgent workflow."""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import os
import resource
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
for import_root in (REPO_ROOT / "ThunderAgent", REPO_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)


@dataclass
class NodeSummary:
    address: str
    is_head: bool
    alive: bool
    cpu: float
    gpu: float
    harbor_head: float


def _require_module(name: str):
    try:
        return importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - exercised in real cluster env
        raise RuntimeError(f"Required Python module '{name}' is unavailable: {exc}") from exc


def _parse_urls(raw_urls: str, *, allow_empty: bool = False) -> list[str]:
    if not raw_urls.strip():
        if allow_empty:
            return []
        raise RuntimeError("Rollout server URLs are required unless rollout health checks are skipped.")
    try:
        parsed = ast.literal_eval(raw_urls)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse rollout server URLs: {raw_urls!r}") from exc
    if not isinstance(parsed, list) or not all(isinstance(url, str) for url in parsed):
        raise RuntimeError(f"Rollout server URLs must be a non-empty Python list of strings, got: {raw_urls!r}")
    if not parsed and not allow_empty:
        raise RuntimeError(f"Rollout server URLs must be a non-empty Python list of strings, got: {raw_urls!r}")
    return parsed


def _detect_rootless_harbor_patch() -> tuple[str, str]:
    docker_mod = _require_module("harbor.environments.docker.docker")
    try:
        module_src = inspect.getsource(docker_mod)
    except OSError:
        return (
            "warn",
            "Could not inspect Harbor docker backend source; rootless upload compatibility remains unverified.",
        )

    has_copy_markers = "compose cp" in module_src or "docker cp" in module_src
    has_rootless_safe_markers = any(
        marker in module_src
        for marker in (
            "numeric-owner",
            "put_archive(",
            "tarfile.open(",
            "exec_run(",
        )
    )

    if has_copy_markers and not has_rootless_safe_markers:
        return (
            "fail",
            "Harbor docker backend still appears to rely on docker cp/compose cp. "
            "The rootless-safe upload patch from the R4/R12 notes is likely missing.",
        )

    if not has_rootless_safe_markers:
        return (
            "warn",
            "Harbor docker backend does not expose the expected rootless-safe markers. "
            "Please manually verify upload_file()/upload_dir() before retrying rootless Harbor.",
        )

    return ("pass", "Detected a Harbor docker backend that looks compatible with rootless uploads.")


def _check_rollout_health(urls: list[str], timeout_s: float) -> None:
    for url in urls:
        health_url = f"{url.rstrip('/')}/health"
        try:
            with urllib.request.urlopen(health_url, timeout=timeout_s) as response:
                status = getattr(response, "status", 200)
                if status >= 400:
                    raise RuntimeError(f"{health_url} returned HTTP {status}")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Rollout health check failed for {health_url}: {exc}") from exc


def _normalize_rlimit_value(raw_value: int) -> int | None:
    if raw_value == resource.RLIM_INFINITY or raw_value < 0:
        return None
    return int(raw_value)


def _format_limit(value: int | None) -> int | str:
    return "unlimited" if value is None else value


def _limit_below(value: int | None, minimum: int) -> bool:
    return value is not None and value < minimum


def _read_proc_nofile_limits(pid: int) -> dict[str, int | str]:
    limits_path = Path("/proc") / str(pid) / "limits"
    try:
        for line in limits_path.read_text().splitlines():
            if line.startswith("Max open files"):
                parts = line.split()
                soft = None if parts[3] == "unlimited" else int(parts[3])
                hard = None if parts[4] == "unlimited" else int(parts[4])
                return {
                    "pid": pid,
                    "limits_path": str(limits_path),
                    "soft": _format_limit(soft),
                    "hard": _format_limit(hard),
                }
    except FileNotFoundError as exc:
        raise RuntimeError(f"Could not inspect /proc limits for pid {pid}: {exc}") from exc
    raise RuntimeError(f"Could not find 'Max open files' in {limits_path}")


def _resolve_rootless_dockerd_pid() -> tuple[int | None, str | None]:
    docker_host = os.environ.get("DOCKER_HOST", "")
    if not docker_host.startswith("unix://"):
        return (None, None)

    socket_path = docker_host.removeprefix("unix://")
    if socket_path == "/var/run/docker.sock":
        return (None, None)

    pid_candidates: list[tuple[Path, str]] = []
    if pidfile := os.environ.get("DOCKER_PIDFILE"):
        pid_candidates.append((Path(pidfile), "DOCKER_PIDFILE"))
    if xdg_runtime_dir := os.environ.get("XDG_RUNTIME_DIR"):
        pid_candidates.append((Path(xdg_runtime_dir) / "docker.pid", "XDG_RUNTIME_DIR/docker.pid"))

    for pid_path, source in pid_candidates:
        if not pid_path.exists():
            continue
        try:
            return (int(pid_path.read_text().strip()), source)
        except ValueError as exc:
            raise RuntimeError(f"Failed to parse dockerd pid from {pid_path}: {exc}") from exc

    try:
        result = subprocess.run(
            ["pgrep", "-f", f"dockerd-rootless.sh --host {docker_host}"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return (None, None)
    if result.returncode != 0 or not result.stdout.strip():
        return (None, None)
    try:
        return (int(result.stdout.splitlines()[0].strip()), "pgrep")
    except ValueError as exc:
        raise RuntimeError(f"Failed to parse dockerd pid from pgrep output: {result.stdout!r}") from exc


def _check_head_nofile(min_head_nofile: int) -> dict[str, object]:
    soft_raw, hard_raw = resource.getrlimit(resource.RLIMIT_NOFILE)
    soft = _normalize_rlimit_value(soft_raw)
    hard = _normalize_rlimit_value(hard_raw)
    if _limit_below(soft, min_head_nofile):
        raise RuntimeError(
            "Head process soft nofile is too low for the documented R13 rootless full run. "
            f"Need at least {min_head_nofile}, got {_format_limit(soft)} (hard={_format_limit(hard)})."
        )

    summary: dict[str, object] = {
        "current_process": {
            "soft": _format_limit(soft),
            "hard": _format_limit(hard),
        }
    }

    dockerd_pid, dockerd_pid_source = _resolve_rootless_dockerd_pid()
    if dockerd_pid is not None:
        dockerd_summary = _read_proc_nofile_limits(dockerd_pid)
        if _limit_below(None if dockerd_summary["soft"] == "unlimited" else int(dockerd_summary["soft"]), min_head_nofile):
            raise RuntimeError(
                "Rootless dockerd soft nofile is too low for the documented R13 rootless full run. "
                f"Need at least {min_head_nofile}, got {dockerd_summary['soft']} "
                f"(hard={dockerd_summary['hard']}, pid={dockerd_pid}, source={dockerd_pid_source})."
            )
        dockerd_summary["pid_source"] = dockerd_pid_source
        summary["rootless_dockerd"] = dockerd_summary

    return summary


def _check_ray_cluster(
    ray_address: str,
    train_num_nodes: int,
    train_gpus_per_node: int,
    require_harbor_head: bool,
    placement_group_timeout_sec: float,
    check_placement_group: bool,
) -> dict[str, Any]:
    ray = _require_module("ray")
    from ray.util.placement_group import remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    local_node_ip = ray._private.services.get_node_ip_address(ray_address)
    ray.init(
        address=ray_address,
        log_to_driver=False,
        ignore_reinit_error=True,
        _node_ip_address=local_node_ip,
    )
    try:
        nodes = ray.nodes()
        cluster_resources = ray.cluster_resources()
        available_resources = ray.available_resources()

        node_summaries = []
        eligible_training_nodes = []
        harbor_head_nodes = []

        for node in nodes:
            resources = node.get("Resources", {})
            summary = NodeSummary(
                address=node.get("NodeManagerAddress", "unknown"),
                is_head=bool(resources.get("node:__internal_head__", 0)),
                alive=bool(node.get("Alive", False)),
                cpu=float(resources.get("CPU", 0.0)),
                gpu=float(resources.get("GPU", 0.0)),
                harbor_head=float(resources.get("harbor_head", 0.0)),
            )
            node_summaries.append(summary)

            if not summary.alive:
                continue
            if summary.harbor_head >= 1:
                harbor_head_nodes.append(summary)
            if summary.gpu >= train_gpus_per_node and summary.cpu >= train_gpus_per_node:
                eligible_training_nodes.append(summary)

        required_gpus = train_num_nodes * train_gpus_per_node
        total_gpus = float(cluster_resources.get("GPU", 0.0))
        available_gpus = float(available_resources.get("GPU", 0.0))

        if require_harbor_head and not harbor_head_nodes:
            raise RuntimeError("Ray cluster does not expose the required custom resource {'harbor_head': 1}.")
        if len(eligible_training_nodes) < train_num_nodes:
            raise RuntimeError(
                "Ray sees too few schedulable trainer nodes. "
                f"Need {train_num_nodes} alive nodes with at least {train_gpus_per_node} GPU and "
                f"{train_gpus_per_node} CPU each, found {len(eligible_training_nodes)}."
            )
        if total_gpus < required_gpus:
            raise RuntimeError(f"Ray cluster_resources() reports only {total_gpus} GPUs, expected at least {required_gpus}.")
        if available_gpus < required_gpus:
            raise RuntimeError(
                f"Ray available_resources() reports only {available_gpus} free GPUs, expected at least {required_gpus}. "
                "Stray actors or mis-registered workers will block placement groups."
            )

        head_probe_node = None
        if require_harbor_head:

            @ray.remote(resources={"harbor_head": 0.001})
            def harbor_head_probe() -> dict[str, str]:
                return {
                    "hostname": socket.gethostname(),
                    "node_ip": ray.util.get_node_ip_address(),
                }

            head_probe_node = ray.get(harbor_head_probe.remote(), timeout=placement_group_timeout_sec)

        placement_group_summary = None
        if check_placement_group:
            pg = ray.util.placement_group(
                bundles=[{"CPU": float(train_gpus_per_node), "GPU": float(train_gpus_per_node)} for _ in range(train_num_nodes)],
                strategy="STRICT_SPREAD",
            )
            try:
                ray.get(pg.ready(), timeout=placement_group_timeout_sec)

                @ray.remote(num_cpus=0)
                def bundle_probe(bundle_index: int) -> dict[str, Any]:
                    return {
                        "bundle_index": bundle_index,
                        "hostname": socket.gethostname(),
                        "node_ip": ray.util.get_node_ip_address(),
                    }

                bundle_probe_refs = []
                for bundle_index in range(train_num_nodes):
                    bundle_probe_refs.append(
                        bundle_probe.options(
                            scheduling_strategy=PlacementGroupSchedulingStrategy(
                                placement_group=pg,
                                placement_group_bundle_index=bundle_index,
                                placement_group_capture_child_tasks=True,
                            )
                        ).remote(bundle_index)
                    )

                bundle_probe_results = ray.get(bundle_probe_refs, timeout=placement_group_timeout_sec)
                distinct_bundle_nodes = {probe["node_ip"] for probe in bundle_probe_results}
                if len(distinct_bundle_nodes) != train_num_nodes:
                    raise RuntimeError(
                        "Placement group became ready, but bundle probes did not land on distinct trainer nodes: "
                        f"{bundle_probe_results}"
                    )

                placement_group_summary = {
                    "bundles": [{"CPU": float(train_gpus_per_node), "GPU": float(train_gpus_per_node)} for _ in range(train_num_nodes)],
                    "bundle_probe_results": bundle_probe_results,
                    "distinct_bundle_nodes": sorted(distinct_bundle_nodes),
                }
            finally:
                remove_placement_group(pg)

        return {
            "cluster_resources": cluster_resources,
            "available_resources": available_resources,
            "nodes": [asdict(summary) for summary in node_summaries],
            "eligible_training_nodes": [asdict(summary) for summary in eligible_training_nodes],
            "harbor_head_nodes": [asdict(summary) for summary in harbor_head_nodes],
            "harbor_head_probe": head_probe_node,
            "placement_group": placement_group_summary,
        }
    finally:
        ray.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ray-address", required=True)
    parser.add_argument("--train-num-nodes", required=True, type=int)
    parser.add_argument("--train-gpus-per-node", required=True, type=int)
    parser.add_argument("--rollout-server-urls", default="")
    parser.add_argument("--skip-rollout-health", action="store_true")
    parser.add_argument("--rollout-timeout-sec", type=float, default=5.0)
    parser.add_argument("--skip-rootless-harbor-check", action="store_true")
    parser.add_argument("--skip-harbor-head-check", action="store_true")
    parser.add_argument("--skip-placement-group-check", action="store_true")
    parser.add_argument("--placement-group-timeout-sec", type=float, default=60.0)
    parser.add_argument("--min-head-nofile", type=int, default=65535)
    args = parser.parse_args()

    for module_name in ("ray", "harbor", "ThunderAgent"):
        _require_module(module_name)

    if not args.skip_rootless_harbor_check:
        status, message = _detect_rootless_harbor_patch()
        print(f"[harbor-rootless] {status.upper()}: {message}")
        if status == "fail":
            raise RuntimeError(message)

    head_nofile_summary = _check_head_nofile(args.min_head_nofile)
    print(f"[head-nofile] PASS: {json.dumps(head_nofile_summary, sort_keys=True)}")

    urls = _parse_urls(args.rollout_server_urls, allow_empty=args.skip_rollout_health)
    if not args.skip_rollout_health:
        _check_rollout_health(urls, timeout_s=args.rollout_timeout_sec)
        print(f"[rollout] PASS: {len(urls)} rollout server(s) responded to /health")
    else:
        if urls:
            print(f"[rollout] SKIP: rollout health checks disabled for {len(urls)} configured rollout server(s)")
        else:
            print("[rollout] SKIP: rollout health checks disabled and no rollout URLs were provided")

    ray_summary = _check_ray_cluster(
        ray_address=args.ray_address,
        train_num_nodes=args.train_num_nodes,
        train_gpus_per_node=args.train_gpus_per_node,
        require_harbor_head=not args.skip_harbor_head_check,
        placement_group_timeout_sec=args.placement_group_timeout_sec,
        check_placement_group=not args.skip_placement_group_check,
    )
    print(
        "[ray] PASS: cluster resources look compatible with "
        f"{args.train_num_nodes}x{args.train_gpus_per_node} trainer placement bundles"
    )
    print(json.dumps(ray_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[preflight] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
