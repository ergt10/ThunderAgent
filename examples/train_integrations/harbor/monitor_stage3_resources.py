#!/usr/bin/env python3
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib import error, request

from torch.utils.tensorboard import SummaryWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor Stage 3 GPU/system resources and write them to TSV + TensorBoard."
    )
    parser.add_argument("run_name", help="SkyRL run name, e.g. codecontest-8xh100-docker-full")
    parser.add_argument("interval_sec", nargs="?", type=float, default=10.0, help="Sampling interval in seconds")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Directory for TSV/log outputs. Defaults to $HOME/<run_name>/monitoring",
    )
    parser.add_argument(
        "--tensorboard-dir",
        default=os.environ.get("TENSORBOARD_DIR"),
        help="TensorBoard event directory. Defaults to $TENSORBOARD_DIR if set.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Kept for backward compatibility. vLLM metrics are now polled from /metrics instead of tailed from logs.",
    )
    parser.add_argument(
        "--log-glob",
        action="append",
        default=None,
        help="Kept for backward compatibility. No longer used for vLLM metrics collection.",
    )
    parser.add_argument(
        "--metrics-endpoint",
        action="append",
        default=None,
        help="Repeatable vLLM metrics endpoint in the form source_name=http://host:port or source_name=http://host:port/metrics.",
    )
    parser.add_argument(
        "--thunderagent-url",
        default=os.environ.get("THUNDERAGENT_URL"),
        help="ThunderAgent base URL or /router_state URL to poll for router state.",
    )
    parser.add_argument(
        "--trials-root",
        default=os.environ.get("SKYRL_TRIALS_ROOT"),
        help="Harbor trials_run directory used to record completed-trial progress over time.",
    )
    return parser.parse_args()


def run_command(args: List[str]) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return result.stdout


def classify_role(cmdline: str, process_name: str) -> str:
    target = f"{process_name} {cmdline}"
    if "VLLM::EngineCore" in target or "AsyncVLLMInferenceEngine" in target or " vllm" in target or "vllm " in target:
        return "rollout"
    if any(token in target for token in ("PolicyWorker", "RefWorker", "CriticWorker")):
        return "trainer"
    if "main_harbor" in target:
        return "driver"
    if "ray::" in target:
        return "ray_worker"
    return "unknown"


def safe_cmdline(pid: str) -> str:
    try:
        return run_command(["ps", "-p", pid, "-o", "cmd="]).strip()
    except Exception:
        return ""


def write_tsv_row(writer: csv.writer, row: Iterable[object]) -> None:
    writer.writerow(list(row))


def parse_human_size_to_bytes(raw: str) -> Optional[int]:
    raw = (raw or "").strip()
    if not raw:
        return None

    raw = raw.replace("iB", "B")
    value_str = ""
    unit_str = ""
    for char in raw:
        if char.isdigit() or char == ".":
            value_str += char
        elif not char.isspace():
            unit_str += char
    if not value_str:
        return None

    try:
        value = float(value_str)
    except ValueError:
        return None

    unit = unit_str.upper() or "B"
    multipliers = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "PB": 1000**5,
    }
    multiplier = multipliers.get(unit)
    if multiplier is None:
        return None
    return int(value * multiplier)


def parse_reclaimable_size_to_bytes(raw: str) -> Optional[int]:
    raw = (raw or "").strip()
    if not raw:
        return None
    return parse_human_size_to_bytes(raw.split("(", 1)[0].strip())


def safe_run_command(args: List[str]) -> Optional[str]:
    try:
        return run_command(args)
    except Exception:
        return None


def normalize_metrics_url(raw_url: str) -> str:
    url = raw_url.strip()
    if not url:
        raise ValueError("Empty metrics URL")
    if url.endswith("/metrics"):
        return url
    return url.rstrip("/") + "/metrics"


def parse_metrics_endpoint_specs(raw_values: Optional[List[str]], env_value: Optional[str]) -> List[Tuple[str, str]]:
    specs: List[str] = []
    if raw_values:
        specs.extend(raw_values)
    if env_value:
        for item in env_value.split(","):
            item = item.strip()
            if item:
                specs.append(item)

    endpoints: List[Tuple[str, str]] = []
    seen = set()
    for spec in specs:
        if "=" in spec:
            source_name, raw_url = spec.split("=", 1)
            source = source_name.strip()
            url = normalize_metrics_url(raw_url)
        else:
            url = normalize_metrics_url(spec)
            source = Path(url).stem or "metrics"
        if not source:
            raise ValueError(f"Invalid metrics endpoint spec: {spec}")
        key = (source, url)
        if key not in seen:
            seen.add(key)
            endpoints.append(key)
    return endpoints


def parse_prometheus_labels(raw_labels: str) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    if not raw_labels:
        return labels

    parts: List[str] = []
    current: List[str] = []
    in_quotes = False
    escape = False
    for char in raw_labels:
        if escape:
            current.append(char)
            escape = False
            continue
        if char == "\\":
            current.append(char)
            escape = True
            continue
        if char == '"':
            in_quotes = not in_quotes
            current.append(char)
            continue
        if char == "," and not in_quotes:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        parts.append("".join(current).strip())

    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        labels[key.strip()] = value.strip().strip('"')
    return labels


def iter_prometheus_samples(text: str) -> Iterable[Tuple[str, Dict[str, str], float]]:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            metric_part, value_part = line.rsplit(None, 1)
            value = float(value_part)
        except ValueError:
            continue
        if "{" in metric_part and metric_part.endswith("}"):
            metric_name, labels_raw = metric_part.split("{", 1)
            labels = parse_prometheus_labels(labels_raw[:-1])
        else:
            metric_name = metric_part
            labels = {}
        yield metric_name, labels, value


@dataclass
class PolledVLLMMetrics:
    metrics_url: str
    kv_cache_usage_pct: float = 0.0
    kv_cache_size_tokens: int = 0
    prompt_tokens_total: int = 0
    generation_tokens_total: int = 0
    prefix_cache_queries_total: int = 0
    prefix_cache_hits_total: int = 0
    prefix_cache_hit_rate_pct: float = 0.0
    num_requests_running: int = 0
    num_requests_waiting: int = 0
    prompt_throughput_tokens_per_s: Optional[float] = None
    generation_throughput_tokens_per_s: Optional[float] = None


@dataclass
class CounterSnapshot:
    timestamp: float
    prompt_tokens_total: int
    generation_tokens_total: int


def fetch_metrics_text(metrics_url: str, timeout_sec: float = 10.0) -> str:
    req = request.Request(metrics_url, headers={"Accept": "text/plain; version=0.0.4"})
    with request.urlopen(req, timeout=timeout_sec) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_json(url: str, timeout_sec: float = 10.0):
    req = request.Request(url, headers={"Accept": "application/json"})
    with request.urlopen(req, timeout=timeout_sec) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def normalize_router_state_url(raw_url: Optional[str]) -> Optional[str]:
    if raw_url is None:
        return None
    url = raw_url.strip()
    if not url:
        return None
    if url.endswith("/router_state"):
        return url
    return url.rstrip("/") + "/router_state"


def format_wall_timestamp(raw_ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(raw_ts))


def collect_trial_progress(trials_root: Optional[Path]) -> Dict[str, int]:
    counts = {
        "trial_dirs": 0,
        "result_json_count": 0,
        "exception_txt_count": 0,
        "trajectory_json_count": 0,
        "completed_trials_count": 0,
    }
    if trials_root is None or not trials_root.exists():
        return counts

    try:
        children = list(trials_root.iterdir())
    except OSError:
        return counts

    for child in children:
        if not child.is_dir():
            continue
        counts["trial_dirs"] += 1
        has_result = (child / "result.json").exists()
        has_exception = (child / "exception.txt").exists()
        has_trajectory = (child / "agent" / "trajectory.json").exists()
        if has_result:
            counts["result_json_count"] += 1
        if has_exception:
            counts["exception_txt_count"] += 1
        if has_trajectory:
            counts["trajectory_json_count"] += 1
        if has_result or has_exception:
            counts["completed_trials_count"] += 1
    return counts


def parse_vllm_metrics(metrics_url: str, metrics_text: str) -> PolledVLLMMetrics:
    metrics = PolledVLLMMetrics(metrics_url=metrics_url)
    block_size = 0
    num_gpu_blocks = 0

    for metric_name, labels, value in iter_prometheus_samples(metrics_text):
        if metric_name == "vllm:num_requests_running":
            metrics.num_requests_running = int(value)
        elif metric_name == "vllm:num_requests_waiting":
            metrics.num_requests_waiting = int(value)
        elif metric_name == "vllm:kv_cache_usage_perc":
            # vLLM exposes this gauge as a 0-1 ratio where 1 means 100% usage.
            metrics.kv_cache_usage_pct = float(value) * 100.0
        elif metric_name in ("vllm:prompt_tokens_total", "vllm:prompt_tokens"):
            metrics.prompt_tokens_total = int(value)
        elif metric_name in ("vllm:generation_tokens_total", "vllm:generation_tokens"):
            metrics.generation_tokens_total = int(value)
        elif metric_name in ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"):
            metrics.prefix_cache_queries_total = int(value)
        elif metric_name in ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"):
            metrics.prefix_cache_hits_total = int(value)
        elif metric_name == "vllm:cache_config_info":
            try:
                block_size = int(labels.get("block_size", "0"))
                num_gpu_blocks = int(labels.get("num_gpu_blocks", "0"))
            except ValueError:
                block_size = 0
                num_gpu_blocks = 0

    metrics.kv_cache_size_tokens = block_size * num_gpu_blocks
    if metrics.prefix_cache_queries_total > 0:
        metrics.prefix_cache_hit_rate_pct = (
            metrics.prefix_cache_hits_total / metrics.prefix_cache_queries_total
        ) * 100.0
    return metrics


def compute_throughput(
    metrics: PolledVLLMMetrics,
    now: float,
    previous: Optional[CounterSnapshot],
) -> CounterSnapshot:
    if previous is not None:
        elapsed = now - previous.timestamp
        if elapsed > 0:
            prompt_delta = metrics.prompt_tokens_total - previous.prompt_tokens_total
            generation_delta = metrics.generation_tokens_total - previous.generation_tokens_total
            if prompt_delta >= 0:
                metrics.prompt_throughput_tokens_per_s = prompt_delta / elapsed
            if generation_delta >= 0:
                metrics.generation_throughput_tokens_per_s = generation_delta / elapsed
    return CounterSnapshot(
        timestamp=now,
        prompt_tokens_total=metrics.prompt_tokens_total,
        generation_tokens_total=metrics.generation_tokens_total,
    )


def main() -> int:
    args = parse_args()

    home = Path.home()
    output_dir = Path(args.output_dir or (home / args.run_name / "monitoring"))
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = Path(args.tensorboard_dir or (home / "tmp_logs" / args.run_name / "tensorboard"))
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    hostname = os.uname().nodename
    docker_data_root = os.environ.get("DOCKER_DATA_ROOT")
    metrics_endpoints = parse_metrics_endpoint_specs(
        args.metrics_endpoint,
        os.environ.get("VLLM_METRICS_ENDPOINTS"),
    )
    thunderagent_state_url = normalize_router_state_url(args.thunderagent_url)
    trials_root = Path(args.trials_root).resolve() if args.trials_root else None

    gpu_summary_path = output_dir / "gpu_summary.tsv"
    gpu_processes_path = output_dir / "gpu_processes.tsv"
    kv_cache_log_path = output_dir / "vllm_kv_cache.log"
    vllm_metrics_path = output_dir / "vllm_metrics.tsv"
    thunderagent_backend_path = output_dir / "thunderagent_backend_state.tsv"
    thunderagent_program_path = output_dir / "thunderagent_program_state.tsv"
    thunderagent_events_path = output_dir / "thunderagent_events.tsv"
    trial_progress_path = output_dir / "trial_progress.tsv"
    docker_storage_path = output_dir / "docker_storage.tsv"
    docker_df_path = output_dir / "docker_system_df.tsv"

    summary_exists = gpu_summary_path.exists() and gpu_summary_path.stat().st_size > 0
    processes_exists = gpu_processes_path.exists() and gpu_processes_path.stat().st_size > 0
    vllm_metrics_exists = vllm_metrics_path.exists() and vllm_metrics_path.stat().st_size > 0
    thunderagent_backend_exists = thunderagent_backend_path.exists() and thunderagent_backend_path.stat().st_size > 0
    thunderagent_program_exists = thunderagent_program_path.exists() and thunderagent_program_path.stat().st_size > 0
    thunderagent_events_exists = thunderagent_events_path.exists() and thunderagent_events_path.stat().st_size > 0
    trial_progress_exists = trial_progress_path.exists() and trial_progress_path.stat().st_size > 0
    docker_storage_exists = docker_storage_path.exists() and docker_storage_path.stat().st_size > 0
    docker_df_exists = docker_df_path.exists() and docker_df_path.stat().st_size > 0

    with gpu_summary_path.open("a", newline="", encoding="utf-8") as gpu_summary_f, \
        gpu_processes_path.open("a", newline="", encoding="utf-8") as gpu_process_f, \
        vllm_metrics_path.open("a", newline="", encoding="utf-8") as vllm_metrics_f, \
        thunderagent_backend_path.open("a", newline="", encoding="utf-8") as thunderagent_backend_f, \
        thunderagent_program_path.open("a", newline="", encoding="utf-8") as thunderagent_program_f, \
        thunderagent_events_path.open("a", newline="", encoding="utf-8") as thunderagent_events_f, \
        trial_progress_path.open("a", newline="", encoding="utf-8") as trial_progress_f, \
        docker_storage_path.open("a", newline="", encoding="utf-8") as docker_storage_f, \
        docker_df_path.open("a", newline="", encoding="utf-8") as docker_df_f, \
        kv_cache_log_path.open("a", encoding="utf-8") as kv_cache_f:

        gpu_summary_writer = csv.writer(gpu_summary_f, delimiter="\t")
        gpu_process_writer = csv.writer(gpu_process_f, delimiter="\t")
        vllm_metrics_writer = csv.writer(vllm_metrics_f, delimiter="\t")
        thunderagent_backend_writer = csv.writer(thunderagent_backend_f, delimiter="\t")
        thunderagent_program_writer = csv.writer(thunderagent_program_f, delimiter="\t")
        thunderagent_events_writer = csv.writer(thunderagent_events_f, delimiter="\t")
        trial_progress_writer = csv.writer(trial_progress_f, delimiter="\t")
        docker_storage_writer = csv.writer(docker_storage_f, delimiter="\t")
        docker_df_writer = csv.writer(docker_df_f, delimiter="\t")
        if not summary_exists:
            gpu_summary_writer.writerow(
                [
                    "timestamp",
                    "hostname",
                    "gpu_index",
                    "gpu_uuid",
                    "memory_total_mib",
                    "memory_used_mib",
                    "memory_free_mib",
                    "util_gpu_pct",
                    "util_mem_pct",
                ]
            )
        if not processes_exists:
            gpu_process_writer.writerow(
                [
                    "timestamp",
                    "hostname",
                    "gpu_index",
                    "gpu_uuid",
                    "pid",
                    "used_gpu_memory_mib",
                    "role",
                    "process_name",
                    "cmdline",
                ]
            )
        if not vllm_metrics_exists:
            vllm_metrics_writer.writerow(
                [
                    "timestamp",
                    "event_timestamp",
                    "hostname",
                    "source",
                    "metrics_url",
                    "kv_cache_usage_pct",
                    "kv_cache_size_tokens",
                    "prompt_throughput_tokens_per_s",
                    "generation_throughput_tokens_per_s",
                    "num_requests_running",
                    "num_requests_waiting",
                    "prefix_cache_hit_rate_pct",
                    "prefix_cache_queries_total",
                    "prefix_cache_hits_total",
                    "prompt_tokens_total",
                    "generation_tokens_total",
                    "fetch_error",
                    "raw_line",
                ]
            )
        if not thunderagent_backend_exists:
            thunderagent_backend_writer.writerow(
                [
                    "timestamp",
                    "event_timestamp",
                    "hostname",
                    "backend_url",
                    "active_program_tokens",
                    "reasoning_program_tokens",
                    "acting_program_tokens",
                    "active_program_count",
                    "reasoning_program_count",
                    "acting_program_count",
                    "paused_program_count",
                    "total_program_tokens",
                    "shared_tokens",
                    "future_paused_tokens",
                    "capacity_overflow",
                    "active_program_tokens_ratio",
                ]
            )
        if not thunderagent_program_exists:
            thunderagent_program_writer.writerow(
                [
                    "timestamp",
                    "event_timestamp",
                    "hostname",
                    "total_programs",
                    "reasoning_programs",
                    "acting_programs",
                    "paused_programs",
                    "marked_for_pause_programs",
                    "created_total",
                    "paused_total",
                    "resumed_total",
                    "released_total",
                    "marked_for_pause_total",
                    "event_seq",
                    "fetch_error",
                ]
            )
        if not thunderagent_events_exists:
            thunderagent_events_writer.writerow(
                [
                    "timestamp",
                    "event_timestamp",
                    "hostname",
                    "seq",
                    "event_type",
                    "program_id",
                    "backend_url",
                    "origin_backend",
                    "status",
                    "state",
                    "total_tokens",
                    "step_count",
                ]
            )
        if not trial_progress_exists:
            trial_progress_writer.writerow(
                [
                    "timestamp",
                    "hostname",
                    "trials_root",
                    "trial_dirs",
                    "result_json_count",
                    "exception_txt_count",
                    "trajectory_json_count",
                    "completed_trials_count",
                ]
            )
        if not docker_storage_exists:
            docker_storage_writer.writerow(
                [
                    "timestamp",
                    "hostname",
                    "docker_data_root",
                    "docker_data_root_bytes",
                    "filesystem_size_bytes",
                    "filesystem_used_bytes",
                    "filesystem_available_bytes",
                    "filesystem_use_pct",
                ]
            )
        if not docker_df_exists:
            docker_df_writer.writerow(
                [
                    "timestamp",
                    "hostname",
                    "type",
                    "total_count",
                    "active",
                    "size_raw",
                    "size_bytes",
                    "reclaimable_raw",
                    "reclaimable_bytes",
                ]
            )

        writer = SummaryWriter(str(tensorboard_dir))
        previous_counters: Dict[str, CounterSnapshot] = {}
        thunderagent_since_seq = 0
        running = True

        def _stop(_signum, _frame):
            nonlocal running
            running = False

        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)

        sample_idx = 0
        while running:
            now = time.time()
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now))

            try:
                gpu_output = run_command(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory",
                        "--format=csv,noheader,nounits",
                    ]
                )
            except Exception as exc:
                print(f"Failed to query nvidia-smi GPUs: {exc}", file=sys.stderr)
                time.sleep(args.interval_sec)
                continue

            total_used = 0.0
            total_free = 0.0
            total_capacity = 0.0
            util_gpu_values: List[float] = []
            util_mem_values: List[float] = []
            gpu_uuid_to_index: Dict[str, str] = {}

            for raw_line in gpu_output.splitlines():
                if not raw_line.strip():
                    continue
                idx, uuid, total, used, free, util_gpu, util_mem = [part.strip() for part in raw_line.split(",")]
                gpu_uuid_to_index[uuid] = idx
                total_f = float(total)
                used_f = float(used)
                free_f = float(free)
                util_gpu_f = float(util_gpu)
                util_mem_f = float(util_mem)

                write_tsv_row(
                    gpu_summary_writer,
                    [timestamp, hostname, idx, uuid, total, used, free, util_gpu, util_mem],
                )

                writer.add_scalar(f"monitor/gpu/{idx}/memory_total_mib", total_f, sample_idx, walltime=now)
                writer.add_scalar(f"monitor/gpu/{idx}/memory_used_mib", used_f, sample_idx, walltime=now)
                writer.add_scalar(f"monitor/gpu/{idx}/memory_free_mib", free_f, sample_idx, walltime=now)
                writer.add_scalar(f"monitor/gpu/{idx}/util_gpu_pct", util_gpu_f, sample_idx, walltime=now)
                writer.add_scalar(f"monitor/gpu/{idx}/util_mem_pct", util_mem_f, sample_idx, walltime=now)

                total_used += used_f
                total_free += free_f
                total_capacity += total_f
                util_gpu_values.append(util_gpu_f)
                util_mem_values.append(util_mem_f)

            if total_capacity > 0:
                writer.add_scalar("monitor/gpu/all/memory_used_mib_sum", total_used, sample_idx, walltime=now)
                writer.add_scalar("monitor/gpu/all/memory_free_mib_sum", total_free, sample_idx, walltime=now)
                writer.add_scalar(
                    "monitor/gpu/all/memory_used_pct",
                    (total_used / total_capacity) * 100.0,
                    sample_idx,
                    walltime=now,
                )
            if util_gpu_values:
                writer.add_scalar(
                    "monitor/gpu/all/util_gpu_pct_mean",
                    sum(util_gpu_values) / len(util_gpu_values),
                    sample_idx,
                    walltime=now,
                )
                writer.add_scalar(
                    "monitor/gpu/all/util_mem_pct_mean",
                    sum(util_mem_values) / len(util_mem_values),
                    sample_idx,
                    walltime=now,
                )

            role_memory = defaultdict(float)
            role_count = defaultdict(int)
            try:
                proc_output = run_command(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=gpu_uuid,pid,used_gpu_memory,process_name",
                        "--format=csv,noheader,nounits",
                    ]
                )
            except Exception:
                proc_output = ""

            for raw_line in proc_output.splitlines():
                if not raw_line.strip():
                    continue
                gpu_uuid, pid, used_gpu_memory, process_name = [part.strip() for part in raw_line.split(",")]
                cmdline = safe_cmdline(pid)
                role = classify_role(cmdline, process_name)
                used_gpu_memory_f = float(used_gpu_memory)
                gpu_index = gpu_uuid_to_index.get(gpu_uuid, "")
                write_tsv_row(
                    gpu_process_writer,
                    [timestamp, hostname, gpu_index, gpu_uuid, pid, used_gpu_memory, role, process_name, cmdline],
                )
                role_memory[role] += used_gpu_memory_f
                role_count[role] += 1

            for role, used_mib in role_memory.items():
                writer.add_scalar(
                    f"monitor/process/{role}/gpu_memory_used_mib_sum",
                    used_mib,
                    sample_idx,
                    walltime=now,
                )
                writer.add_scalar(
                    f"monitor/process/{role}/process_count",
                    role_count[role],
                    sample_idx,
                    walltime=now,
                )

            if docker_data_root:
                du_output = safe_run_command(["du", "-sb", docker_data_root])
                df_output = safe_run_command(["df", "-B1", docker_data_root])
                docker_data_root_bytes = None
                filesystem_size_bytes = None
                filesystem_used_bytes = None
                filesystem_available_bytes = None
                filesystem_use_pct = None

                if du_output:
                    du_parts = du_output.splitlines()[0].split("\t", 1)
                    if du_parts:
                        try:
                            docker_data_root_bytes = int(du_parts[0])
                        except ValueError:
                            docker_data_root_bytes = None

                if df_output:
                    df_lines = [line for line in df_output.splitlines() if line.strip()]
                    if len(df_lines) >= 2:
                        df_parts = df_lines[1].split()
                        if len(df_parts) >= 5:
                            try:
                                filesystem_size_bytes = int(df_parts[1])
                                filesystem_used_bytes = int(df_parts[2])
                                filesystem_available_bytes = int(df_parts[3])
                                filesystem_use_pct = float(df_parts[4].rstrip("%"))
                            except ValueError:
                                filesystem_size_bytes = None
                                filesystem_used_bytes = None
                                filesystem_available_bytes = None
                                filesystem_use_pct = None

                write_tsv_row(
                    docker_storage_writer,
                    [
                        timestamp,
                        hostname,
                        docker_data_root,
                        docker_data_root_bytes if docker_data_root_bytes is not None else "",
                        filesystem_size_bytes if filesystem_size_bytes is not None else "",
                        filesystem_used_bytes if filesystem_used_bytes is not None else "",
                        filesystem_available_bytes if filesystem_available_bytes is not None else "",
                        filesystem_use_pct if filesystem_use_pct is not None else "",
                    ],
                )

                if docker_data_root_bytes is not None:
                    writer.add_scalar(
                        "monitor/docker/data_root_bytes",
                        docker_data_root_bytes,
                        sample_idx,
                        walltime=now,
                    )
                if filesystem_use_pct is not None:
                    writer.add_scalar(
                        "monitor/docker/filesystem_use_pct",
                        filesystem_use_pct,
                        sample_idx,
                        walltime=now,
                    )

            docker_df_output = safe_run_command(["docker", "system", "df", "--format", "{{json .}}"])
            if docker_df_output:
                for raw_line in docker_df_output.splitlines():
                    if not raw_line.strip():
                        continue
                    try:
                        row = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    type_name = row.get("Type", "")
                    total_count = row.get("TotalCount", "")
                    active = row.get("Active", "")
                    size_raw = row.get("Size", "")
                    reclaimable_raw = row.get("Reclaimable", "")
                    size_bytes = parse_human_size_to_bytes(size_raw)
                    reclaimable_bytes = parse_reclaimable_size_to_bytes(reclaimable_raw)

                    write_tsv_row(
                        docker_df_writer,
                        [
                            timestamp,
                            hostname,
                            type_name,
                            total_count,
                            active,
                            size_raw,
                            size_bytes if size_bytes is not None else "",
                            reclaimable_raw,
                            reclaimable_bytes if reclaimable_bytes is not None else "",
                        ],
                    )

                    type_tag = type_name.lower().replace(" ", "_")
                    if size_bytes is not None:
                        writer.add_scalar(
                            f"monitor/docker/system_df/{type_tag}/size_bytes",
                            size_bytes,
                            sample_idx,
                            walltime=now,
                        )
                    if reclaimable_bytes is not None:
                        writer.add_scalar(
                            f"monitor/docker/system_df/{type_tag}/reclaimable_bytes",
                            reclaimable_bytes,
                            sample_idx,
                            walltime=now,
                        )

            for source_name, metrics_url in metrics_endpoints:
                source_tag = Path(source_name).stem.replace("-", "_")
                fetch_error = ""
                try:
                    metrics_text = fetch_metrics_text(metrics_url)
                    metrics = parse_vllm_metrics(metrics_url, metrics_text)
                    previous_counters[source_name] = compute_throughput(
                        metrics,
                        now,
                        previous_counters.get(source_name),
                    )
                except (error.URLError, TimeoutError, OSError, ValueError) as exc:
                    metrics = PolledVLLMMetrics(metrics_url=metrics_url)
                    fetch_error = f"{type(exc).__name__}: {exc!r}"
                    print(f"Failed to scrape {metrics_url}: {fetch_error}", file=sys.stderr)

                if fetch_error:
                    continue

                raw_line = json.dumps(
                    {
                        "kind": "polled_metrics",
                        "metrics_url": metrics_url,
                    },
                    separators=(",", ":"),
                )
                write_tsv_row(
                    vllm_metrics_writer,
                    [
                        timestamp,
                        timestamp,
                        hostname,
                        source_name,
                        metrics.metrics_url,
                        metrics.kv_cache_usage_pct,
                        metrics.kv_cache_size_tokens,
                        metrics.prompt_throughput_tokens_per_s if metrics.prompt_throughput_tokens_per_s is not None else "",
                        metrics.generation_throughput_tokens_per_s if metrics.generation_throughput_tokens_per_s is not None else "",
                        metrics.num_requests_running,
                        metrics.num_requests_waiting,
                        metrics.prefix_cache_hit_rate_pct,
                        metrics.prefix_cache_queries_total,
                        metrics.prefix_cache_hits_total,
                        metrics.prompt_tokens_total,
                        metrics.generation_tokens_total,
                        "",
                        raw_line,
                    ],
                )

                writer.add_scalar(
                    f"monitor/vllm/{source_tag}/kv_cache_usage_pct",
                    metrics.kv_cache_usage_pct,
                    sample_idx,
                    walltime=now,
                )
                writer.add_scalar(
                    f"monitor/vllm/{source_tag}/kv_cache_size_tokens",
                    metrics.kv_cache_size_tokens,
                    sample_idx,
                    walltime=now,
                )
                writer.add_scalar(
                    f"monitor/vllm/{source_tag}/num_requests_running",
                    metrics.num_requests_running,
                    sample_idx,
                    walltime=now,
                )
                writer.add_scalar(
                    f"monitor/vllm/{source_tag}/num_requests_waiting",
                    metrics.num_requests_waiting,
                    sample_idx,
                    walltime=now,
                )
                writer.add_scalar(
                    f"monitor/vllm/{source_tag}/prefix_cache_hit_rate_pct",
                    metrics.prefix_cache_hit_rate_pct,
                    sample_idx,
                    walltime=now,
                )
                if metrics.prompt_throughput_tokens_per_s is not None:
                    writer.add_scalar(
                        f"monitor/vllm/{source_tag}/prompt_throughput_tokens_per_s",
                        metrics.prompt_throughput_tokens_per_s,
                        sample_idx,
                        walltime=now,
                    )
                if metrics.generation_throughput_tokens_per_s is not None:
                    writer.add_scalar(
                        f"monitor/vllm/{source_tag}/generation_throughput_tokens_per_s",
                        metrics.generation_throughput_tokens_per_s,
                        sample_idx,
                        walltime=now,
                    )
                kv_cache_f.write(
                    "\t".join(
                        [
                            timestamp,
                            hostname,
                            source_name,
                            f"kv_cache_usage_pct={metrics.kv_cache_usage_pct:.3f}",
                            f"kv_cache_size_tokens={metrics.kv_cache_size_tokens}",
                            f"num_requests_running={metrics.num_requests_running}",
                            f"num_requests_waiting={metrics.num_requests_waiting}",
                            f"prefix_cache_hit_rate_pct={metrics.prefix_cache_hit_rate_pct:.3f}",
                            f"metrics_url={metrics.metrics_url}",
                        ]
                    )
                    + "\n"
                )

            if thunderagent_state_url:
                router_fetch_error = ""
                router_event_timestamp = timestamp
                try:
                    router_state = fetch_json(f"{thunderagent_state_url}?since_seq={thunderagent_since_seq}")
                except (error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
                    router_state = {}
                    router_fetch_error = f"{type(exc).__name__}: {exc!r}"
                    print(f"Failed to scrape {thunderagent_state_url}: {router_fetch_error}", file=sys.stderr)

                if isinstance(router_state, dict) and router_state:
                    router_ts = router_state.get("timestamp")
                    if isinstance(router_ts, (int, float)):
                        router_event_timestamp = format_wall_timestamp(float(router_ts))

                    counters = router_state.get("counters") or {}
                    stats = router_state.get("program_stats") or {}
                    event_seq = int(router_state.get("event_seq") or thunderagent_since_seq)
                    thunderagent_since_seq = max(thunderagent_since_seq, event_seq)

                    write_tsv_row(
                        thunderagent_program_writer,
                        [
                            timestamp,
                            router_event_timestamp,
                            hostname,
                            int(stats.get("total") or 0),
                            int(stats.get("reasoning") or 0),
                            int(stats.get("acting") or 0),
                            int(stats.get("paused") or 0),
                            int(stats.get("marked_for_pause") or 0),
                            int(counters.get("created") or 0),
                            int(counters.get("paused") or 0),
                            int(counters.get("resumed") or 0),
                            int(counters.get("released") or 0),
                            int(counters.get("marked_for_pause") or 0),
                            event_seq,
                            "",
                        ],
                    )

                    writer.add_scalar(
                        "monitor/thunderagent/programs/total",
                        int(stats.get("total") or 0),
                        sample_idx,
                        walltime=now,
                    )
                    writer.add_scalar(
                        "monitor/thunderagent/programs/reasoning",
                        int(stats.get("reasoning") or 0),
                        sample_idx,
                        walltime=now,
                    )
                    writer.add_scalar(
                        "monitor/thunderagent/programs/acting",
                        int(stats.get("acting") or 0),
                        sample_idx,
                        walltime=now,
                    )
                    writer.add_scalar(
                        "monitor/thunderagent/programs/paused",
                        int(stats.get("paused") or 0),
                        sample_idx,
                        walltime=now,
                    )
                    writer.add_scalar(
                        "monitor/thunderagent/counters/released_total",
                        int(counters.get("released") or 0),
                        sample_idx,
                        walltime=now,
                    )
                    writer.add_scalar(
                        "monitor/thunderagent/counters/paused_total",
                        int(counters.get("paused") or 0),
                        sample_idx,
                        walltime=now,
                    )
                    writer.add_scalar(
                        "monitor/thunderagent/counters/resumed_total",
                        int(counters.get("resumed") or 0),
                        sample_idx,
                        walltime=now,
                    )

                    for backend_url, backend_data in sorted((router_state.get("backends") or {}).items()):
                        backend_tag = backend_url.replace("http://", "").replace("https://", "").replace(":", "_").replace("/", "_")
                        write_tsv_row(
                            thunderagent_backend_writer,
                            [
                                timestamp,
                                router_event_timestamp,
                                hostname,
                                backend_url,
                                int(backend_data.get("active_program_tokens") or 0),
                                int(backend_data.get("reasoning_program_tokens") or 0),
                                int(backend_data.get("acting_program_tokens") or 0),
                                int(backend_data.get("active_program_count") or 0),
                                int(backend_data.get("reasoning_program_count") or 0),
                                int(backend_data.get("acting_program_count") or 0),
                                int(backend_data.get("paused_program_count") or 0),
                                int(backend_data.get("total_program_tokens") or 0),
                                int(backend_data.get("shared_tokens") or 0),
                                int(backend_data.get("future_paused_tokens") or 0),
                                int(backend_data.get("capacity_overflow") or 0),
                                float(backend_data.get("active_program_tokens_ratio") or 0.0),
                            ],
                        )
                        writer.add_scalar(
                            f"monitor/thunderagent/backend/{backend_tag}/reasoning_program_tokens",
                            int(backend_data.get("reasoning_program_tokens") or 0),
                            sample_idx,
                            walltime=now,
                        )
                        writer.add_scalar(
                            f"monitor/thunderagent/backend/{backend_tag}/acting_program_tokens",
                            int(backend_data.get("acting_program_tokens") or 0),
                            sample_idx,
                            walltime=now,
                        )
                        writer.add_scalar(
                            f"monitor/thunderagent/backend/{backend_tag}/reasoning_program_count",
                            int(backend_data.get("reasoning_program_count") or 0),
                            sample_idx,
                            walltime=now,
                        )
                        writer.add_scalar(
                            f"monitor/thunderagent/backend/{backend_tag}/acting_program_count",
                            int(backend_data.get("acting_program_count") or 0),
                            sample_idx,
                            walltime=now,
                        )
                        writer.add_scalar(
                            f"monitor/thunderagent/backend/{backend_tag}/paused_program_count",
                            int(backend_data.get("paused_program_count") or 0),
                            sample_idx,
                            walltime=now,
                        )

                    for event_row in router_state.get("recent_events") or []:
                        event_seq = int(event_row.get("seq") or 0)
                        thunderagent_since_seq = max(thunderagent_since_seq, event_seq)
                        event_ts = event_row.get("timestamp")
                        event_ts_str = (
                            format_wall_timestamp(float(event_ts))
                            if isinstance(event_ts, (int, float))
                            else router_event_timestamp
                        )
                        write_tsv_row(
                            thunderagent_events_writer,
                            [
                                timestamp,
                                event_ts_str,
                                hostname,
                                event_seq,
                                event_row.get("event_type") or "",
                                event_row.get("program_id") or "",
                                event_row.get("backend_url") or "",
                                event_row.get("origin_backend") or "",
                                event_row.get("status") or "",
                                event_row.get("state") or "",
                                int(event_row.get("total_tokens") or 0),
                                int(event_row.get("step_count") or 0),
                            ],
                        )
                else:
                    write_tsv_row(
                        thunderagent_program_writer,
                        [
                            timestamp,
                            timestamp,
                            hostname,
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            thunderagent_since_seq,
                            router_fetch_error,
                        ],
                    )

            if trials_root is not None:
                progress = collect_trial_progress(trials_root)
                write_tsv_row(
                    trial_progress_writer,
                    [
                        timestamp,
                        hostname,
                        str(trials_root),
                        progress["trial_dirs"],
                        progress["result_json_count"],
                        progress["exception_txt_count"],
                        progress["trajectory_json_count"],
                        progress["completed_trials_count"],
                    ],
                )
                writer.add_scalar(
                    "monitor/trials/completed_trials_count",
                    progress["completed_trials_count"],
                    sample_idx,
                    walltime=now,
                )
                writer.add_scalar(
                    "monitor/trials/result_json_count",
                    progress["result_json_count"],
                    sample_idx,
                    walltime=now,
                )
                writer.add_scalar(
                    "monitor/trials/exception_txt_count",
                    progress["exception_txt_count"],
                    sample_idx,
                    walltime=now,
                )

            gpu_summary_f.flush()
            gpu_process_f.flush()
            vllm_metrics_f.flush()
            thunderagent_backend_f.flush()
            thunderagent_program_f.flush()
            thunderagent_events_f.flush()
            trial_progress_f.flush()
            docker_storage_f.flush()
            docker_df_f.flush()
            kv_cache_f.flush()
            writer.flush()

            sample_idx += 1
            time.sleep(args.interval_sec)

        writer.flush()
        writer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
