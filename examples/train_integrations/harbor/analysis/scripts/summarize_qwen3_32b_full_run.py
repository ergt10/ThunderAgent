#!/usr/bin/env python3
"""Aggregate trainer GPU, rollout KV-cache, and training timing artifacts for a Harbor full run."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--trainer-monitor-root", required=True)
    parser.add_argument("--rollout-monitor-dir", required=True)
    parser.add_argument("--tensorboard-dir", required=True)
    parser.add_argument("--async-trace", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    return parser.parse_args()


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def write_tsv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    rows = list(rows)
    extra_fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames and key not in extra_fields:
                extra_fields.append(key)
    all_fields = list(fieldnames) + extra_fields
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=all_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_float(raw: object) -> float | None:
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def summarize_trainer_gpu(trainer_monitor_root: Path, output_dir: Path) -> Dict[str, object]:
    timeline_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for gpu_summary_path in sorted(trainer_monitor_root.glob("*/gpu_summary.tsv")):
        node = gpu_summary_path.parent.name
        rows = read_tsv(gpu_summary_path)
        if not rows:
            continue

        by_gpu: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in rows:
            row_with_node = dict(row)
            row_with_node["node"] = node
            timeline_rows.append(row_with_node)
            by_gpu[row["gpu_index"]].append(row)

        for gpu_index, gpu_rows in sorted(by_gpu.items(), key=lambda item: int(item[0])):
            used_values = [to_float(row["memory_used_mib"]) for row in gpu_rows]
            total_values = [to_float(row["memory_total_mib"]) for row in gpu_rows]
            util_values = [to_float(row["util_gpu_pct"]) for row in gpu_rows]
            used_values = [value for value in used_values if value is not None]
            total_values = [value for value in total_values if value is not None]
            util_values = [value for value in util_values if value is not None]
            peak_used = max(used_values) if used_values else None
            peak_idx = used_values.index(peak_used) if used_values and peak_used is not None else None
            peak_timestamp = gpu_rows[peak_idx]["timestamp"] if peak_idx is not None else ""
            total_capacity = max(total_values) if total_values else None
            summary_rows.append(
                {
                    "node": node,
                    "gpu_index": gpu_index,
                    "samples": len(gpu_rows),
                    "peak_memory_used_mib": f"{peak_used:.2f}" if peak_used is not None else "",
                    "peak_memory_used_pct": (
                        f"{(peak_used / total_capacity) * 100.0:.2f}"
                        if peak_used is not None and total_capacity not in (None, 0.0)
                        else ""
                    ),
                    "mean_memory_used_mib": f"{statistics.fmean(used_values):.2f}" if used_values else "",
                    "peak_util_gpu_pct": f"{max(util_values):.2f}" if util_values else "",
                    "peak_timestamp": peak_timestamp,
                }
            )

    trainer_timeline_path = output_dir / "trainer_gpu_timeline.tsv"
    trainer_summary_path = output_dir / "trainer_gpu_summary.tsv"
    timeline_fields = [
        "timestamp",
        "node",
        "hostname",
        "gpu_index",
        "gpu_uuid",
        "memory_total_mib",
        "memory_used_mib",
        "memory_free_mib",
        "util_gpu_pct",
        "util_mem_pct",
    ]
    summary_fields = [
        "node",
        "gpu_index",
        "samples",
        "peak_memory_used_mib",
        "peak_memory_used_pct",
        "mean_memory_used_mib",
        "peak_util_gpu_pct",
        "peak_timestamp",
    ]
    write_tsv(trainer_timeline_path, timeline_rows, timeline_fields)
    write_tsv(trainer_summary_path, summary_rows, summary_fields)

    peak_by_node: Dict[str, float] = {}
    for row in summary_rows:
        peak = to_float(row["peak_memory_used_mib"])
        if peak is None:
            continue
        peak_by_node[row["node"]] = max(peak_by_node.get(row["node"], 0.0), peak)

    return {
        "trainer_gpu_timeline": str(trainer_timeline_path),
        "trainer_gpu_summary": str(trainer_summary_path),
        "trainer_gpu_peak_mib_by_node": peak_by_node,
    }


def summarize_rollout_kv(rollout_monitor_dir: Path, output_dir: Path) -> Dict[str, object]:
    metrics_path = rollout_monitor_dir / "vllm_metrics.tsv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing rollout metrics file: {metrics_path}")

    raw_rows = read_tsv(metrics_path)
    timeline_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    by_source: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for row in raw_rows:
        if row.get("kv_cache_usage_pct", "") == "":
            continue
        timeline_rows.append(row)
        by_source[row["source"]].append(row)

    for source, rows in sorted(by_source.items()):
        usage_values = [to_float(row["kv_cache_usage_pct"]) for row in rows]
        size_values = [to_float(row["kv_cache_size_tokens"]) for row in rows]
        running_values = [to_float(row["num_requests_running"]) for row in rows]
        waiting_values = [to_float(row["num_requests_waiting"]) for row in rows]
        usage_values = [value for value in usage_values if value is not None]
        size_values = [value for value in size_values if value is not None]
        running_values = [value for value in running_values if value is not None]
        waiting_values = [value for value in waiting_values if value is not None]
        peak_usage = max(usage_values) if usage_values else None
        peak_idx = usage_values.index(peak_usage) if usage_values and peak_usage is not None else None
        peak_timestamp = rows[peak_idx]["timestamp"] if peak_idx is not None else ""
        summary_rows.append(
            {
                "source": source,
                "samples": len(rows),
                "peak_kv_cache_usage_pct": f"{peak_usage:.2f}" if peak_usage is not None else "",
                "peak_kv_cache_size_tokens": f"{max(size_values):.0f}" if size_values else "",
                "mean_kv_cache_usage_pct": f"{statistics.fmean(usage_values):.2f}" if usage_values else "",
                "peak_running_requests": f"{max(running_values):.0f}" if running_values else "",
                "peak_waiting_requests": f"{max(waiting_values):.0f}" if waiting_values else "",
                "peak_timestamp": peak_timestamp,
            }
        )

    timeline_path = output_dir / "rollout_kv_cache_timeline.tsv"
    summary_path = output_dir / "rollout_kv_cache_summary.tsv"
    timeline_fields = [
        "timestamp",
        "hostname",
        "source",
        "kv_cache_usage_pct",
        "kv_cache_size_tokens",
        "prompt_throughput_tokens_per_s",
        "generation_throughput_tokens_per_s",
        "num_requests_running",
        "num_requests_waiting",
        "raw_line",
    ]
    summary_fields = [
        "source",
        "samples",
        "peak_kv_cache_usage_pct",
        "peak_kv_cache_size_tokens",
        "mean_kv_cache_usage_pct",
        "peak_running_requests",
        "peak_waiting_requests",
        "peak_timestamp",
    ]
    write_tsv(timeline_path, timeline_rows, timeline_fields)
    write_tsv(summary_path, summary_rows, summary_fields)

    peak_kv_usage = max(
        (to_float(row["peak_kv_cache_usage_pct"]) for row in summary_rows if row["peak_kv_cache_usage_pct"] != ""),
        default=0.0,
    )
    return {
        "rollout_kv_cache_timeline": str(timeline_path),
        "rollout_kv_cache_summary": str(summary_path),
        "rollout_peak_kv_cache_usage_pct": peak_kv_usage,
    }


def preferred_timing_order(tags: List[str]) -> List[str]:
    preferred = [
        "timing/step",
        "timing/wait_for_generation_buffer",
        "timing/convert_to_training_input",
        "timing/run_training",
        "timing/sync_weights",
        "timing/fwd_logprobs_values_reward",
        "timing/apply_reward_kl_penalty",
        "timing/compute_advantages_and_returns",
        "timing/train_critic_and_policy",
        "timing/eval",
        "timing/save_checkpoints",
        "timing/save_hf_model",
    ]
    ordered = [tag for tag in preferred if tag in tags]
    ordered.extend(sorted(tag for tag in tags if tag not in ordered))
    return ordered


def summarize_training_timings(tensorboard_dir: Path, output_dir: Path) -> Dict[str, object]:
    accumulator = EventAccumulator(str(tensorboard_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = accumulator.Tags().get("scalars", [])
    timing_tags = preferred_timing_order([tag for tag in scalar_tags if tag.startswith("timing/")])
    if not timing_tags:
        return {
            "train_step_timing": "",
            "train_step_timing_summary": "",
            "train_timing_steps": 0,
            "max_wait_for_rollout_sec": 0.0,
            "train_timing_missing": str(tensorboard_dir),
        }

    row_by_step: Dict[int, Dict[str, object]] = defaultdict(dict)
    for tag in timing_tags:
        short_name = tag.split("/", 1)[1]
        for event in accumulator.Scalars(tag):
            row_by_step[event.step]["step"] = event.step
            row_by_step[event.step][short_name] = event.value
            if short_name == "wait_for_generation_buffer":
                row_by_step[event.step]["wait_for_rollout_sec"] = event.value

    timeline_rows = [row_by_step[step] for step in sorted(row_by_step)]
    summary_rows: List[Dict[str, object]] = []
    for tag in timing_tags:
        short_name = tag.split("/", 1)[1]
        values = [row.get(short_name) for row in timeline_rows if row.get(short_name) is not None]
        if not values:
            continue
        max_value = max(values)
        max_step = next(row["step"] for row in timeline_rows if row.get(short_name) == max_value)
        summary_rows.append(
            {
                "metric": short_name,
                "samples": len(values),
                "mean_sec": f"{statistics.fmean(values):.4f}",
                "max_sec": f"{max_value:.4f}",
                "max_step": max_step,
                "last_sec": f"{values[-1]:.4f}",
            }
        )

    timeline_fields = ["step"]
    for tag in timing_tags:
        short_name = tag.split("/", 1)[1]
        if short_name not in timeline_fields:
            timeline_fields.append(short_name)
        if short_name == "wait_for_generation_buffer" and "wait_for_rollout_sec" not in timeline_fields:
            timeline_fields.append("wait_for_rollout_sec")
    summary_fields = ["metric", "samples", "mean_sec", "max_sec", "max_step", "last_sec"]

    timeline_path = output_dir / "train_step_timing.tsv"
    summary_path = output_dir / "train_step_timing_summary.tsv"
    write_tsv(timeline_path, timeline_rows, timeline_fields)
    write_tsv(summary_path, summary_rows, summary_fields)

    max_wait_row = next((row for row in summary_rows if row["metric"] == "wait_for_generation_buffer"), None)
    return {
        "train_step_timing": str(timeline_path),
        "train_step_timing_summary": str(summary_path),
        "train_timing_steps": len(timeline_rows),
        "max_wait_for_rollout_sec": to_float(max_wait_row["max_sec"]) if max_wait_row else 0.0,
    }


def maybe_render_async_trace(args: argparse.Namespace, output_dir: Path) -> Dict[str, object]:
    if not args.async_trace:
        return {}
    trace_path = Path(args.async_trace)
    if not trace_path.exists() or trace_path.stat().st_size == 0:
        return {}

    plot_script = Path(__file__).with_name("plot_fully_async_buffer_trace.py")
    output_path = output_dir / "async_buffer_summary.png"
    csv_dir = output_dir / "async_trace_aggregates"
    subprocess.run(
        [
            args.python_bin,
            str(plot_script),
            "--trace",
            str(trace_path),
            "--output",
            str(output_path),
            "--csv-dir",
            str(csv_dir),
        ],
        check=True,
    )
    return {
        "async_trace": str(trace_path),
        "async_buffer_summary_png": str(output_path),
        "async_trace_csv_dir": str(csv_dir),
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer_summary = summarize_trainer_gpu(Path(args.trainer_monitor_root), output_dir)
    rollout_summary = summarize_rollout_kv(Path(args.rollout_monitor_dir), output_dir)
    timing_summary = summarize_training_timings(Path(args.tensorboard_dir), output_dir)
    async_trace_summary = maybe_render_async_trace(args, output_dir)

    summary = {
        "run_name": args.run_name,
        "log_dir": args.log_dir,
        **trainer_summary,
        **rollout_summary,
        **timing_summary,
        **async_trace_summary,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY_JSON={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
