#!/usr/bin/env python3
"""Aggregate fully-async queue trace JSONL and render a summary figure."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_events(trace_path: Path) -> list[dict]:
    events = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    events.sort(key=lambda item: (item.get("relative_time_sec", 0.0), item.get("event", "")))
    return events


def bin_index(relative_time_sec: float, bin_seconds: int) -> int:
    return int(relative_time_sec // bin_seconds)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, help="Path to async_trace.jsonl")
    parser.add_argument("--output", required=True, help="Output figure path, e.g. summary.png")
    parser.add_argument("--bin-seconds", type=int, default=60, help="Aggregation window size in seconds")
    parser.add_argument("--csv-dir", default=None, help="Optional directory to dump aggregated CSVs")
    args = parser.parse_args()

    trace_path = Path(args.trace)
    output_path = Path(args.output)
    csv_dir = Path(args.csv_dir) if args.csv_dir else None
    events = load_events(trace_path)
    if not events:
        raise SystemExit(f"No events found in {trace_path}")

    enqueue_events = [e for e in events if e["event"] == "enqueue_group"]
    dequeue_events = [e for e in events if e["event"] == "dequeue_batch"]

    max_rel_time = max(event.get("relative_time_sec", 0.0) for event in events)
    num_bins = max(1, math.ceil((max_rel_time + 1e-9) / args.bin_seconds))
    bin_centers_min = [((i + 0.5) * args.bin_seconds) / 60.0 for i in range(num_bins)]
    bin_start_min = [(i * args.bin_seconds) / 60.0 for i in range(num_bins)]

    workers = sorted({event.get("worker_id") for event in enqueue_events if "worker_id" in event})
    worker_tokens_by_bin = {worker: [0] * num_bins for worker in workers}
    worker_groups_by_bin = {worker: [0] * num_bins for worker in workers}
    dequeue_tokens_by_bin = [0] * num_bins
    dequeue_groups_by_bin = [0] * num_bins
    dequeue_samples_by_bin = [0] * num_bins

    buffer_points_t = [0.0]
    buffer_points_q = [0]
    for event in events:
        event_type = event["event"]
        rel_time = event.get("relative_time_sec", 0.0)
        idx = min(bin_index(rel_time, args.bin_seconds), num_bins - 1)
        if event_type == "enqueue_group":
            worker_id = event["worker_id"]
            worker_tokens_by_bin[worker_id][idx] += int(event.get("total_response_tokens", 0))
            worker_groups_by_bin[worker_id][idx] += 1
            buffer_points_t.append(rel_time / 60.0)
            buffer_points_q.append(int(event.get("buffer_qsize_after", 0)))
        elif event_type == "dequeue_batch":
            dequeue_tokens_by_bin[idx] += int(event.get("total_response_tokens", 0))
            dequeue_groups_by_bin[idx] += int(event.get("groups_consumed", 0))
            dequeue_samples_by_bin[idx] += int(event.get("total_samples", 0))
            buffer_points_t.append(rel_time / 60.0)
            buffer_points_q.append(int(event.get("buffer_qsize_after", 0)))
        elif event_type == "buffer_wait_start":
            buffer_points_t.append(rel_time / 60.0)
            buffer_points_q.append(int(event.get("buffer_qsize", 0)))

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)

    ax = axes[0]
    ax.step(buffer_points_t, buffer_points_q, where="post", linewidth=2, color="#0f766e")
    ax.set_title("Fully Async Buffer Queue Size")
    ax.set_ylabel("Queue size (groups)")
    ax.set_xlabel("Relative time (min)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    bottom = [0] * num_bins
    colors = plt.cm.tab10.colors
    for idx, worker_id in enumerate(workers):
        values = worker_tokens_by_bin[worker_id]
        ax.bar(
            bin_start_min,
            values,
            width=args.bin_seconds / 60.0 * 0.9,
            align="edge",
            bottom=bottom,
            color=colors[idx % len(colors)],
            label=f"worker {worker_id}",
        )
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_title(f"Rollout Worker Submission Volume per {args.bin_seconds}s")
    ax.set_ylabel("Submitted response tokens")
    ax.set_xlabel("Relative time (min)")
    ax.grid(alpha=0.3)
    if workers:
        ax.legend(ncol=min(4, len(workers)), fontsize=9)

    ax = axes[2]
    ax.bar(
        bin_start_min,
        dequeue_tokens_by_bin,
        width=args.bin_seconds / 60.0 * 0.9,
        align="edge",
        color="#1d4ed8",
        alpha=0.75,
        label="trainer consumed tokens",
    )
    ax.set_title(f"Trainer Consumption per {args.bin_seconds}s")
    ax.set_ylabel("Consumed response tokens")
    ax.set_xlabel("Relative time (min)")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(bin_centers_min, dequeue_groups_by_bin, color="#b91c1c", marker="o", label="groups consumed")
    ax2.plot(bin_centers_min, dequeue_samples_by_bin, color="#7c3aed", marker="x", label="samples consumed")
    ax2.set_ylabel("Groups / samples")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    if output_path.suffix.lower() != ".svg":
        fig.savefig(output_path.with_suffix(".svg"))
    plt.close(fig)

    if csv_dir:
        csv_dir.mkdir(parents=True, exist_ok=True)
        worker_rows = []
        for bin_idx in range(num_bins):
            row = {"window_start_min": bin_start_min[bin_idx]}
            for worker_id in workers:
                row[f"worker_{worker_id}_tokens"] = worker_tokens_by_bin[worker_id][bin_idx]
                row[f"worker_{worker_id}_groups"] = worker_groups_by_bin[worker_id][bin_idx]
            worker_rows.append(row)
        worker_fields = ["window_start_min"]
        for worker_id in workers:
            worker_fields.extend([f"worker_{worker_id}_tokens", f"worker_{worker_id}_groups"])
        write_csv(csv_dir / "rollout_worker_volume.csv", worker_rows, worker_fields)

        consume_rows = [
            {
                "window_start_min": bin_start_min[idx],
                "consumed_tokens": dequeue_tokens_by_bin[idx],
                "consumed_groups": dequeue_groups_by_bin[idx],
                "consumed_samples": dequeue_samples_by_bin[idx],
            }
            for idx in range(num_bins)
        ]
        write_csv(
            csv_dir / "trainer_consumption.csv",
            consume_rows,
            ["window_start_min", "consumed_tokens", "consumed_groups", "consumed_samples"],
        )

        qsize_rows = [
            {"relative_time_min": t, "buffer_qsize": q}
            for t, q in zip(buffer_points_t, buffer_points_q)
        ]
        write_csv(csv_dir / "buffer_qsize.csv", qsize_rows, ["relative_time_min", "buffer_qsize"])


if __name__ == "__main__":
    main()
