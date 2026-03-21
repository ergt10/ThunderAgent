#!/usr/bin/env python3
"""Analyze SkyRL memory event traces and generate an overview figure."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


GIB = 2**30


@dataclass
class Event:
    timestamp_ns: int
    timestamp: str
    reason: str
    role: str
    rank: int
    hostname: str
    gpu_id: str
    local_rank: int
    metrics: dict[str, float]
    extra: dict[str, object]


@dataclass
class StageSpan:
    label: str
    start_ns: int
    end_ns: int
    color: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def gib_metrics(raw: dict[str, object]) -> dict[str, float]:
    return {
        "active_gib": float(raw["cuda_active_bytes"]) / GIB,
        "allocated_gib": float(raw["cuda_allocated_bytes"]) / GIB,
        "reserved_gib": float(raw["cuda_reserved_bytes"]) / GIB,
        "reserved_unallocated_gib": float(raw["cuda_reserved_unallocated_bytes"]) / GIB,
        "free_gib": float(raw["cuda_free_bytes"]) / GIB,
        "model_param_gib": float(raw["model_param_bytes_cuda"]) / GIB,
        "optimizer_state_gib": float(raw["optimizer_state_bytes_cuda"]) / GIB,
        "grad_gib": float(raw["grad_bytes_cuda"]) / GIB,
        "buffer_gib": float(raw["model_buffer_bytes_cuda"]) / GIB,
    }


def load_events(memory_dir: Path) -> tuple[list[Event], dict[str, list[Event]]]:
    all_events: list[Event] = []
    by_file: dict[str, list[Event]] = {}
    for path in sorted(memory_dir.glob("*.jsonl")):
        file_events: list[Event] = []
        with path.open() as f:
            for line in f:
                raw = json.loads(line)
                event = Event(
                    timestamp_ns=int(raw["timestamp_ns"]),
                    timestamp=raw["timestamp"],
                    reason=str(raw["reason"]),
                    role=str(raw["worker_role"]),
                    rank=int(raw["rank"]),
                    hostname=str(raw["hostname"]).split(".")[0],
                    gpu_id=str(raw["gpu_id"]),
                    local_rank=int(raw["local_rank"]),
                    metrics=gib_metrics(raw["metrics"]),
                    extra=dict(raw.get("extra", {})),
                )
                file_events.append(event)
                all_events.append(event)
        by_file[path.name] = file_events
    all_events.sort(key=lambda e: e.timestamp_ns)
    return all_events, by_file


def group_gpu_streams(all_events: Iterable[Event]) -> dict[tuple[str, int], dict[str, list[Event]]]:
    grouped: dict[tuple[str, int], dict[str, list[Event]]] = defaultdict(lambda: {"policy": [], "ref": []})
    for event in all_events:
        key = (event.hostname, event.local_rank)
        role = "policy" if "Policy" in event.role else "ref"
        grouped[key][role].append(event)
    return grouped


def merged_gpu_timeline(streams: dict[str, list[Event]]) -> dict[str, np.ndarray]:
    policy_events = sorted(streams["policy"], key=lambda e: e.timestamp_ns)
    ref_events = sorted(streams["ref"], key=lambda e: e.timestamp_ns)
    timestamps = sorted({e.timestamp_ns for e in policy_events} | {e.timestamp_ns for e in ref_events})

    if not timestamps:
        raise ValueError("No events found for GPU stream")

    metric_keys = list(policy_events[0].metrics.keys() if policy_events else ref_events[0].metrics.keys())
    out = {key: np.zeros(len(timestamps), dtype=float) for key in metric_keys}
    out["timestamp_ns"] = np.array(timestamps, dtype=np.int64)

    p_idx = 0
    r_idx = 0
    current_policy = {k: 0.0 for k in metric_keys}
    current_ref = {k: 0.0 for k in metric_keys}

    for i, ts in enumerate(timestamps):
        while p_idx < len(policy_events) and policy_events[p_idx].timestamp_ns <= ts:
            current_policy = policy_events[p_idx].metrics
            p_idx += 1
        while r_idx < len(ref_events) and ref_events[r_idx].timestamp_ns <= ts:
            current_ref = ref_events[r_idx].metrics
            r_idx += 1
        for key in metric_keys:
            out[key][i] = current_policy.get(key, 0.0) + current_ref.get(key, 0.0)
    return out


def find_rank0_files(by_file: dict[str, list[Event]]) -> tuple[list[Event], list[Event]]:
    policy = None
    ref = None
    for events in by_file.values():
        if not events:
            continue
        first = events[0]
        if first.rank == 0 and "Policy" in first.role:
            policy = events
        if first.rank == 0 and "Ref" in first.role:
            ref = events
    if policy is None or ref is None:
        raise ValueError("Missing rank 0 policy/ref traces")
    return policy, ref


def stage_name(reason: str) -> str:
    if reason.startswith("ref_"):
        return "ref_logprob"
    if reason.startswith("policy_forward_micro_") or reason.startswith("policy_backload_to_gpu_"):
        return "policy_logprob"
    if reason.startswith("policy_offload_to_cpu_"):
        return "policy_offload"
    if reason.startswith("policy_optim_step_"):
        return "optimizer"
    if reason.startswith("policy_forward_backward") or reason.startswith("policy_micro_"):
        return "policy_train"
    return "other"


def step_events(policy_events: list[Event], reason: str, key: str) -> dict[int, Event]:
    out: dict[int, Event] = {}
    for event in policy_events:
        if event.reason != reason:
            continue
        idx = event.extra.get(key)
        if isinstance(idx, int):
            out[idx] = event
    return out


def nearest_before(events: list[Event], reason: str, before_ns: int) -> Event | None:
    matches = [e for e in events if e.reason == reason and e.timestamp_ns < before_ns]
    return matches[-1] if matches else None


def first_after(events: list[Event], reason: str, after_ns: int, before_ns: int | None = None) -> Event | None:
    for event in events:
        if event.reason != reason:
            continue
        if event.timestamp_ns <= after_ns:
            continue
        if before_ns is not None and event.timestamp_ns >= before_ns:
            continue
        return event
    return None


def build_stage_spans(policy_events: list[Event], ref_events: list[Event], step_idx: int) -> list[StageSpan]:
    train_start = step_events(policy_events, "policy_forward_backward_start", "forward_backward_call_index")[step_idx]
    train_end = step_events(policy_events, "policy_forward_backward_end", "forward_backward_call_index")[step_idx]
    optim_start = step_events(policy_events, "policy_optim_step_start", "optim_step_call_index")[step_idx]
    optim_end = step_events(policy_events, "policy_optim_step_end", "optim_step_call_index")[step_idx]

    ref_end = nearest_before(ref_events, "ref_offload_to_cpu_end", train_start.timestamp_ns)
    policy_logprob_end = nearest_before(policy_events, "policy_forward_micro_post_to_cpu", train_start.timestamp_ns)
    ref_start = nearest_before(
        ref_events,
        "ref_backload_to_gpu_start",
        ref_end.timestamp_ns if ref_end else train_start.timestamp_ns,
    )
    policy_logprob_start = nearest_before(
        policy_events,
        "policy_backload_to_gpu_start",
        policy_logprob_end.timestamp_ns if policy_logprob_end else train_start.timestamp_ns,
    )

    spans: list[StageSpan] = []
    if ref_start and ref_end and ref_start.timestamp_ns < ref_end.timestamp_ns:
        spans.append(StageSpan("ref_logprob", ref_start.timestamp_ns, ref_end.timestamp_ns, "#e6f2ff"))
    if (
        policy_logprob_start
        and policy_logprob_end
        and policy_logprob_start.timestamp_ns < policy_logprob_end.timestamp_ns
    ):
        spans.append(
            StageSpan("policy_logprob", policy_logprob_start.timestamp_ns, policy_logprob_end.timestamp_ns, "#fff2cc")
        )
    spans.append(StageSpan("policy_train", train_start.timestamp_ns, train_end.timestamp_ns, "#f4cccc"))
    spans.append(StageSpan("optimizer", optim_start.timestamp_ns, optim_end.timestamp_ns, "#d9ead3"))
    return spans


def peak_training_step(policy_events: list[Event]) -> int:
    best_idx = -1
    best_active = -1.0
    for event in policy_events:
        if event.reason not in {"policy_micro_post_forward", "policy_micro_post_loss_build", "policy_micro_pre_backward"}:
            continue
        idx = event.extra.get("forward_backward_call_index")
        if not isinstance(idx, int):
            continue
        active = event.metrics["active_gib"]
        if active > best_active:
            best_active = active
            best_idx = idx
    if best_idx < 0:
        raise ValueError("Could not find peak policy training step")
    return best_idx


def to_relative_seconds(ts_ns: np.ndarray | int, origin_ns: int) -> np.ndarray | float:
    return (np.asarray(ts_ns) - origin_ns) / 1e9


def resample_stepwise(timestamps_ns: np.ndarray, values: np.ndarray, grid_ns: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(timestamps_ns, grid_ns, side="right") - 1
    indices = np.clip(indices, 0, len(timestamps_ns) - 1)
    return values[indices]


def plot_figure(
    output_path: Path,
    gpu_timelines: dict[tuple[str, int], dict[str, np.ndarray]],
    policy_rank0: list[Event],
    ref_rank0: list[Event],
    rank0_gpu_key: tuple[str, int],
    per_gpu_summary: list[dict[str, object]],
    stage_summary: list[dict[str, object]],
    dpi: int,
) -> None:
    rank0 = gpu_timelines[rank0_gpu_key]
    t0_ns = int(rank0["timestamp_ns"][0])
    full_t = to_relative_seconds(rank0["timestamp_ns"], t0_ns)

    step_end_events = step_events(policy_rank0, "policy_optim_step_end", "optim_step_call_index")
    peak_step = peak_training_step(policy_rank0)
    spans = build_stage_spans(policy_rank0, ref_rank0, peak_step)

    grid_ns = np.linspace(rank0["timestamp_ns"][0], rank0["timestamp_ns"][-1], 1400, dtype=np.int64)
    heat = []
    labels = []
    for host, local_rank in sorted(gpu_timelines):
        tl = gpu_timelines[(host, local_rank)]
        heat.append(resample_stepwise(tl["timestamp_ns"], tl["active_gib"], grid_ns))
        labels.append(f"{host.split('-')[-1]}:{local_rank}")
    heat_arr = np.vstack(heat)
    heat_t = to_relative_seconds(grid_ns, t0_ns)

    peak_row = next(row for row in per_gpu_summary if row["rank"] == 0)

    fig = plt.figure(figsize=(18, 13), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.0, 1.3])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(full_t, rank0["active_gib"], label="active", color="#1f77b4", linewidth=1.6)
    ax1.plot(full_t, rank0["reserved_gib"], label="reserved", color="#d62728", linewidth=1.2)
    ax1.plot(full_t, rank0["model_param_gib"], label="params", color="#2ca02c", linewidth=1.0)
    ax1.plot(full_t, rank0["optimizer_state_gib"], label="optimizer", color="#9467bd", linewidth=1.0)
    ax1.plot(full_t, rank0["grad_gib"], label="grads", color="#8c564b", linewidth=1.0)
    ax1.fill_between(
        full_t,
        rank0["active_gib"],
        rank0["reserved_gib"],
        color="#d62728",
        alpha=0.08,
        label="reserved-unallocated",
    )
    for step_idx, event in sorted(step_end_events.items()):
        x = to_relative_seconds(event.timestamp_ns, t0_ns)
        ax1.axvline(x, color="gray", linestyle="--", alpha=0.25, linewidth=0.8)
        ax1.text(x, ax1.get_ylim()[1] if ax1.get_ylim()[1] else 0, str(step_idx), fontsize=7, alpha=0.7)
    ax1.set_title("Rank0 GPU Total Memory Across 20 Steps (policy + ref on the same GPU)")
    ax1.set_ylabel("GiB")
    ax1.legend(loc="upper right", ncol=5, fontsize=8)
    ax1.grid(alpha=0.2)

    ax2 = fig.add_subplot(gs[1, 0])
    im = ax2.imshow(
        heat_arr,
        aspect="auto",
        origin="lower",
        extent=[heat_t[0], heat_t[-1], -0.5, len(labels) - 0.5],
        cmap="viridis",
    )
    ax2.set_title("Per-GPU Total Active Memory Heatmap")
    ax2.set_ylabel("GPU rank")
    ax2.set_xlabel("Seconds since first memory event")
    cbar = fig.colorbar(im, ax=ax2, pad=0.01)
    cbar.set_label("Active GiB")
    ax2.set_yticks(range(0, len(labels), 4))
    ax2.set_yticklabels([str(i) for i in range(0, len(labels), 4)])

    step_start = min(span.start_ns for span in spans) - int(2e9)
    step_end = max(span.end_ns for span in spans) + int(2e9)
    mask = (rank0["timestamp_ns"] >= step_start) & (rank0["timestamp_ns"] <= step_end)
    zoom_t = to_relative_seconds(rank0["timestamp_ns"][mask], step_start)
    zoom_policy = rank0["active_gib"][mask]
    zoom_reserved = rank0["reserved_gib"][mask]

    ax3 = fig.add_subplot(gs[2, 0])
    for span in spans:
        ax3.axvspan(
            to_relative_seconds(span.start_ns, step_start),
            to_relative_seconds(span.end_ns, step_start),
            color=span.color,
            alpha=0.35,
            label=span.label,
        )
    ax3.plot(zoom_t, zoom_policy, color="#1f77b4", linewidth=1.8, label="total active")
    ax3.plot(zoom_t, zoom_reserved, color="#d62728", linewidth=1.2, label="total reserved")
    for stage in stage_summary:
        if stage["scope"] != f"step_{peak_step}":
            continue
        x = to_relative_seconds(stage["peak_timestamp_ns"], step_start)
        y = stage["peak_active_gib"]
        ax3.scatter([x], [y], color="black", s=18, zorder=5)
        ax3.annotate(
            f"{stage['label']}\\n{y:.2f} GiB",
            xy=(x, y),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=8,
        )
    ax3.set_title(f"Peak Training Step Zoom (step {peak_step})")
    ax3.set_xlabel("Seconds within zoom window")
    ax3.set_ylabel("GiB")
    handles, labels_seen = ax3.get_legend_handles_labels()
    uniq = dict(zip(labels_seen, handles))
    ax3.legend(uniq.values(), uniq.keys(), loc="upper left", ncol=6, fontsize=8)
    ax3.grid(alpha=0.2)

    fig.suptitle(
        "Qwen3-32B SkyRL/Harbor Full-Async Memory Trace\\n"
        f"Rank0 peak total active: {peak_row['peak_active_gib']:.2f} GiB at {peak_row['peak_reason']}",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def compute_per_gpu_summary(
    gpu_timelines: dict[tuple[str, int], dict[str, np.ndarray]],
    gpu_ranks: dict[tuple[str, int], int],
) -> list[dict[str, object]]:
    summary = []
    for (host, local_rank), timeline in sorted(gpu_timelines.items()):
        peak_i = int(np.argmax(timeline["active_gib"]))
        summary.append(
            {
                "rank": gpu_ranks[(host, local_rank)],
                "hostname": host,
                "local_rank": local_rank,
                "peak_active_gib": float(timeline["active_gib"][peak_i]),
                "peak_reserved_gib": float(timeline["reserved_gib"][peak_i]),
                "peak_timestamp_ns": int(timeline["timestamp_ns"][peak_i]),
            }
        )
    return summary


def stage_rows_for_step(
    policy_events: list[Event],
    ref_events: list[Event],
    rank0_timeline: dict[str, np.ndarray],
    step_idx: int,
) -> list[dict[str, object]]:
    spans = build_stage_spans(policy_events, ref_events, step_idx)
    rows = []
    for span in spans:
        mask = (rank0_timeline["timestamp_ns"] >= span.start_ns) & (rank0_timeline["timestamp_ns"] <= span.end_ns)
        peak_i = int(np.argmax(rank0_timeline["active_gib"][mask]))
        masked_idx = np.where(mask)[0][peak_i]
        rows.append(
            {
                "scope": f"step_{step_idx}",
                "label": span.label,
                "start_ns": span.start_ns,
                "end_ns": span.end_ns,
                "peak_active_gib": float(rank0_timeline["active_gib"][masked_idx]),
                "peak_reserved_gib": float(rank0_timeline["reserved_gib"][masked_idx]),
                "peak_timestamp_ns": int(rank0_timeline["timestamp_ns"][masked_idx]),
            }
        )
    return rows


def timeline_row_at_or_before(timeline: dict[str, np.ndarray], ts_ns: int) -> dict[str, float]:
    idx = np.searchsorted(timeline["timestamp_ns"], ts_ns, side="right") - 1
    idx = int(np.clip(idx, 0, len(timeline["timestamp_ns"]) - 1))
    return {key: float(timeline[key][idx]) for key in timeline if key != "timestamp_ns"}


def compute_step_summary(
    policy_events: list[Event],
    ref_events: list[Event],
    rank0_timeline: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    optim_end = step_events(policy_events, "policy_optim_step_end", "optim_step_call_index")
    rows = []
    for step_idx in sorted(optim_end):
        spans = build_stage_spans(policy_events, ref_events, step_idx)
        start_ns = min(span.start_ns for span in spans)
        end_ns = max(span.end_ns for span in spans)
        mask = (rank0_timeline["timestamp_ns"] >= start_ns) & (rank0_timeline["timestamp_ns"] <= end_ns)
        peak_i = int(np.argmax(rank0_timeline["active_gib"][mask]))
        masked_idx = np.where(mask)[0][peak_i]
        rows.append(
            {
                "step": step_idx,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_s": (end_ns - start_ns) / 1e9,
                "peak_active_gib": float(rank0_timeline["active_gib"][masked_idx]),
                "peak_reserved_gib": float(rank0_timeline["reserved_gib"][masked_idx]),
                "peak_timestamp_ns": int(rank0_timeline["timestamp_ns"][masked_idx]),
            }
        )
    return rows


def compute_stage_component_rows(
    stage_summary: list[dict[str, object]],
    rank0_timeline: dict[str, np.ndarray],
    peak_step: int,
) -> list[dict[str, object]]:
    rows = []
    for stage in stage_summary:
        if stage["scope"] != f"step_{peak_step}":
            continue
        snapshot = timeline_row_at_or_before(rank0_timeline, int(stage["peak_timestamp_ns"]))
        params = snapshot["model_param_gib"]
        optimizer = snapshot["optimizer_state_gib"]
        grads = snapshot["grad_gib"]
        other = max(0.0, snapshot["active_gib"] - params - optimizer - grads)
        rows.append(
            {
                "label": stage["label"],
                "active_gib": snapshot["active_gib"],
                "reserved_gib": snapshot["reserved_gib"],
                "params_gib": params,
                "optimizer_gib": optimizer,
                "grads_gib": grads,
                "other_active_gib": other,
            }
        )
    return rows


def assign_peak_reason(
    per_gpu_summary: list[dict[str, object]],
    policy_rank0: list[Event],
    ref_rank0: list[Event],
    gpu_timelines: dict[tuple[str, int], dict[str, np.ndarray]],
    rank0_gpu_key: tuple[str, int],
) -> None:
    event_lookup: dict[int, list[Event]] = defaultdict(list)
    for event in policy_rank0 + ref_rank0:
        event_lookup[event.timestamp_ns].append(event)
    rank0_timeline = gpu_timelines[rank0_gpu_key]
    peak_ts = int(rank0_timeline["timestamp_ns"][int(np.argmax(rank0_timeline["active_gib"]))])
    reasons = ",".join(sorted({event.reason for event in event_lookup.get(peak_ts, [])}))
    for row in per_gpu_summary:
        row["peak_reason"] = reasons if row["rank"] == 0 else ""


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    figure_path: Path,
    peak_step: int,
    per_gpu_summary: list[dict[str, object]],
    stage_summary: list[dict[str, object]],
) -> None:
    rank0 = next(row for row in per_gpu_summary if row["rank"] == 0)
    lines = [
        "# Qwen3-32B Memory Trace Summary",
        "",
        f"Figure: `{figure_path}`",
        "",
        "## What To Look At",
        "",
        "- Top panel: one physical GPU total memory trace (policy + ref summed together).",
        "- Middle panel: all 32 trainer GPUs as an active-memory heatmap.",
        f"- Bottom panel: the heaviest training step (`step {peak_step}`) with stage spans.",
        "",
        "## Key Numbers",
        "",
        f"- Peak total active memory on rank0 GPU: `{rank0['peak_active_gib']:.2f} GiB`.",
        f"- Peak total reserved memory on rank0 GPU: `{rank0['peak_reserved_gib']:.2f} GiB`.",
    ]
    for row in stage_summary:
        if row["scope"] != f"step_{peak_step}":
            continue
        lines.append(
            f"- `{row['label']}` peak in step {peak_step}: "
            f"`active={row['peak_active_gib']:.2f} GiB`, `reserved={row['peak_reserved_gib']:.2f} GiB`."
        )
    lines.extend(
        [
            "",
            "## Stage Interpretation",
            "",
            "- `ref_logprob`: ref model is backloaded, computes reference logprobs, then offloads again.",
            "- `policy_logprob`: policy model is backloaded only for rollout logprob collection before training.",
            "- `policy_train`: two micro-batches run forward -> loss build -> backward. This is the peak stage.",
            "- `optimizer`: gradients are scaled/applied; active memory drops while reserved memory mostly stays cached.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    default_log_root = Path("/home/hkang/zthunder_yagent/tmp_logs")
    if args.output_dir:
        output_dir = args.output_dir.resolve()
    elif run_dir.is_relative_to(default_log_root):
        output_dir = run_dir / "analysis"
    else:
        output_dir = default_log_root / run_dir.name / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    memory_dir = run_dir / "memory_events"
    all_events, by_file = load_events(memory_dir)
    gpu_streams = group_gpu_streams(all_events)
    gpu_timelines = {key: merged_gpu_timeline(streams) for key, streams in gpu_streams.items()}
    policy_rank0, ref_rank0 = find_rank0_files(by_file)
    rank0_gpu_key = (policy_rank0[0].hostname, policy_rank0[0].local_rank)
    gpu_ranks = {}
    for key, streams in gpu_streams.items():
        if streams["policy"]:
            gpu_ranks[key] = streams["policy"][0].rank
        elif streams["ref"]:
            gpu_ranks[key] = streams["ref"][0].rank
        else:
            raise ValueError(f"Empty GPU stream for {key}")
    peak_step = peak_training_step(policy_rank0)

    per_gpu_summary = compute_per_gpu_summary(gpu_timelines, gpu_ranks)
    assign_peak_reason(per_gpu_summary, policy_rank0, ref_rank0, gpu_timelines, rank0_gpu_key)
    stage_summary = stage_rows_for_step(policy_rank0, ref_rank0, gpu_timelines[rank0_gpu_key], peak_step)

    figure_path = output_dir / "memory_trace_overview.png"
    plot_figure(
        output_path=figure_path,
        gpu_timelines=gpu_timelines,
        policy_rank0=policy_rank0,
        ref_rank0=ref_rank0,
        rank0_gpu_key=rank0_gpu_key,
        per_gpu_summary=per_gpu_summary,
        stage_summary=stage_summary,
        dpi=args.dpi,
    )
    write_csv(output_dir / "per_gpu_peak_summary.csv", per_gpu_summary)
    write_csv(output_dir / "stage_peak_summary.csv", stage_summary)
    write_markdown(output_dir / "memory_trace_summary.md", figure_path, peak_step, per_gpu_summary, stage_summary)

    print(f"wrote {figure_path}")
    print(f"wrote {output_dir / 'per_gpu_peak_summary.csv'}")
    print(f"wrote {output_dir / 'stage_peak_summary.csv'}")
    print(f"wrote {output_dir / 'memory_trace_summary.md'}")


if __name__ == "__main__":
    main()
