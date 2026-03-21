#!/usr/bin/env python3
"""Generate a more readable memory figure for the Harbor 32B run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.train_integrations.harbor.analyze_memory_trace import (
    assign_peak_reason,
    build_stage_spans,
    compute_per_gpu_summary,
    compute_stage_component_rows,
    compute_step_summary,
    find_rank0_files,
    group_gpu_streams,
    load_events,
    merged_gpu_timeline,
    peak_training_step,
    stage_rows_for_step,
    to_relative_seconds,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def default_output_dir(run_dir: Path) -> Path:
    log_root = Path("/home/hkang/zthunder_yagent/tmp_logs")
    if run_dir.is_relative_to(log_root):
        return run_dir / "analysis"
    return log_root / run_dir.name / "analysis"


def stage_order_key(label: str) -> int:
    order = {
        "ref_logprob": 0,
        "policy_logprob": 1,
        "policy_train": 2,
        "optimizer": 3,
    }
    return order.get(label, 99)


def write_summary(
    path: Path,
    figure_path: Path,
    peak_step: int,
    per_gpu_summary: list[dict[str, object]],
    step_summary: list[dict[str, object]],
    stage_components: list[dict[str, object]],
) -> None:
    peak_row = next(row for row in per_gpu_summary if row["rank"] == 0)
    peak_step_row = next(row for row in step_summary if row["step"] == peak_step)
    lines = [
        "# Readable Memory Summary",
        "",
        f"Figure: `{figure_path}`",
        "",
        "## Read Order",
        "",
        "- Top: each training step's peak active and reserved memory.",
        f"- Middle: the heaviest step (`step {peak_step}`) laid out in time order.",
        "- Bottom: what each stage peak is made of.",
        "",
        "## Key Numbers",
        "",
        f"- Heaviest step: `{peak_step}`.",
        f"- Peak active memory: `{peak_row['peak_active_gib']:.2f} GiB`.",
        f"- Peak reserved memory: `{peak_row['peak_reserved_gib']:.2f} GiB`.",
        f"- Heaviest step duration: `{peak_step_row['duration_s']:.2f}s`.",
    ]
    for row in stage_components:
        lines.append(
            f"- `{row['label']}`: active `{row['active_gib']:.2f} GiB`, "
            f"reserved `{row['reserved_gib']:.2f} GiB`, "
            f"params `{row['params_gib']:.2f}`, optimizer `{row['optimizer_gib']:.2f}`, "
            f"grads `{row['grads_gib']:.2f}`, other `{row['other_active_gib']:.2f}`."
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else default_output_dir(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_events, by_file = load_events(run_dir / "memory_events")
    gpu_streams = group_gpu_streams(all_events)
    gpu_timelines = {key: merged_gpu_timeline(streams) for key, streams in gpu_streams.items()}
    policy_rank0, ref_rank0 = find_rank0_files(by_file)
    rank0_gpu_key = (policy_rank0[0].hostname, policy_rank0[0].local_rank)
    gpu_ranks = {}
    for key, streams in gpu_streams.items():
        if streams["policy"]:
            gpu_ranks[key] = streams["policy"][0].rank
        else:
            gpu_ranks[key] = streams["ref"][0].rank

    rank0_timeline = gpu_timelines[rank0_gpu_key]
    peak_step = peak_training_step(policy_rank0)
    step_summary = compute_step_summary(policy_rank0, ref_rank0, rank0_timeline)
    per_gpu_summary = compute_per_gpu_summary(gpu_timelines, gpu_ranks)
    assign_peak_reason(per_gpu_summary, policy_rank0, ref_rank0, gpu_timelines, rank0_gpu_key)
    stage_summary = stage_rows_for_step(policy_rank0, ref_rank0, rank0_timeline, peak_step)
    stage_components = compute_stage_component_rows(stage_summary, rank0_timeline, peak_step)
    stage_components.sort(key=lambda row: stage_order_key(str(row["label"])))

    peak_row = next(row for row in per_gpu_summary if row["rank"] == 0)
    spans = build_stage_spans(policy_rank0, ref_rank0, peak_step)

    step_x = [row["step"] for row in step_summary]
    step_active = [row["peak_active_gib"] for row in step_summary]
    step_reserved = [row["peak_reserved_gib"] for row in step_summary]
    step_duration = [row["duration_s"] for row in step_summary]

    step_start_ns = min(span.start_ns for span in spans) - int(2e9)
    step_end_ns = max(span.end_ns for span in spans) + int(2e9)
    mask = (rank0_timeline["timestamp_ns"] >= step_start_ns) & (rank0_timeline["timestamp_ns"] <= step_end_ns)
    zoom_t = to_relative_seconds(rank0_timeline["timestamp_ns"][mask], step_start_ns)
    zoom_active = rank0_timeline["active_gib"][mask]
    zoom_reserved = rank0_timeline["reserved_gib"][mask]

    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.9, 1.15, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(step_x, step_active, color="#1f77b4", marker="o", linewidth=2.1, label="peak active")
    ax1.plot(step_x, step_reserved, color="#d62728", marker="o", linewidth=2.1, label="peak reserved")
    ax1.fill_between(step_x, step_active, step_reserved, color="#f4cccc", alpha=0.45, label="cached but idle")
    peak_step_row = next(row for row in step_summary if row["step"] == peak_step)
    ax1.scatter([peak_step], [peak_step_row["peak_active_gib"]], color="black", s=36, zorder=5)
    ax1.annotate(
        f"step {peak_step}\n{peak_step_row['peak_active_gib']:.2f} GiB",
        xy=(peak_step, peak_step_row["peak_active_gib"]),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=9,
    )
    ax1b = ax1.twinx()
    ax1b.bar(step_x, step_duration, width=0.45, color="#d9ead3", alpha=0.4, label="step duration")
    ax1b.set_ylabel("Step duration (s)")
    ax1b.set_ylim(0, max(step_duration) * 1.25)
    ax1.set_title("1. Peak Memory Per Training Step")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("GiB")
    ax1.set_xticks(step_x)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", ncol=4, fontsize=9)
    ax1.grid(alpha=0.2)

    ax2 = fig.add_subplot(gs[1, 0])
    for span in spans:
        x0 = to_relative_seconds(span.start_ns, step_start_ns)
        x1 = to_relative_seconds(span.end_ns, step_start_ns)
        ax2.axvspan(x0, x1, color=span.color, alpha=0.6)
        ax2.text((x0 + x1) / 2, max(zoom_reserved) * 1.02, span.label, ha="center", va="bottom", fontsize=10)
    ax2.plot(zoom_t, zoom_active, color="#1f77b4", linewidth=2.2, label="total active")
    ax2.plot(zoom_t, zoom_reserved, color="#d62728", linewidth=1.8, label="total reserved")
    for row in stage_summary:
        x = to_relative_seconds(row["peak_timestamp_ns"], step_start_ns)
        y = row["peak_active_gib"]
        ax2.scatter([x], [y], color="black", s=24, zorder=5)
        ax2.annotate(
            f"{row['label']}\n{y:.2f} GiB",
            xy=(x, y),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=9,
        )
    ax2.set_title(f"2. Time Order Inside The Heaviest Step (step {peak_step})")
    ax2.set_xlabel("Seconds within the step window")
    ax2.set_ylabel("GiB")
    ax2.set_ylim(0, max(zoom_reserved) * 1.12)
    ax2.legend(loc="upper left", ncol=2, fontsize=9)
    ax2.grid(alpha=0.2)

    ax3 = fig.add_subplot(gs[2, 0])
    labels = [str(row["label"]) for row in stage_components]
    xpos = np.arange(len(labels))
    params = np.array([row["params_gib"] for row in stage_components])
    optimizer = np.array([row["optimizer_gib"] for row in stage_components])
    grads = np.array([row["grads_gib"] for row in stage_components])
    other = np.array([row["other_active_gib"] for row in stage_components])
    reserved = np.array([row["reserved_gib"] for row in stage_components])

    ax3.bar(xpos, params, color="#2ca02c", label="params")
    ax3.bar(xpos, optimizer, bottom=params, color="#9467bd", label="optimizer")
    ax3.bar(xpos, grads, bottom=params + optimizer, color="#8c564b", label="grads")
    ax3.bar(xpos, other, bottom=params + optimizer + grads, color="#1f77b4", label="other active")
    ax3.plot(xpos, reserved, color="#d62728", marker="o", linewidth=2.0, label="reserved")
    for i, row in enumerate(stage_components):
        ax3.annotate(
            f"active {row['active_gib']:.2f}\nreserved {row['reserved_gib']:.2f}",
            xy=(i, row["reserved_gib"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax3.set_title(f"3. What Each Stage Peak Is Made Of (step {peak_step})")
    ax3.set_xticks(xpos)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel("GiB")
    ax3.legend(loc="upper left", ncol=5, fontsize=9)
    ax3.grid(alpha=0.2, axis="y")

    fig.suptitle(
        "Qwen3-32B SkyRL/Harbor Full-Async Memory Use\n"
        f"Rank0 peak active = {peak_row['peak_active_gib']:.2f} GiB; all 32 trainer GPUs are effectively identical",
        fontsize=14,
    )

    figure_path = output_dir / "memory_trace_overview.png"
    fig.savefig(figure_path, dpi=args.dpi)
    plt.close(fig)

    write_csv(output_dir / "step_peak_summary.csv", step_summary)
    write_summary(
        output_dir / "memory_trace_summary.md",
        figure_path,
        peak_step,
        per_gpu_summary,
        step_summary,
        stage_components,
    )
    print(f"wrote {figure_path}")
    print(f"wrote {output_dir / 'step_peak_summary.csv'}")
    print(f"wrote {output_dir / 'memory_trace_summary.md'}")


if __name__ == "__main__":
    main()
