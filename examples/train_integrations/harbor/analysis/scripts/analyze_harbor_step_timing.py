from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt


TIMING_ORDER = [
    ("session_alive_ms", "Session alive"),
    ("episode_logging_setup_ms", "Episode log setup"),
    ("llm_request_ms", "LLM request"),
    ("response_parse_ms", "Response parse"),
    ("declared_command_wait_budget_ms", "Declared cmd wait"),
    ("command_dispatch_and_wait_ms", "Command dispatch/wait"),
    ("terminal_capture_ms", "Terminal capture"),
    ("observation_build_ms", "Observation build"),
    ("step_record_build_ms", "Step record build"),
]

STEP_TOTAL_KEY = "step_total_pre_dump_ms"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze fine-grained Harbor Terminus step timings from trajectory.json files."
    )
    parser.add_argument("--run-dir", required=True, help="Harbor run artifact dir containing trials_run/")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for plots/reports. Defaults to <run-dir>/analysis",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Maximum episode/step index to include in step-aligned plots.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def iter_trial_dirs(trials_root: Path):
    if not trials_root.exists():
        return
    for child in sorted(trials_root.iterdir()):
        if child.is_dir():
            yield child


def reward_from_result(result_data: dict | None) -> float | None:
    if not isinstance(result_data, dict):
        return None
    rewards = (result_data.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward")
    if isinstance(reward, (int, float)):
        return float(reward)
    return None


def collect_step_rows(run_dir: Path) -> tuple[list[dict], dict]:
    trials_root = run_dir / "trials_run"
    rows: list[dict] = []
    trial_lengths: list[int] = []
    rewards: list[float] = []

    for trial_dir in iter_trial_dirs(trials_root):
        trajectory_data = load_json(trial_dir / "agent" / "trajectory.json")
        if not isinstance(trajectory_data, dict):
            continue

        result_data = load_json(trial_dir / "result.json")
        reward = reward_from_result(result_data)
        if reward is not None:
            rewards.append(reward)

        agent_steps = [
            step
            for step in (trajectory_data.get("steps") or [])
            if isinstance(step, dict) and step.get("source") == "agent"
        ]
        trial_lengths.append(len(agent_steps))

        for step_idx, step in enumerate(agent_steps, start=1):
            extra = step.get("extra") or {}
            timing = extra.get("timing_ms") or {}
            if not isinstance(timing, dict):
                timing = {}

            row = {
                "trial_name": trial_dir.name,
                "step_index": step_idx,
                "reward": reward,
                "parsed_command_count": extra.get("parsed_command_count"),
                "timeout_occurred": bool(extra.get("timeout_occurred", False)),
                "is_task_complete": bool(extra.get("is_task_complete", False)),
            }
            for key, _label in TIMING_ORDER:
                value = timing.get(key)
                row[key] = float(value) / 1000.0 if isinstance(value, (int, float)) else 0.0

            total_value = timing.get(STEP_TOTAL_KEY)
            row[STEP_TOTAL_KEY] = float(total_value) / 1000.0 if isinstance(total_value, (int, float)) else 0.0
            rows.append(row)

    summary = {
        "trial_count": len(trial_lengths),
        "trial_step_count_mean": mean(trial_lengths) if trial_lengths else 0.0,
        "trial_step_count_median": median(trial_lengths) if trial_lengths else 0.0,
        "trial_step_count_max": max(trial_lengths) if trial_lengths else 0,
        "reward_histogram": Counter(rewards),
    }
    return rows, summary


def summarize_by_step(rows: list[dict], max_steps: int) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        step_index = int(row["step_index"])
        if step_index <= max_steps:
            grouped[step_index].append(row)

    summary_rows: list[dict] = []
    for step_index in sorted(grouped):
        step_rows = grouped[step_index]
        summary_row = {
            "step_index": step_index,
            "trial_count": len(step_rows),
            "avg_total_sec": mean(r[STEP_TOTAL_KEY] for r in step_rows),
            "avg_parsed_command_count": mean(
                float(r["parsed_command_count"])
                for r in step_rows
                if isinstance(r["parsed_command_count"], (int, float))
            )
            if any(isinstance(r["parsed_command_count"], (int, float)) for r in step_rows)
            else 0.0,
            "timeout_count": sum(1 for r in step_rows if r["timeout_occurred"]),
            "task_complete_count": sum(1 for r in step_rows if r["is_task_complete"]),
        }
        for key, _label in TIMING_ORDER:
            summary_row[key] = mean(r[key] for r in step_rows)
        summary_rows.append(summary_row)
    return summary_rows


def plot_step_stacked_bars(summary_rows: list[dict], output_path: Path) -> None:
    if not summary_rows:
        plot_no_data(output_path, "No agent step timing data")
        return

    fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)
    x = [row["step_index"] for row in summary_rows]
    bottom = [0.0] * len(summary_rows)
    colors = [
        "#4E79A7",
        "#59A14F",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#EDC948",
        "#B07AA1",
        "#FF9DA7",
        "#9C755F",
    ]

    for color, (key, label) in zip(colors, TIMING_ORDER):
        values = [row[key] for row in summary_rows]
        ax.bar(x, values, bottom=bottom, label=label, color=color, edgecolor="white", width=0.72)
        for i, value in enumerate(values):
            if value >= 1.0:
                ax.text(x[i], bottom[i] + value / 2.0, f"{value:.1f}s", ha="center", va="center", fontsize=8, color="white")
        bottom = [b + v for b, v in zip(bottom, values)]

    totals = [row["avg_total_sec"] for row in summary_rows]
    for idx, total in enumerate(totals):
        ax.text(x[idx], total + 0.8, f"total {total:.1f}s\nn={summary_rows[idx]['trial_count']}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Average Agent Step Timing Breakdown")
    ax.set_xlabel("Agent step index")
    ax.set_ylabel("Average wall time per step (seconds)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_step_heatmap(summary_rows: list[dict], output_path: Path) -> None:
    if not summary_rows:
        plot_no_data(output_path, "No agent step timing data")
        return

    labels = [label for _key, label in TIMING_ORDER] + ["Step total"]
    data = []
    for row in summary_rows:
        data.append([row[key] for key, _label in TIMING_ORDER] + [row["avg_total_sec"]])

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(summary_rows)))
    ax.set_yticklabels([f"step {row['step_index']}" for row in summary_rows])
    ax.set_title("Average Step Timing Heatmap (seconds)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("seconds")

    for row_idx, row in enumerate(summary_rows):
        values = [row[key] for key, _label in TIMING_ORDER] + [row["avg_total_sec"]]
        for col_idx, value in enumerate(values):
            ax.text(col_idx, row_idx, f"{value:.1f}", ha="center", va="center", fontsize=8, color="black")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_no_data(output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=18, weight="bold")
    ax.text(
        0.5,
        0.4,
        "All completed trials in this run recorded zero agent-source steps in trajectory.json.",
        ha="center",
        va="center",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(summary_rows: list[dict], overall_summary: dict, output_path: Path, max_steps: int) -> None:
    lines = [
        "# Harbor Step Timing Breakdown",
        "",
        f"- trials: {overall_summary['trial_count']}",
        f"- mean_agent_steps_per_trial: {overall_summary['trial_step_count_mean']:.2f}",
        f"- median_agent_steps_per_trial: {overall_summary['trial_step_count_median']:.2f}",
        f"- max_agent_steps_per_trial: {overall_summary['trial_step_count_max']}",
        f"- plotted_steps: 1..{max_steps}",
        "",
        "## Reward Histogram",
        "",
    ]

    reward_histogram = overall_summary["reward_histogram"]
    if reward_histogram:
        for reward_value, count in sorted(reward_histogram.items()):
            lines.append(f"- reward={reward_value:.1f}: {count}")
    else:
        lines.append("- no rewards recorded")

    lines.extend(
        [
            "",
            "## Step Table",
            "",
            "| step | trials | avg_total_s | session_alive | log_setup | llm | parse | declared_wait | dispatch_wait | terminal_capture | observation_build | record_build | avg_commands | timeouts | task_complete |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in summary_rows:
        lines.append(
            "| {step} | {trial_count} | {avg_total_sec:.2f} | {session_alive_ms:.2f} | {episode_logging_setup_ms:.2f} | {llm_request_ms:.2f} | {response_parse_ms:.2f} | {declared_command_wait_budget_ms:.2f} | {command_dispatch_and_wait_ms:.2f} | {terminal_capture_ms:.2f} | {observation_build_ms:.2f} | {step_record_build_ms:.2f} | {avg_parsed_command_count:.2f} | {timeout_count} | {task_complete_count} |".format(
                step=row["step_index"],
                **row,
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `declared_command_wait_budget_ms` is the duration budget requested by the agent for `bash_command` tool calls.",
            "- `command_dispatch_and_wait_ms` is the measured wall time spent dispatching commands and waiting inside Terminus.",
            "- `step_total_pre_dump_ms` is the best per-step wall clock total before trajectory dump.",
        ]
    )

    output_path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, overall_summary = collect_step_rows(run_dir)
    summary_rows = summarize_by_step(rows, args.max_steps)

    plot_step_stacked_bars(summary_rows, output_dir / "trial_step_timing_breakdown.png")
    plot_step_heatmap(summary_rows, output_dir / "trial_step_timing_heatmap.png")
    write_report(summary_rows, overall_summary, output_dir / "trial_step_timing_report.md", args.max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
