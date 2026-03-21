from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Iterable

import matplotlib.pyplot as plt


@dataclass
class TrialBreakdown:
    total_sec: float
    environment_setup_sec: float
    agent_setup_sec: float
    agent_execution_sec: float
    verifier_sec: float
    llm_api_sec: float
    command_wait_budget_sec: float
    agent_other_sec: float
    post_trial_other_sec: float
    reward: float | None


def _parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _duration_seconds(section: dict | None) -> float | None:
    if not isinstance(section, dict):
        return None
    start = _parse_iso8601(section.get("started_at"))
    end = _parse_iso8601(section.get("finished_at"))
    if start is None or end is None:
        return None
    return (end - start).total_seconds()


def _sum_llm_api_seconds(result_data: dict) -> float:
    metadata = ((result_data.get("agent_result") or {}).get("metadata") or {})
    request_times = metadata.get("api_request_times_msec") or []
    return sum(float(x) for x in request_times if isinstance(x, (int, float))) / 1000.0


def _sum_command_wait_budget_seconds(trial_dir: Path) -> float:
    trajectory_path = trial_dir / "agent" / "trajectory.json"
    if not trajectory_path.exists():
        return 0.0

    try:
        trajectory = json.loads(trajectory_path.read_text())
    except Exception:
        return 0.0

    total = 0.0
    for step in trajectory.get("steps", []):
        if step.get("source") != "agent":
            continue
        for tool_call in step.get("tool_calls") or []:
            if tool_call.get("function_name") != "bash_command":
                continue
            duration = (tool_call.get("arguments") or {}).get("duration")
            if isinstance(duration, (int, float)):
                total += float(duration)
    return total


def _load_trial_breakdown(trial_dir: Path) -> TrialBreakdown | None:
    result_path = trial_dir / "result.json"
    if not result_path.exists():
        return None

    try:
        result_data = json.loads(result_path.read_text())
    except Exception:
        return None

    started_at = _parse_iso8601(result_data.get("started_at"))
    finished_at = _parse_iso8601(result_data.get("finished_at"))
    if started_at is None or finished_at is None:
        return None

    total_sec = (finished_at - started_at).total_seconds()
    env_sec = _duration_seconds(result_data.get("environment_setup")) or 0.0
    agent_setup_sec = _duration_seconds(result_data.get("agent_setup")) or 0.0
    agent_exec_sec = _duration_seconds(result_data.get("agent_execution")) or 0.0
    verifier_sec = _duration_seconds(result_data.get("verifier")) or 0.0
    llm_api_sec = _sum_llm_api_seconds(result_data)
    command_wait_budget_sec = _sum_command_wait_budget_seconds(trial_dir)
    agent_other_sec = max(agent_exec_sec - llm_api_sec - command_wait_budget_sec, 0.0)
    post_trial_other_sec = max(
        total_sec - env_sec - agent_setup_sec - agent_exec_sec - verifier_sec,
        0.0,
    )
    reward = ((result_data.get("verifier_result") or {}).get("rewards") or {}).get("reward")

    return TrialBreakdown(
        total_sec=total_sec,
        environment_setup_sec=env_sec,
        agent_setup_sec=agent_setup_sec,
        agent_execution_sec=agent_exec_sec,
        verifier_sec=verifier_sec,
        llm_api_sec=llm_api_sec,
        command_wait_budget_sec=command_wait_budget_sec,
        agent_other_sec=agent_other_sec,
        post_trial_other_sec=post_trial_other_sec,
        reward=reward,
    )


def _mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return mean(values) if values else 0.0


def _summarize_trials(trials: list[TrialBreakdown]) -> dict[str, float]:
    return {
        "count": float(len(trials)),
        "total_sec": _mean_or_zero(t.total_sec for t in trials),
        "environment_setup_sec": _mean_or_zero(t.environment_setup_sec for t in trials),
        "agent_setup_sec": _mean_or_zero(t.agent_setup_sec for t in trials),
        "agent_execution_sec": _mean_or_zero(t.agent_execution_sec for t in trials),
        "verifier_sec": _mean_or_zero(t.verifier_sec for t in trials),
        "llm_api_sec": _mean_or_zero(t.llm_api_sec for t in trials),
        "command_wait_budget_sec": _mean_or_zero(t.command_wait_budget_sec for t in trials),
        "agent_other_sec": _mean_or_zero(t.agent_other_sec for t in trials),
        "post_trial_other_sec": _mean_or_zero(t.post_trial_other_sec for t in trials),
    }


def _pct(part: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return part / total * 100.0


def _draw_label(ax, left: float, width: float, y: float, text: str) -> None:
    if width < 4.0:
        return
    ax.text(left + width / 2.0, y, text, ha="center", va="center", fontsize=9, color="white")


def _plot_breakdown(all_summary: dict[str, float], successful_summary: dict[str, float], output_path: Path) -> None:
    total_components = [
        ("environment_setup_sec", "Env setup", "#4E79A7"),
        ("agent_setup_sec", "Agent setup", "#59A14F"),
        ("llm_api_sec", "LLM API", "#F28E2B"),
        ("command_wait_budget_sec", "Cmd wait budget", "#E15759"),
        ("agent_other_sec", "Agent other", "#B07AA1"),
        ("verifier_sec", "Verifier", "#76B7B2"),
        ("post_trial_other_sec", "Post-trial other", "#9C755F"),
    ]
    agent_components = [
        ("llm_api_sec", "LLM API", "#F28E2B"),
        ("command_wait_budget_sec", "Cmd wait budget", "#E15759"),
        ("agent_other_sec", "Agent other", "#B07AA1"),
    ]

    fig, (ax_total, ax_agent) = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True)

    summaries = [("All trials", all_summary), ("Successful only", successful_summary)]
    y_positions = [1, 0]

    for y, (label, summary) in zip(y_positions, summaries):
        left = 0.0
        total = summary["total_sec"]
        for key, legend_label, color in total_components:
            width = _pct(summary[key], total)
            ax_total.barh(y, width, left=left, color=color, edgecolor="white", height=0.48)
            _draw_label(
                ax_total,
                left,
                width,
                y,
                f"{legend_label}\n{summary[key]:.1f}s\n{width:.1f}%",
            )
            left += width

    ax_total.set_xlim(0, 100)
    ax_total.set_yticks(y_positions)
    ax_total.set_yticklabels([label for label, _ in summaries])
    ax_total.set_xlabel("Average share of total trial wall time (%)")
    ax_total.set_title("Average Harbor Trial Time Breakdown")
    ax_total.grid(axis="x", linestyle="--", alpha=0.35)
    ax_total.legend(
        [plt.Rectangle((0, 0), 1, 1, color=color) for _, _, color in total_components],
        [legend_label for _, legend_label, _ in total_components],
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
    )

    for y, (label, summary) in zip(y_positions, summaries):
        left = 0.0
        agent_total = summary["agent_execution_sec"]
        for key, legend_label, color in agent_components:
            width = _pct(summary[key], agent_total)
            ax_agent.barh(y, width, left=left, color=color, edgecolor="white", height=0.48)
            _draw_label(
                ax_agent,
                left,
                width,
                y,
                f"{legend_label}\n{summary[key]:.1f}s\n{width:.1f}%",
            )
            left += width

    ax_agent.set_xlim(0, 100)
    ax_agent.set_yticks(y_positions)
    ax_agent.set_yticklabels([label for label, _ in summaries])
    ax_agent.set_xlabel("Average share of agent_execution time (%)")
    ax_agent.set_title("Average Agent Execution Breakdown")
    ax_agent.grid(axis="x", linestyle="--", alpha=0.35)
    ax_agent.legend(
        [plt.Rectangle((0, 0), 1, 1, color=color) for _, _, color in agent_components],
        [legend_label for _, legend_label, _ in agent_components],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
    )

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_report(
    all_summary: dict[str, float],
    successful_summary: dict[str, float],
    output_path: Path,
) -> None:
    def section(name: str, summary: dict[str, float]) -> str:
        total = summary["total_sec"]
        agent_total = summary["agent_execution_sec"]
        lines = [
            f"## {name}",
            "",
            f"- trials_count: {int(summary['count'])}",
            f"- avg_total_sec: {total:.2f}",
            f"- environment_setup: {summary['environment_setup_sec']:.2f}s ({_pct(summary['environment_setup_sec'], total):.2f}%)",
            f"- agent_setup: {summary['agent_setup_sec']:.2f}s ({_pct(summary['agent_setup_sec'], total):.2f}%)",
            f"- agent_execution: {summary['agent_execution_sec']:.2f}s ({_pct(summary['agent_execution_sec'], total):.2f}%)",
            f"- verifier: {summary['verifier_sec']:.2f}s ({_pct(summary['verifier_sec'], total):.2f}%)",
            f"- post_trial_other: {summary['post_trial_other_sec']:.2f}s ({_pct(summary['post_trial_other_sec'], total):.2f}%)",
            "",
            f"- llm_api_inside_agent: {summary['llm_api_sec']:.2f}s ({_pct(summary['llm_api_sec'], total):.2f}% of total, {_pct(summary['llm_api_sec'], agent_total):.2f}% of agent_execution)",
            f"- command_wait_budget_inside_agent: {summary['command_wait_budget_sec']:.2f}s ({_pct(summary['command_wait_budget_sec'], total):.2f}% of total, {_pct(summary['command_wait_budget_sec'], agent_total):.2f}% of agent_execution)",
            f"- agent_other_inside_agent: {summary['agent_other_sec']:.2f}s ({_pct(summary['agent_other_sec'], total):.2f}% of total, {_pct(summary['agent_other_sec'], agent_total):.2f}% of agent_execution)",
            "",
        ]
        return "\n".join(lines)

    caveat = "\n".join(
        [
            "# Harbor Trial Stage Breakdown",
            "",
            "## Notes",
            "",
            "- `llm_api_inside_agent` comes from `agent_result.metadata.api_request_times_msec`.",
            "- `command_wait_budget_inside_agent` is the sum of per-step `bash_command.duration` values recorded in `agent/trajectory.json`.",
            "- `command_wait_budget_inside_agent` is an execution budget / requested wait, not a perfect wall-clock measurement of tool runtime.",
            "- `agent_other_inside_agent` is the residual: `agent_execution - llm_api - command_wait_budget`.",
            "",
            section("All Trials", all_summary),
            section("Successful Trials", successful_summary),
        ]
    )
    output_path.write_text(caveat)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze average Harbor trial stage breakdown.")
    parser.add_argument("--run-artifacts-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    trials_dir = args.run_artifacts_dir / "trials_run"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trials: list[TrialBreakdown] = []
    for trial_dir in sorted(trials_dir.iterdir()):
        if not trial_dir.is_dir():
            continue
        breakdown = _load_trial_breakdown(trial_dir)
        if breakdown is not None:
            trials.append(breakdown)

    successful_trials = [trial for trial in trials if trial.reward is not None]

    all_summary = _summarize_trials(trials)
    successful_summary = _summarize_trials(successful_trials)

    _plot_breakdown(
        all_summary,
        successful_summary,
        args.output_dir / "trial_stage_breakdown.png",
    )
    _write_report(
        all_summary,
        successful_summary,
        args.output_dir / "trial_stage_breakdown_report.md",
    )


if __name__ == "__main__":
    main()
