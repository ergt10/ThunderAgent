from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _stringify_message(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        parts: list[str] = []
        for item in message:
            if isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"])
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, indent=2))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return json.dumps(message, ensure_ascii=False, indent=2)


def _extract_observation_text(step: dict[str, Any]) -> str:
    observation = step.get("observation") or {}
    chunks: list[str] = []
    for result in observation.get("results") or []:
        content = result.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif content is not None:
            chunks.append(json.dumps(content, ensure_ascii=False, indent=2))
    return "\n\n".join(chunks)


def _format_tool_calls(step: dict[str, Any]) -> str:
    tool_calls = step.get("tool_calls") or []
    if not tool_calls:
        return ""
    lines = ["### Tool Calls", ""]
    for idx, tool_call in enumerate(tool_calls, start=1):
        function_name = tool_call.get("function_name", "unknown")
        lines.append(f"{idx}. `{function_name}`")
        arguments = tool_call.get("arguments") or {}
        if arguments:
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(arguments, ensure_ascii=False, indent=2))
            lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _format_metrics(step: dict[str, Any]) -> str:
    metrics = step.get("metrics") or {}
    if not metrics:
        return ""
    fields = [
        ("prompt_tokens", metrics.get("prompt_tokens")),
        ("completion_tokens", metrics.get("completion_tokens")),
        ("cached_tokens", metrics.get("cached_tokens")),
    ]
    parts = [f"{name}={value}" for name, value in fields if value is not None]
    if not parts:
        return ""
    return "### Metrics\n\n- " + "\n- ".join(parts) + "\n"


def render_trial(trial_dir: Path) -> str:
    result = _read_json(trial_dir / "result.json")
    trajectory = _read_json(trial_dir / "agent" / "trajectory.json")

    reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    exception_info = result.get("exception_info") or {}

    lines: list[str] = [
        f"# Harbor Trial Transcript: {result.get('trial_name', trial_dir.name)}",
        "",
        "## Summary",
        "",
        f"- task_name: {result.get('task_name')}",
        f"- trial_name: {result.get('trial_name')}",
        f"- started_at: {result.get('started_at')}",
        f"- finished_at: {result.get('finished_at')}",
        f"- reward: {reward}",
        f"- exception_type: {exception_info.get('exception_type')}",
        "",
    ]

    timings = [
        ("environment_setup", result.get("environment_setup")),
        ("agent_setup", result.get("agent_setup")),
        ("agent_execution", result.get("agent_execution")),
        ("verifier", result.get("verifier")),
    ]
    lines.extend(["## Trial Timings", ""])
    for name, value in timings:
        if isinstance(value, dict):
            lines.append(
                f"- {name}: {value.get('started_at')} -> {value.get('finished_at')}"
            )
    lines.append("")

    lines.extend(["## Trajectory", ""])
    for step in trajectory.get("steps", []):
        step_id = step.get("step_id")
        source = step.get("source")
        timestamp = step.get("timestamp")
        source_label = {
            "user": "USER",
            "agent": "ASSISTANT",
            "system": "SYSTEM",
        }.get(source, str(source).upper())
        lines.append(f"## {source_label} STEP {step_id}")
        lines.append("")
        lines.append(f"- source: {source_label}")
        lines.append(f"- timestamp: {timestamp}")
        model_name = step.get("model_name")
        if model_name:
            lines.append(f"- model_name: {model_name}")
        lines.append("")
        lines.append("### Message")
        lines.append("")
        lines.append("```text")
        lines.append(_stringify_message(step.get("message")))
        lines.append("```")
        lines.append("")

        tool_calls = _format_tool_calls(step)
        if tool_calls:
            lines.append(tool_calls.rstrip())
            lines.append("")

        observation_text = _extract_observation_text(step)
        if observation_text:
            lines.append("### Observation")
            lines.append("")
            lines.append("```text")
            lines.append(observation_text)
            lines.append("```")
            lines.append("")

        metrics_text = _format_metrics(step)
        if metrics_text:
            lines.append(metrics_text.rstrip())
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Harbor trial trajectory as readable markdown.")
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    markdown = render_trial(args.trial_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown)


if __name__ == "__main__":
    main()
