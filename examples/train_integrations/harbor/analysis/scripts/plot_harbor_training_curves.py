#!/usr/bin/env python3
"""Plot key training curves from TensorBoard scalars for Harbor runs."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# TensorBoard is installed in the project venv. Matplotlib is available in the
# system python used for ad-hoc plotting, so import tensorboard from the venv.
VENV_SITE_PACKAGES = Path(
    "/home/hkang/zthunder_yagent/SkyRL/.venv/lib/python3.13/site-packages"
)
if str(VENV_SITE_PACKAGES) not in sys.path:
    sys.path.append(str(VENV_SITE_PACKAGES))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


BYTES_PER_GIB = 1024 ** 3


def load_scalars(tb_dir: Path) -> dict[str, list[tuple[int, float]]]:
    ea = event_accumulator.EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in ea.Tags().get("scalars", []):
        out[tag] = [(int(ev.step), float(ev.value)) for ev in ea.Scalars(tag)]
    return out


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scalar_rows(data: dict[str, list[tuple[int, float]]], tags: list[str]) -> list[dict[str, object]]:
    steps = sorted({step for tag in tags for step, _ in data.get(tag, [])})
    by_tag = {tag: dict(data.get(tag, [])) for tag in tags}
    rows = []
    for step in steps:
        row: dict[str, object] = {"step": step}
        for tag in tags:
            row[tag] = by_tag[tag].get(step)
        rows.append(row)
    return rows


def save_fig(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=180)
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


def plot_gpu_overview(data: dict[str, list[tuple[int, float]]], out_base: Path, csv_dir: Path) -> None:
    tags = [
        "monitor/gpu/all/memory_used_mib_sum",
        "monitor/process/rollout/gpu_memory_used_mib_sum",
        "monitor/process/trainer/gpu_memory_used_mib_sum",
        "monitor/vllm/kv_cache_size_tokens",
    ]
    rows = scalar_rows(data, tags)
    write_csv(csv_dir / "gpu_memory_overview.csv", rows, ["step", *tags])

    fig, ax1 = plt.subplots(figsize=(12, 6))
    for tag, label in [
        ("monitor/gpu/all/memory_used_mib_sum", "all_gpus_used_mib_sum"),
        ("monitor/process/rollout/gpu_memory_used_mib_sum", "rollout_process_mib_sum"),
        ("monitor/process/trainer/gpu_memory_used_mib_sum", "trainer_process_mib_sum"),
    ]:
        vals = data.get(tag, [])
        if vals:
            ax1.plot([x for x, _ in vals], [y for _, y in vals], label=label)
    ax1.set_title("GPU Memory Overview")
    ax1.set_xlabel("monitor step")
    ax1.set_ylabel("MiB")
    ax1.grid(alpha=0.3)

    kv_vals = data.get("monitor/vllm/kv_cache_size_tokens", [])
    if kv_vals:
        ax2 = ax1.twinx()
        ax2.plot([x for x, _ in kv_vals], [y for _, y in kv_vals], color="black", linestyle="--", label="vllm_kv_cache_tokens")
        ax2.set_ylabel("tokens")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    else:
        ax1.legend(loc="upper left")

    save_fig(fig, out_base)


def plot_training_memory_breakdown(
    data: dict[str, list[tuple[int, float]]], out_base: Path, csv_dir: Path
) -> None:
    tags = [
        "memory/after_train/policy/model_param_bytes_cuda_mean",
        "memory/after_train/policy/optimizer_state_bytes_cuda_mean",
        "memory/after_train/policy/grad_bytes_cuda_mean",
        "memory/after_train/policy/cuda_reserved_unallocated_bytes_mean",
        "memory/after_train/policy/cuda_allocated_bytes_mean",
        "memory/after_train/policy/cuda_reserved_bytes_mean",
    ]
    rows = scalar_rows(data, tags)
    write_csv(csv_dir / "training_memory_breakdown.csv", rows, ["step", *tags])

    fig, ax = plt.subplots(figsize=(12, 6))
    for tag, label in [
        ("memory/after_train/policy/model_param_bytes_cuda_mean", "model_param_gib"),
        ("memory/after_train/policy/optimizer_state_bytes_cuda_mean", "optimizer_state_gib"),
        ("memory/after_train/policy/grad_bytes_cuda_mean", "grad_gib"),
        ("memory/after_train/policy/cuda_reserved_unallocated_bytes_mean", "reserved_unallocated_gib"),
        ("memory/after_train/policy/cuda_allocated_bytes_mean", "allocated_gib"),
        ("memory/after_train/policy/cuda_reserved_bytes_mean", "reserved_gib"),
    ]:
        vals = data.get(tag, [])
        if vals:
            ax.plot([x for x, _ in vals], [y / BYTES_PER_GIB for _, y in vals], label=label)
    ax.set_title("Policy Memory Breakdown After Train")
    ax.set_xlabel("global step")
    ax.set_ylabel("GiB")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    save_fig(fig, out_base)


def plot_loss_reward(data: dict[str, list[tuple[int, float]]], out_base: Path, csv_dir: Path) -> None:
    tags = [
        "policy/final_loss",
        "policy/policy_loss",
        "policy/policy_kl",
        "policy/grad_norm",
        "reward/avg_pass_at_2",
        "reward/avg_raw_reward",
        "reward/mean_positive_reward",
        "loss/avg_final_rewards",
    ]
    rows = scalar_rows(data, tags)
    write_csv(csv_dir / "loss_reward_curves.csv", rows, ["step", *tags])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for tag, label in [
        ("policy/final_loss", "final_loss"),
        ("policy/policy_loss", "policy_loss"),
    ]:
        vals = data.get(tag, [])
        if vals:
            ax1.plot([x for x, _ in vals], [y for _, y in vals], label=label)
    ax1b = ax1.twinx()
    for tag, label, style in [
        ("policy/policy_kl", "policy_kl", "--"),
        ("policy/grad_norm", "grad_norm", ":"),
    ]:
        vals = data.get(tag, [])
        if vals:
            ax1b.plot([x for x, _ in vals], [y for _, y in vals], linestyle=style, label=label)
    ax1.set_title("Loss Curves")
    ax1.set_ylabel("loss")
    ax1b.set_ylabel("kl / grad_norm")
    ax1.grid(alpha=0.3)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    for tag, label in [
        ("reward/avg_pass_at_2", "avg_pass_at_2"),
        ("reward/avg_raw_reward", "avg_raw_reward"),
        ("reward/mean_positive_reward", "mean_positive_reward"),
        ("loss/avg_final_rewards", "avg_final_rewards"),
    ]:
        vals = data.get(tag, [])
        if vals:
            ax2.plot([x for x, _ in vals], [y for _, y in vals], label=label)
    ax2.set_title("Reward Curves")
    ax2.set_xlabel("global step")
    ax2.set_ylabel("value")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left")
    save_fig(fig, out_base)


def plot_eval_curves(data: dict[str, list[tuple[int, float]]], out_base: Path, csv_dir: Path) -> None:
    tags = [
        "eval/all/pass_at_1",
        "eval/all/avg_score",
        "eval/all/mean_positive_reward",
    ]
    rows = scalar_rows(data, tags)
    write_csv(csv_dir / "eval_curves.csv", rows, ["step", *tags])

    fig, ax = plt.subplots(figsize=(10, 5))
    for tag, label in [
        ("eval/all/pass_at_1", "pass_at_1"),
        ("eval/all/avg_score", "avg_score"),
        ("eval/all/mean_positive_reward", "mean_positive_reward"),
    ]:
        vals = data.get(tag, [])
        if vals:
            ax.plot([x for x, _ in vals], [y for _, y in vals], marker="o", label=label)
    ax.set_title("Eval Curves")
    ax.set_xlabel("global step")
    ax.set_ylabel("value")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    save_fig(fig, out_base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensorboard-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    tb_dir = Path(args.tensorboard_dir)
    output_dir = Path(args.output_dir)
    csv_dir = output_dir / "csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    data = load_scalars(tb_dir)
    plot_gpu_overview(data, output_dir / "gpu_memory_overview", csv_dir)
    plot_training_memory_breakdown(data, output_dir / "training_memory_breakdown_after_train", csv_dir)
    plot_loss_reward(data, output_dir / "loss_reward_curves", csv_dir)
    plot_eval_curves(data, output_dir / "eval_curves", csv_dir)


if __name__ == "__main__":
    main()
