# Implementation Summary: CodeContests 8xH100 Docker

## Overview

This implements the plan from `CODECONTESTS_8XH100_DOCKER_PLAN.md`, adding a dedicated run script and dataset subsetting support for running SkyRL + Harbor RL training on CodeContests with 8x H100 GPUs and local Docker sandboxes.

## Files Changed

### New file

- **`run_codecontest_8xh100_docker.sh`** — Main run script with 3-stage support:
  - `./run_codecontest_8xh100_docker.sh smoke` — Stage 1: verify the full loop (32 train / 10 eval tasks, batch=4, concurrency=8, `main_harbor_generate` entrypoint)
  - `./run_codecontest_8xh100_docker.sh pilot` — Stage 2: real RL updates (256 train / 20 eval tasks, batch=8, concurrency=16, `main_harbor` entrypoint)
  - `./run_codecontest_8xh100_docker.sh full` — Stage 3: full training (all tasks, batch=16, concurrency=32, `main_harbor` entrypoint)

### Modified files

- **`dataset.py`** — Added `max_tasks: Optional[int]` parameter to `HarborTaskDataset.__init__()`. When set, limits the dataset to the first N tasks. No behavior change when `None` (default).

- **`entrypoints/main_harbor.py`** — Added `max_train_tasks` and `max_eval_tasks` fields to `HarborSkyRLConfig`. Both are `Optional[int]`, default `None`, parseable from CLI (e.g., `max_train_tasks=32` or `max_train_tasks=null`). Wired through to `HarborTaskDataset` in `get_train_dataset()` and `get_eval_dataset()`.

- **`entrypoints/main_harbor_generate.py`** — Same wiring: `get_train_dataset()` passes `max_tasks=self.cfg.max_train_tasks`.

## Key Configuration

### GPU Placement (4 + 4 split)

| Pool | GPUs | Role |
|------|------|------|
| Training | 4 | Policy + Ref (colocated, 0.75 + 0.25 fractional GPU) |
| Rollout | 4 | 4x vLLM engines (TP=1, local, async) |

Settings:
- `trainer.placement.colocate_all=false`
- `trainer.placement.colocate_policy_ref=true`
- `trainer.algorithm.use_kl_loss=true` (activates ref model)
- `trainer.algorithm.kl_loss_coef=0.001`
- `trainer.critic.model.path=null` (no critic for GRPO)

### Docker Sandbox Overrides

The existing `default.yaml` targets Daytona. The run script overrides individual keys instead of trying to replace the entire `kwargs` dict (which would fail due to `_deep_merge` recursive behavior with empty dicts):

- `harbor_trial_config.environment.type=docker`
- `harbor_trial_config.environment.override_cpus=2`
- `harbor_trial_config.environment.override_memory_mb=4096`
- `harbor_trial_config.environment.override_storage_mb=4096`
- `harbor_trial_config.environment.kwargs.auto_stop_interval_mins=null` (nullifies Daytona-specific key)

### Settings Carried Forward

All required settings from the existing `run_codecontest.sh` are preserved:

- Chat template: `qwen3_acc_thinking.jinja2`
- HTTP endpoint: `127.0.0.1:8000` (required for Harbor agent LiteLLM calls)
- `async_engine=true`, `weight_sync_backend=nccl`, `max_model_len=32768`
- `apply_overlong_filtering=true`
- Dr. GRPO: `loss_reduction=seq_mean_token_sum_norm`, `grpo_norm_by_std=false`

## Feasibility Notes

### Confirmed working

- `colocate_all=false` + `colocate_policy_ref=true` is a supported configuration. Ray creates separate placement groups for training (4 GPUs) and inference (4 GPUs) on the same node.
- `use_kl_loss=true` correctly instantiates the ref model (`trainer.py:388`), runs ref forward passes (`trainer.py:944-947`), and computes KL loss (`worker.py:880-892`).
- Docker environment type is supported by Harbor. Overriding individual keys via CLI + `_deep_merge` works correctly.
- Qwen3-8B (~16GB bf16) with FSDP2 across 4 H100s fits comfortably with `micro_*_batch_size_per_gpu=1`.

### Issue found and fixed

The plan proposed `harbor_trial_config.environment.kwargs={}` to clear Daytona defaults. This does **not** work because `_deep_merge` recurses into both dicts — an empty override dict has no keys to iterate, so the base dict survives unchanged. The fix: override the specific key `harbor_trial_config.environment.kwargs.auto_stop_interval_mins=null`.

## Prerequisites

```bash
# Prepare datasets
uv run examples/train_integrations/harbor/prepare_harbor_dataset.py --dataset open-thoughts/CodeContests
uv run examples/train_integrations/harbor/prepare_harbor_dataset.py --dataset open-thoughts/OpenThoughts-TB-dev

# Set wandb key (for pilot/full stages)
export WANDB_API_KEY=YOUR_KEY_HERE
```

## Tests Performed

| Test | Result |
|------|--------|
| Shell script syntax (`bash -n`) | Pass |
| `HarborTaskDataset` with `max_tasks` (full, limited, over-limit, None) | Pass |
| `HarborSkyRLConfig` field defaults | Pass |
| CLI override parsing (`max_train_tasks=32`, `max_train_tasks=null`) | Pass |
| Harbor Docker config deep merge (`environment.type=docker`, `auto_stop_interval_mins=null`) | Pass |
| Stage argument handling (smoke/pilot/full/invalid) | Pass |
