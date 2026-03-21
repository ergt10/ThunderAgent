# SkyRL + Harbor Training Plan for CodeContests on 8x H100

## Status Update (2026-03-07)

- `Stage 1: Smoke Run` 已完成
  - 详细记录见 [CODECONTESTS_8XH100_DOCKER_SMOKE_TEST_NOTES.md](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/CODECONTESTS_8XH100_DOCKER_SMOKE_TEST_NOTES.md)
- `Stage 2: Pilot Run` 已完成
  - 详细记录见 [CODECONTESTS_8XH100_DOCKER_PILOT_RUN_NOTES.md](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/CODECONTESTS_8XH100_DOCKER_PILOT_RUN_NOTES.md)
  - 实跑收敛后的关键偏离：
    - 训练侧 `trainer.algorithm.max_seq_len=8192`
    - 最终接受 run 使用 curated `256` 任务子集，排除了 `code_contests-0028`
    - checkpoint / HF export 验证 run 使用 `eval_before_train=false`、`ckpt_interval=1`、`hf_save_interval=1`
- `Stage 3: Full Run` 已完成第一版可接受基线
  - 详细记录见 [CODECONTESTS_8XH100_DOCKER_FULL_RUN_NOTES.md](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/CODECONTESTS_8XH100_DOCKER_FULL_RUN_NOTES.md)
  - 当前结论：
    - 已在全量 `CodeContests` 上连续完成 `20` 个 training step
    - `1 epoch` 长跑尚未完成
    - 当前主要阻塞不是训练逻辑，而是 `/home` 不适合继续承载 full run 的 checkpoint/export

## Summary

This document captures the current training requirements, available hardware, and the execution plan for running RL fine-tuning with the existing `SkyRL + Harbor` integration under `examples/train_integrations/harbor/`.

The intended setup is:

- Training framework: `SkyRL + Harbor`
- Train dataset: `open-thoughts/CodeContests`
- Eval dataset: `open-thoughts/OpenThoughts-TB-dev`
- Base model: `Qwen/Qwen3-8B`
- Sandbox provider: local `docker`
- Placement requirement:
  - `trainer.placement.colocate_all=false`
  - `trainer.placement.colocate_policy_ref=true`
  - policy and ref should be colocated with each other
  - rollout / inference should not be colocated with training

## Current Requirements

- Use the Harbor integration that already exists in:
  - `examples/train_integrations/harbor/`
- Keep `CodeContests` as the training dataset.
- Keep `OpenThoughts-TB-dev` as the evaluation dataset.
- Run with local Docker sandboxes instead of Daytona or Modal.
- Use an 8-GPU single node.
- Avoid `colocate_all=true`.
- Keep policy and ref colocated with each other.
- Use a GRPO-based training setup, but enable a ref model so that policy/ref colocation is actually active.

## Hardware and Resource Envelope

- GPUs: `8 x H100`
- Disk available for Docker: about `5.9T`
- CPU: `150+` cores
- CPU RAM: `500G+`

Implications for the initial plan:

- There is enough GPU to split training and rollout cleanly on the same node.
- There is enough CPU and disk to run many Harbor Docker tasks in parallel, but the first training attempt should still ramp up concurrency in stages.
- Docker image build and sandbox churn should be manageable locally.

## Integration Baseline

The plan is based on the existing Harbor integration files:

- `examples/train_integrations/harbor/run_codecontest.sh`
- `examples/train_integrations/harbor/run_harbor_gen.sh`
- `examples/train_integrations/harbor/entrypoints/main_harbor.py`
- `examples/train_integrations/harbor/entrypoints/main_harbor_generate.py`
- `examples/train_integrations/harbor/harbor_trial_config/default.yaml`

Important current behavior:

- `HarborTaskDataset` expects directories containing `instruction.md`.
- `prepare_harbor_dataset.py` already supports:
  - `CodeContests` via parquet extraction
  - `OpenThoughts-TB-dev` via direct task-directory snapshot
- `trainer.placement.colocate_policy_ref=true` only matters when a ref model is actually used.
- In current Harbor example settings, `trainer.algorithm.use_kl_loss=false`, which means no ref worker is created.
- Therefore, this plan explicitly turns `trainer.algorithm.use_kl_loss=true`.

## Planned Placement and Runtime Topology

Use one node and split GPUs into two logical pools:

- Training pool: `4 GPUs`
  - policy on `4 GPUs`
  - ref on the same `4 GPUs` via `colocate_policy_ref=true`
- Rollout pool: `4 GPUs`
  - local vLLM engines on `4 GPUs`

Planned placement settings:

- `trainer.placement.colocate_all=false`
- `trainer.placement.colocate_policy_ref=true`
- `trainer.placement.policy_num_nodes=1`
- `trainer.placement.policy_num_gpus_per_node=4`
- `trainer.placement.ref_num_nodes=1`
- `trainer.placement.ref_num_gpus_per_node=4`
- `generator.inference_engine.num_engines=4`
- `generator.inference_engine.tensor_parallel_size=1`
- `generator.inference_engine.run_engines_locally=true`

Keep critic disabled:

- `trainer.critic.model.path=null`

Reason:

- GRPO does not require a critic.
- This preserves more GPUs for policy, ref, and rollout.
- It matches the existing Harbor integration path more closely.

## Planned Harbor Configuration Changes

The default Harbor YAML is Daytona-oriented. For local Docker runs, the effective overrides should be:

- `harbor_trial_config.environment.type=docker`
- `harbor_trial_config.environment.kwargs={}`
- `harbor_trial_config.environment.override_cpus=2`
- `harbor_trial_config.environment.override_memory_mb=4096`
- `harbor_trial_config.environment.override_storage_mb=4096`
- `harbor_trial_config.agent.override_timeout_sec=900`
- `harbor_trial_config.agent.kwargs.max_turns=24`
- `harbor_trial_config.agent.kwargs.enable_summarize=false`
- `harbor_trial_config.agent.kwargs.store_all_messages=true`
- `harbor_trial_config.agent.kwargs.model_info.max_input_tokens=32768`
- `harbor_trial_config.agent.kwargs.model_info.max_output_tokens=32768`

Keep:

- `generator.apply_overlong_filtering=true`

Reason:

- Harbor trajectories are expensive because each sample is a full task trial.
- Lower `max_turns` and a bounded timeout reduce wasted rollout time during the first tuning cycle.
- Clearing Daytona-specific kwargs avoids provider confusion.

## Training Strategy

This plan uses three stages instead of jumping directly into a full run.

### Stage 1: Smoke Run

Purpose:

- Verify the full `SkyRL -> Harbor -> Docker sandbox -> verifier reward -> SkyRL` loop.
- Catch configuration, Docker image, Harbor agent, and HTTP endpoint failures early.

Dataset:

- Train subset from `CodeContests`: `32` tasks
- Eval subset from `OpenThoughts-TB-dev`: `10` tasks

Entrypoint:

- `examples.train_integrations.harbor.entrypoints.main_harbor_generate`

Recommended settings:

- `generator.n_samples_per_prompt=1`
- `generator.eval_n_samples_per_prompt=1`
- `trainer.train_batch_size=4`
- `trainer.policy_mini_batch_size=4`
- `trainer.micro_forward_batch_size_per_gpu=1`
- `trainer.micro_train_batch_size_per_gpu=1`
- `generator.rate_limit.trajectories_per_second=1`
- `generator.rate_limit.max_concurrency=8`

Success criteria:

- Harbor trials start successfully.
- Docker sandboxes build and run.
- Rewards are returned by the verifier.
- No systemic HTTP, tokenizer, or chat template failures.

### Stage 2: Pilot Run

Purpose:

- Run real RL updates with moderate cost.
- Validate memory, throughput, timeout rate, and masking behavior.

Dataset:

- Train subset from `CodeContests`: `256` tasks
- Eval subset from `OpenThoughts-TB-dev`: `20` tasks

Entrypoint:

- `examples.train_integrations.harbor.entrypoints.main_harbor`

Recommended settings:

- `trainer.epochs=1`
- `trainer.train_batch_size=8`
- `trainer.policy_mini_batch_size=8`
- `trainer.micro_forward_batch_size_per_gpu=1`
- `trainer.micro_train_batch_size_per_gpu=1`
- `trainer.algorithm.advantage_estimator=grpo`
- `trainer.algorithm.loss_reduction=seq_mean_token_sum_norm`
- `trainer.algorithm.grpo_norm_by_std=false`
- `trainer.algorithm.use_kl_loss=true`
- `trainer.algorithm.kl_loss_coef=0.001`
- `generator.n_samples_per_prompt=4`
- `generator.eval_n_samples_per_prompt=1`
- `trainer.eval_before_train=true`
- `trainer.eval_interval=8`
- `generator.rate_limit.enabled=true`
- `generator.rate_limit.trajectories_per_second=2`
- `generator.rate_limit.max_concurrency=16`

Success criteria:

- No training OOM.
- Timeout and error trajectories remain bounded.
- At least some non-zero reward signal appears consistently.
- Checkpointing and evaluation run normally.

### Stage 3: Full Run

Purpose:

- Run the first full training attempt on the full CodeContests dataset.
- Run with full monitoring enabled so memory pressure, reward quality, and convergence can be diagnosed instead of inferred.

Dataset:

- Train: full `CodeContests`
- Eval: full `OpenThoughts-TB-dev`

Recommended settings:

- `trainer.epochs=1`
- `trainer.train_batch_size=16`
- `trainer.policy_mini_batch_size=16`
- `trainer.micro_forward_batch_size_per_gpu=1`
- `trainer.micro_train_batch_size_per_gpu=1`
- `generator.n_samples_per_prompt=4`
- `generator.eval_n_samples_per_prompt=1`
- `trainer.eval_before_train=true`
- `trainer.eval_interval=50`
- `generator.rate_limit.enabled=true`
- `generator.rate_limit.trajectories_per_second=4`
- `generator.rate_limit.max_concurrency=32`
- `trainer.algorithm.max_seq_len=8192` as the first full-run baseline
- `trainer.flash_attn=false`
- `trainer.use_sample_packing=false`
- `harbor_trial_config.agent.kwargs.max_turns=12`
- `harbor_trial_config.agent.kwargs.temperature=0.3`
- `harbor_trial_config.agent.kwargs.llm_call_kwargs.extra_body.chat_template_kwargs.enable_thinking=false`
- `harbor_trial_config.agent.kwargs.llm_call_kwargs.extra_body.include_reasoning=false`
- `trainer.collect_memory_metrics=true`
- `trainer.collect_memory_metrics_interval=1`
- `trainer.logger=['tensorboard','console']` by default
  - if `WANDB_API_KEY` is set, the launch script upgrades this to `['wandb','tensorboard','console']`

Do not start with:

- `generator.n_samples_per_prompt=8`
- `trainer.train_batch_size=32`
- very high Harbor concurrency

Reason:

- In Harbor, each sample is a full multi-turn agent trial in a Docker sandbox.
- First full runs should bias toward stability, not maximum theoretical throughput.
- Stage 2 already showed that the original `32768` training-side sequence length is not a safe first full-run baseline.

### Stage 3 Monitoring Additions

Stage 3 should collect two kinds of monitoring data:

1. SkyRL internal metrics
- Source: tracker metrics from the trainer
- What this covers:
  - loss convergence
  - policy update health
  - reward quality
  - training-worker CUDA allocator state
  - training-worker model / grad / optimizer-state bytes

2. External system sampling
- Source: `examples/train_integrations/harbor/monitor_stage3_resources.sh`
- What this covers:
  - per-GPU total memory used / free
  - per-process GPU memory usage
  - rollout-engine GPU footprint vs training-worker GPU footprint
  - vLLM `GPU KV cache usage` lines from infra logs
  - all of the above mirrored into the same `TensorBoard` run directory used by SkyRL trainer metrics

Important distinction:

- `memory/*` metrics from SkyRL are the authoritative source for training-side parameter / gradient / optimizer-state bytes.
- Rollout KV cache is not exposed by SkyRL as an exact byte metric today.
- For rollout KV cache, Stage 3 should treat vLLM's `GPU KV cache usage` as the authoritative signal, and combine it with per-process GPU memory logs for context.

Required Stage 3 launch procedure:

1. Launch full run with the Stage 2-stable baseline plus memory metrics enabled:
   - `bash examples/train_integrations/harbor/run_codecontest_8xh100_docker.sh full`
   - this now auto-starts `monitor_stage3_resources.sh`
2. Keep both:
   - tracker outputs under `$HOME/tmp_logs/codecontest-8xh100-docker-full`
   - external samples under `$HOME/codecontest-8xh100-docker-full/monitoring`
3. Open a unified dashboard with:
   - `tensorboard --logdir $HOME/tmp_logs/codecontest-8xh100-docker-full/tensorboard`

Manual option:

- If the monitor needs to be run independently, use:
  - `bash examples/train_integrations/harbor/monitor_stage3_resources.sh codecontest-8xh100-docker-full 10`

Stage 3 should explicitly review all of the following after the first sustained run:

- whether `loss/avg_final_rewards` and `policy/final_loss` move in a stable direction
- whether `reward/avg_pass_at_4`, `reward/avg_raw_reward`, and `reward/mean_positive_reward` remain non-degenerate
- whether `generate/num_timeout_trajectories`, `generate/num_error_trajectories`, and `generate/num_masked_instances` remain bounded
- whether `memory/after_logprobs/*` and `memory/after_train/*` show acceptable reserved / fragmentation growth
- whether `optimizer_state_bytes_cuda_*` and `optimizer_state_bytes_cpu_*` match expected placement behavior
- whether per-process rollout-engine GPU memory stays compatible with the observed vLLM KV cache usage

## Metrics to Monitor

Track these metrics throughout pilot and full runs:

- `avg_score`
- `pass_at_n`
- `mean_positive_reward`
- `reward/avg_pass_at_4`
- `reward/avg_raw_reward`
- `reward/mean_positive_reward`
- `loss/avg_final_rewards`
- `loss/avg_raw_advantages`
- `loss/avg_raw_advantages_abs`
- `loss/avg_kl`
- `policy/final_loss`
- `policy/grad_norm`
- `policy/policy_entropy`
- `generate/num_timeout_trajectories`
- `generate/num_error_trajectories`
- `generate/num_masked_instances`
- `generate/trajectories_context_length_exceeded`
- `generate/avg_num_turns`
- `memory/after_logprobs/policy/*`
- `memory/after_train/policy/*`
- `memory/after_logprobs/ref/*`
- `memory/after_train/ref/*`

Interpretation:

- Rising `avg_score` or `pass_at_n` is the core signal that training is helping.
- `reward/*` and `loss/*` should be interpreted together; raw reward can rise while policy loss remains noisy.
- High `num_timeout_trajectories` usually indicates Harbor sandbox pressure, too many turns, or too much concurrency.
- High `num_error_trajectories` suggests task, Docker, environment, or agent failures.
- High `num_masked_instances` means many prompts are being discarded from training because one or more trajectories failed.
- High `trajectories_context_length_exceeded` suggests the context budget or turn budget is too aggressive.
- Rising `cuda_reserved_unallocated_bytes_*` with flat reward/loss often means allocator churn or unreclaimed cache is growing.
- `cuda_inactive_split_bytes_*` is the closest built-in proxy for fragmentation that is not being reused efficiently.
- `optimizer_state_bytes_cuda_*` indicates how much training GPU memory is tied up in optimizer state on the policy workers.

## Failure Handling Defaults

If the first runs are unstable, adjust in this order:

1. If training OOM happens:
   - reduce `trainer.train_batch_size`
   - keep `micro_*_batch_size_per_gpu=1`
   - do not reduce context length first

2. If Docker or Harbor becomes unstable:
   - reduce `generator.rate_limit.max_concurrency`
   - then reduce `generator.rate_limit.trajectories_per_second`

3. If trajectories timeout too often:
   - reduce Harbor concurrency
   - reduce `agent.kwargs.max_turns`
   - increase container CPU or memory if needed

4. If context overflows too often:
   - keep `generator.apply_overlong_filtering=true`
   - reduce `max_turns`
   - only reduce `max_model_len` if memory forces it

## Concrete Implementation Follow-Up

The implementation should add:

- a dedicated run script for this environment, for example:
  - `examples/train_integrations/harbor/run_codecontest_8xh100_docker.sh`
- a small dataset subset helper for smoke and pilot stages
- explicit Harbor Docker overrides in the launch command instead of relying on Daytona defaults

The implementation should not change Harbor integration core logic unless a real blocker appears during smoke testing.

## Final Defaults Chosen

- `CodeContests` is train only.
- `OpenThoughts-TB-dev` is eval only.
- `Qwen/Qwen3-8B` is the base model.
- `docker` is the Harbor environment provider.
- `colocate_all=false`
- `colocate_policy_ref=true`
- `use_kl_loss=true` so ref is actually instantiated
- `critic` remains disabled
- rollout and training are split across the 8 GPUs as `4 + 4`
