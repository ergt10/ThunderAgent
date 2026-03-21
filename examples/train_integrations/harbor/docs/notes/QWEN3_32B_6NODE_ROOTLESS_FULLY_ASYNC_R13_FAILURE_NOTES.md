# Qwen3-32B 6-Node Rootless Harbor Full-Async `r13` Failure Notes

Date: 2026-03-12

## Scope

This note records the latest planned aggressive 32B Harbor full-async run that failed before step `0`.

Run name:

- `codecontest-qwen3-32b-6node-rootless-full-r13-64x4x64-stale2-conc256`

Goal:

- keep the same `4 trainer + 1 rollout + 1 Harbor head` topology
- raise training utilization to `64/64/4/4`
- raise rollout to `n_samples=4`, `num_parallel_generation_workers=64`, `max_concurrency=256`

Final result:

- rollout infrastructure started correctly
- ThunderAgent started correctly
- Harbor datasets loaded correctly
- trainer failed before creating the first placement group
- no trials, no checkpoints, no exports

## Direct Evidence

Primary log:

- `/home/hkang/zthunder_yagent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-r13-64x4x64-stale2-conc256/launcher-interactive.log`

Other relevant logs:

- `/home/hkang/zthunder_yagent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-r13-64x4x64-stale2-conc256/thunderagent.log`
- `/home/hkang/zthunder_yagent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-r13-64x4x64-stale2-conc256/rollout/rollout_a.log`
- `/home/hkang/zthunder_yagent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-r13-64x4x64-stale2-conc256/rollout/rollout_b.log`

Artifact root:

- `/data/zy/models/hkang/harbor_runs/codecontest-qwen3-32b-6node-rootless-full-r13-64x4x64-stale2-conc256`

## Planned Topology

Operator-selected nodes for this failed run:

- Harbor head / rootless Docker / Ray head: `research-secure-14`
- Trainer nodes:
  - `research-secure-23`
  - `research-secure-06`
  - `research-secure-11`
  - `research-secure-27`
- External rollout node: `research-secure-32`

Directly visible from logs:

- head IP: `172.27.31.10`
- rollout backend URLs:
  - `http://172.27.18.89:18000`
  - `http://172.27.18.89:18001`

## Effective `r13` Config

These values are visible in `launcher-interactive.log`.

### Training

- `trainer.strategy=fsdp2`
- `trainer.flash_attn=true`
- `trainer.train_batch_size=64`
- `trainer.policy_mini_batch_size=64`
- `trainer.micro_forward_batch_size_per_gpu=4`
- `trainer.micro_train_batch_size_per_gpu=4`
- `trainer.algorithm.max_seq_len=6144`
- `trainer.fully_async.max_staleness_steps=2`
- `trainer.fully_async.num_parallel_generation_workers=64`
- `trainer.placement.colocate_all=false`
- `trainer.placement.colocate_policy_ref=true`
- `trainer.placement.policy_num_nodes=4`
- `trainer.placement.policy_num_gpus_per_node=8`
- `trainer.placement.ref_num_nodes=4`
- `trainer.placement.ref_num_gpus_per_node=8`

### Rollout

- `generator.n_samples_per_prompt=4`
- `generator.eval_n_samples_per_prompt=1`
- `generator.rate_limit.trajectories_per_second=2`
- `generator.rate_limit.max_concurrency=256`
- `generator.inference_engine.run_engines_locally=false`
- `generator.inference_engine.external_server_urls=['http://172.27.18.89:18000','http://172.27.18.89:18001']`
- `generator.inference_engine.tensor_parallel_size=4`
- `generator.inference_engine.enforce_eager=false`
- `generator.inference_engine.gpu_memory_utilization=0.8`
- `generator.inference_engine.engine_init_kwargs.max_model_len=32768`
- `generator.inference_engine.thunder_agent_mode=tr`

### Other Important Values

- model path: `/data/zy/models/hkang/models/Qwen3-32B`
- `generator.sampling_params.logprobs=1`
- `trainer.algorithm.off_policy_correction.tis_ratio_type=token`
- `max_train_tasks=null`
- `max_eval_tasks=20`
- `trainer.eval_before_train=false`
- `trainer.eval_interval=50`
- `trainer.ckpt_interval=5`
- `trainer.hf_save_interval=5`

## What Worked

The failure was not in the Harbor verifier and not in external rollout bring-up.

What is directly confirmed to have worked:

1. ThunderAgent started.
2. ThunderAgent discovered both rollout backends.
3. ThunderAgent started metrics monitoring for both rollout backends.
4. Harbor train and eval datasets loaded.
5. HarborGenerator initialized with the ThunderAgent proxy URL.
6. Rate limiter initialized with the intended aggressive settings.

Direct evidence from logs:

- ThunderAgent started over:
  - `['http://172.27.18.89:18000', 'http://172.27.18.89:18001']`
- ThunderAgent started at:
  - `http://172.27.31.10:8081`
- rollout logs show both TP=4 vLLM servers loaded Qwen3-32B successfully
- rollout logs show:
  - `Using Flash Attention backend on V1 engine`
  - model load took about `15.392 GiB` per TP worker
- HarborGenerator initialized with:
  - `api_base: http://172.27.31.10:8081/v1`
- rate limiter initialized with:
  - `2 trajectories/second, max 256 concurrent`

## Actual Failure

The run died in trainer model build before step `0`.

Direct error:

```text
RuntimeError: Failed to create placement group with 4 bundles (requiring 32.0 GPUs, 32.0 CPUs total) in 180 seconds. This might indicate insufficient GPU resources.
```

The same log also contains repeated autoscaler messages:

```text
Error: No available node types can fulfill resource request {'CPU': 8.0, 'GPU': 8.0}.
```

This happened after:

- Ray connection succeeded
- dataset loading succeeded
- ThunderAgent router creation succeeded
- HarborGenerator creation succeeded
- dataloader build succeeded

The trainer never reached:

- `Started: 'step'`
- `wait_for_generation_buffer`
- any rollout consumption
- any optimizer step

## Observable Outcome

Run outputs were empty:

- `trials_run`: `0` entries
- `ckpts`: `0` entries
- `exports`: `0` entries

So this was a pure cluster scheduling failure, not a later training failure.

## Most Likely Cause

This part is diagnosis, not direct proof from the launcher log.

The most likely cause is Ray resource registration mismatch on the trainer pool:

- the run requested `4` bundles of `8 GPU + 8 CPU`
- Ray did not believe the cluster had schedulable nodes matching `{'CPU': 8.0, 'GPU': 8.0}`
- in the bring-up sequence around this run, trainer workers had to be restarted with explicit:
  - `--num-cpus=176`
  - `--num-gpus=8`

So the likely issue was not real hardware shortage, but incorrect or incomplete Ray worker resource registration.

## Repro Steps For This Failure

These steps reproduce the `r13` failure mode, assuming the same broken Ray resource state.

### 1. Head node

- start rootless Docker on the Harbor head
- start Ray head on `172.27.31.10:6381`
- ensure ThunderAgent head-pinned entrypoint is used

### 2. Rollout node

- start `examples/train_integrations/harbor/start_qwen3_32b_external_rollout_servers.sh`
- use:
  - `TP_SIZE=4`
  - ports `18000` and `18001`
  - `MAX_MODEL_LEN=32768`

### 3. Training launch

Launch:

```bash
export RUN_NAME_OVERRIDE=codecontest-qwen3-32b-6node-rootless-full-r13-64x4x64-stale2-conc256
export LOG_ROOT=/home/hkang/zthunder_yagent/tmp_logs
export RAY_ADDRESS=172.27.31.10:6381
export ROLLOUT_SERVER_URLS="['http://172.27.18.89:18000','http://172.27.18.89:18001']"
export FULL_TRAIN_BATCH_SIZE=64
export FULL_POLICY_MINI_BATCH_SIZE=64
export FULL_MICRO_FORWARD_BATCH_SIZE_PER_GPU=4
export FULL_MICRO_TRAIN_BATCH_SIZE_PER_GPU=4
export FULL_N_SAMPLES=4
export FULL_NUM_PARALLEL_GENERATION_WORKERS=64
export FULL_MAX_CONCURRENCY=256
export FULL_TRAJ_PER_SEC=2
export FULL_MAX_STALENESS_STEPS=2
export FLASH_ATTN=true
export TRAIN_MAX_SEQ_LEN=6144

bash examples/train_integrations/harbor/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh full
```

Under the bad Ray worker resource state, this reaches the same placement-group timeout.

## What Needs To Be Fixed Before Retrying

Before retrying this exact `64/64/4/4 + 4/64/256` plan:

1. verify Ray cluster resources from the head node
2. make sure the head exposes `{"harbor_head": 1}`
3. restart each trainer worker with explicit `--num-cpus` and `--num-gpus`
4. confirm `ray.cluster_resources()` really shows:
   - `GPU: 32`
   - enough CPU for 4 trainer bundles

Until that is fixed, this run shape fails before any meaningful Harbor or training traffic happens.
