# QWEN3-32B 6-Node Rootless Fully Async 3-Step 4-Server Rerun4 Reconstructed Runbook

Date: 2026-03-16

This document reconstructs how the historical run
`codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207`
was actually launched.

It is reconstructed from the real launcher logs under:

- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207`

Important scope note:

- The exact top-level interactive shell history was not saved.
- The sequence below is the closest equivalent command chain that matches the
  observed logs and the effective runtime configuration.
- When a value is stated below, it came from the actual logs unless explicitly
  marked as an inferred shell reconstruction.

## 1. Topology

This run used:

- Slurm job: `1144`
- head: `research-dev-coder-003`
- rollout: `research-dev-coder-008`
- trainers: `research-dev-coder-012,013,014,015`
- Ray head address: `172.21.44.54:6381`
- rollout backend URLs:
  - `http://172.21.44.94:18000`
  - `http://172.21.44.94:18001`
  - `http://172.21.44.94:18002`
  - `http://172.21.44.94:18003`

Primary evidence:

- [launcher_ray.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/launcher_ray.log)
- [launcher_validation.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/launcher_validation.log)
- [launcher_rollout.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/launcher_rollout.log)

## 2. Effective Run Parameters

The train driver ended up with these effective overrides:

- `trainer.run_name=codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207`
- `trainer.resume_mode=none`
- `max_train_tasks=192`
- `max_eval_tasks=20`
- `trainer.train_batch_size=64`
- `trainer.policy_mini_batch_size=64`
- `trainer.fully_async.num_parallel_generation_workers=64`
- `trainer.fully_async.max_staleness_steps=2`
- `generator.n_samples_per_prompt=4`
- `generator.rate_limit.max_concurrency=256`
- `generator.rate_limit.trajectories_per_second=2`
- `generator.inference_engine.num_engines=4`
- `generator.inference_engine.tensor_parallel_size=2`
- `generator.inference_engine.external_server_urls=[18000,18001,18002,18003]`
- `generator.inference_engine.thunder_agent_mode=tr`
- `trainer.ckpt_interval=3`
- `trainer.hf_save_interval=-1`
- `trainer.eval_interval=50`

Primary evidence:

- [launcher_train_driver.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/launcher_train_driver.log#L626)
- [launcher_train_driver.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/launcher_train_driver.log#L641)
- [launcher_train_driver.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/launcher_train_driver.log#L644)

Why it became exactly 3 steps:

- `max_train_tasks=192`
- `train_batch_size=64`
- `192 / 64 = 3`

This is confirmed by:

- [launcher_train_driver.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/launcher_train_driver.log#L645)

## 3. Data And Artifacts

This run still used the old Harbor datasets:

- train: `/home/hkang/zthunder_agent/data/harbor/CodeContests`
- eval: `/home/hkang/zthunder_agent/data/harbor/OpenThoughts-TB-dev`

Artifacts:

- trials: `/home/hkang/zthunder_agent/harbor_runs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/trials_run`
- checkpoints: `/home/hkang/zthunder_agent/harbor_runs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/ckpts`
- exports: `/home/hkang/zthunder_agent/harbor_runs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/exports`
- tmp logs: `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207`

Primary evidence:

- [launcher_train_driver.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/launcher_train_driver.log#L642)

## 4. Observed Launch Order

The real launch order was:

1. start rootless Docker on head `003`
2. start Ray on `003 + 012-015`
3. start 4 rollout servers on `008`
4. run full validation
5. start trainer-node GPU monitors
6. run train driver
7. attempt post-run summary

This order is directly visible from:

- [launcher_rootless_docker.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/launcher_rootless_docker.log)
- [launcher_ray.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/launcher_ray.log)
- [launcher_rollout.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/launcher_rollout.log)
- [launcher_validation.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/launcher_validation.log)
- [launcher_trainer_monitors.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/launcher_trainer_monitors.log)
- [launcher_train_driver.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/launcher_train_driver.log)

## 5. Equivalent Reproduction Commands

The following command sequence matches the observed run.

### 5.1 Shared shell variables

```bash
cd /home/hkang/zthunder_agent/SkyRL

export JOB_ID=1144
export REPO=/home/hkang/zthunder_agent/SkyRL
export RUN_NAME=codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207

export HEAD_NODE=research-dev-coder-003
export ROLLOUT_NODE=research-dev-coder-008
export TRAINER_NODES_CSV=research-dev-coder-012,research-dev-coder-013,research-dev-coder-014,research-dev-coder-015

export HEAD_IP=172.21.44.54
export ROLLOUT_IP=172.21.44.94

export DOCKER_MODE=rootless
export XDG_RUNTIME_DIR=/tmp/xdg-test-hkang
export DOCKER_HOST=unix:///tmp/xdg-test-hkang/docker.sock
export HEAD_NOFILE_SOFT=131072
export RAY_PORT=6381
export RAY_ADDRESS=${HEAD_IP}:${RAY_PORT}
```

### 5.2 Start rootless Docker on the head node

Observed behavior:

- rootless socket: `/tmp/xdg-test-hkang/docker.sock`
- rootless pidfile: `/tmp/xdg-test-hkang/docker.pid`
- rootless data-root: `/tmp/hkang/docker-rootless/docker-rootless`

Equivalent command:

```bash
srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 \
  -w "$HEAD_NODE" --ntasks=1 --nodes=1 \
  bash -lc "cd '$REPO' && \
    export XDG_RUNTIME_DIR='$XDG_RUNTIME_DIR' && \
    export DOCKER_HOST='$DOCKER_HOST' && \
    export HEAD_NOFILE_SOFT='$HEAD_NOFILE_SOFT' && \
    export ROOTLESS_DOCKER_START_MODE=block && \
    bash examples/train_integrations/harbor/start_rootless_docker_for_harbor.sh" \
  2>&1 | tee /home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/launcher_rootless_docker.log
```

Primary evidence:

- [launcher_rootless_docker.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/launcher_rootless_docker.log#L36)

### 5.3 Start Ray

```bash
JOB_ID="$JOB_ID" \
HEAD_NODE="$HEAD_NODE" \
TRAINER_NODES_CSV="$TRAINER_NODES_CSV" \
RAY_PORT="$RAY_PORT" \
RAY_START_MODE=block \
bash examples/train_integrations/harbor/launch_qwen3_32b_ray_cluster.sh \
  2>&1 | tee /home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/launcher_ray.log
```

Primary evidence:

- [launcher_ray.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/launcher_ray.log#L1)

### 5.4 Start rollout servers on the rollout node

This historical run used:

- 4 servers
- ports `18000,18001,18002,18003`
- `TP_SIZE=2`
- one server per 2 GPUs

Equivalent command:

```bash
srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 \
  -w "$ROLLOUT_NODE" --ntasks=1 --nodes=1 --gres=gpu:8 \
  bash -lc "cd '$REPO' && \
    export RUN_NAME='$RUN_NAME' && \
    export RAY_HEAD_IP='$HEAD_IP' && \
    export ROLLOUT_SERVER_PORTS_CSV=18000,18001,18002,18003 && \
    export TP_SIZE=2 && \
    bash examples/train_integrations/harbor/start_qwen3_32b_external_rollout_servers.sh" \
  2>&1 | tee /home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/launcher_rollout.log
```

Primary evidence:

- [launcher_rollout.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/launcher_rollout.log#L84)

### 5.5 Run full validation

```bash
JOB_ID="$JOB_ID" \
VALIDATION_MODE=full \
HEAD_NODE="$HEAD_NODE" \
ROLLOUT_NODE="$ROLLOUT_NODE" \
TRAINER_NODES_CSV="$TRAINER_NODES_CSV" \
DOCKER_MODE=rootless \
XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" \
DOCKER_HOST="$DOCKER_HOST" \
RAY_ADDRESS="$RAY_ADDRESS" \
ROLLOUT_SERVER_URLS='["http://172.21.44.94:18000","http://172.21.44.94:18001","http://172.21.44.94:18002","http://172.21.44.94:18003"]' \
bash examples/train_integrations/harbor/run_full_run_validation_suite.sh \
  2>&1 | tee /home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/launcher_validation.log
```

Primary evidence:

- [launcher_validation.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/launcher_validation.log#L1)

### 5.6 Start trainer GPU monitors

Equivalent command:

```bash
JOB_ID="$JOB_ID" \
TRAINER_NODES_CSV="$TRAINER_NODES_CSV" \
RUN_NAME="$RUN_NAME" \
bash examples/train_integrations/harbor/start_trainer_node_monitors.sh \
  2>&1 | tee /home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/launcher_trainer_monitors.log
```

Primary evidence:

- [launcher_trainer_monitors.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/launcher_trainer_monitors.log)

### 5.7 Launch the 3-step train driver

This is the critical reconstructed command.

```bash
cd /home/hkang/zthunder_agent/SkyRL

export RUN_NAME_OVERRIDE="$RUN_NAME"
export DOCKER_MODE=rootless
export XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR"
export DOCKER_HOST="$DOCKER_HOST"
export RAY_HEAD_IP="$HEAD_IP"
export ROLLOUT_HOST_IP="$ROLLOUT_IP"
export RAY_ADDRESS="$RAY_ADDRESS"
export ROLLOUT_SERVER_PORTS_CSV=18000,18001,18002,18003
export ROLLOUT_ENGINES=4
export ROLLOUT_TP_SIZE=2
export HEAD_NOFILE_SOFT=131072

stdbuf -oL -eL \
  bash examples/train_integrations/harbor/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh full \
    max_train_tasks=192 \
    trainer.resume_mode=none \
    trainer.ckpt_interval=3 \
    trainer.hf_save_interval=-1 \
  2>&1 | tee /home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/launcher_train_driver.log
```

Why these overrides are required:

- without `max_train_tasks=192`, this would not stop at 3 train steps
- without `trainer.resume_mode=none`, current script default is `latest`
- without `ROLLOUT_SERVER_PORTS_CSV=18000,18001,18002,18003`, current script default is only 2 rollout ports
- without `ROLLOUT_ENGINES=4` and `ROLLOUT_TP_SIZE=2`, current script defaults do not match the historical run

Primary evidence:

- [launcher_train_driver.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/launcher_train_driver.log#L626)
- [launcher_train_driver.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/launcher_train_driver.log#L641)
- [launcher_train_driver.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/launcher_train_driver.log#L645)

## 6. What Happened During The Run

Training:

- bring-up and preflight: `202.68s`
- step 1 wait/train/finalize: `453.28s / 63.58s / 28.47s`
- step 2 wait/train/finalize: `325.74s / 64.95s / 22.18s`
- step 3 wait/train/finalize: `566.11s / 52.83s / 3.41s`
- then eval ran for `1446.00s`
- final checkpoint saved at `global_step_3`
- trainer logged `Training done!`

References:

- [analysis/timeline_report.md](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/analysis/timeline_report.md)
- [launcher_train_driver.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/launcher_train_driver.log#L4036)

ThunderAgent:

- started on head `003` at `http://172.21.44.54:8080`
- configured with 4 backends
- mode `tr`
- backend KV capacity recorded as `254032` tokens each

References:

- [thunderagent.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/thunderagent.log#L1)

## 7. Known Post-Run Detail

The historical run itself completed, but the post-run summary helper failed.

The failure was in:

- `examples/train_integrations/harbor/summarize_qwen3_32b_full_run.py`

The reason was a schema mismatch while writing rollout KV timeline TSV rows:

- extra fields such as `event_timestamp`, `metrics_url`, `prompt_tokens_total`,
  `generation_tokens_total`, `prefix_cache_hits_total`, and others were present
  in a row but not in the TSV field list.

Primary evidence:

- [launcher_summary.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/launcher_summary.log)

## 8. Practical Reuse Notes

If you want to reproduce this exact pattern again, do not forget these four
differences from the default script path:

1. 4 rollout servers, not 2
2. rollout TP size `2`, not `4`
3. `max_train_tasks=192` to force 3 train steps
4. `trainer.resume_mode=none`

Those four are the main reasons the historical run looked different from the
default current `full` path.
