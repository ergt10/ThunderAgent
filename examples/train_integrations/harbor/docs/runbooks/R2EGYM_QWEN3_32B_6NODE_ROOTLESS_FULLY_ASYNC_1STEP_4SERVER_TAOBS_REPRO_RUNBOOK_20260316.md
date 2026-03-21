# R2EGYM QWEN3-32B 6-Node Rootless Fully Async 1-Step 4-Server TA-Observability Repro Runbook

Date: 2026-03-16

This document reconstructs the exact workflow used for the successful 1-step observability run:

- `r2egym-qwen3-32b-6node-rootless-full-1step-4srv-taobs-20260316_061103`

Primary logs:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-taobs-20260316_061103`

Primary artifacts:

- `/data/zy/models/hkang/harbor_runs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-taobs-20260316_061103`

This is the runbook to use if you want the same:

1. `r2egym` 4-split training input
2. 4 rollout servers with `TP=2`
3. only `1` training step
4. ThunderAgent backend/program observability TSVs
5. SkyRL completed-trial curve inputs
6. no shortcut wrapper
7. no separate standalone validation launcher

## 1. Parent Runbook And Exact Differences

This run is derived from:

- `/home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/R2EGYM_QWEN3_32B_6NODE_ROOTLESS_FULLY_ASYNC_1EPOCH_4SERVER_REPRO_RUNBOOK_20260316.md`

Compared with that parent runbook, this 1-step observability run changed only these things:

1. `RUN_NAME` changed to `r2egym-qwen3-32b-6node-rootless-full-1step-4srv-taobs-20260316_061103`
2. training was truncated to `1` step by setting `max_train_tasks=64`
3. a standalone `launcher_validation.log` step was not run
4. the train driver relied on its own inline preflight:
   - `head-nofile`
   - `harbor-rootless`
   - rollout `/health`
   - Ray placement-group probe
5. ThunderAgent observability was captured through `/router_state`
6. post-run analysis included:
   - ThunderAgent token/program curves
   - completed trial curve
   - fine-grained Harbor step timing plots

Important boundary:

- the successful 1-step run reused an already-running rootless Docker daemon on head `003`
- therefore this run directory does not contain a fresh `launcher_rootless_docker.log`
- if you are reproducing from a fresh allocation, start rootless Docker first exactly as described in the parent runbook section `10`

## 2. Historical Topology

The successful 1-step run used:

- Slurm allocation: `1144`
- head: `research-dev-coder-003`
- rollout: `research-dev-coder-008`
- trainers:
  - `research-dev-coder-012`
  - `research-dev-coder-013`
  - `research-dev-coder-014`
  - `research-dev-coder-015`
- head IP: `172.21.44.54`
- rollout IP: `172.21.44.94`
- Ray address: `172.21.44.54:6381`
- rollout backend URLs:
  - `http://172.21.44.94:18000`
  - `http://172.21.44.94:18001`
  - `http://172.21.44.94:18002`
  - `http://172.21.44.94:18003`

Evidence:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-taobs-20260316_061103/launcher_ray.log`
- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-taobs-20260316_061103/launcher_rollout.log`
- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-taobs-20260316_061103/thunderagent.log`

## 3. Effective Runtime Configuration

The successful run actually used:

- `trainer.run_name=r2egym-qwen3-32b-6node-rootless-full-1step-4srv-taobs-20260316_061103`
- `trainer.resume_mode=none`
- `max_train_tasks=64`
- `max_eval_tasks=20`
- `trainer.epochs=1`
- `trainer.train_batch_size=64`
- `trainer.policy_mini_batch_size=64`
- `trainer.fully_async.num_parallel_generation_workers=64`
- `trainer.fully_async.max_staleness_steps=2`
- `generator.n_samples_per_prompt=4`
- `generator.rate_limit.max_concurrency=256`
- `generator.rate_limit.trajectories_per_second=2`
- `generator.inference_engine.num_engines=4`
- `generator.inference_engine.tensor_parallel_size=2`
- `generator.inference_engine.thunder_agent_mode=tr`
- `generator.inference_engine.thunder_agent_metrics_enabled=true`
- `generator.inference_engine.thunder_agent_metrics_interval=5.0`
- `generator.inference_engine.thunder_agent_scheduler_interval=5.0`

The train driver log proves the key deltas:

- `max_train_tasks: 64`
- `resume_mode: none`
- `run_name: r2egym-qwen3-32b-6node-rootless-full-1step-4srv-taobs-20260316_061103`
- `tensor_parallel_size: 2`
- `thunder_agent_mode: tr`
- `num_parallel_generation_workers: 64`
- `max_concurrency: 256`

Evidence:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-taobs-20260316_061103/launcher_train_driver.log`

## 4. Datasets

Train and eval both used the same four Harbor-format directories:

- `/home/hkang/zthunder_agent/data/harbor/r2egym-trivial`
- `/home/hkang/zthunder_agent/data/harbor/r2egym-easy`
- `/home/hkang/zthunder_agent/data/harbor/r2egym-medium`
- `/home/hkang/zthunder_agent/data/harbor/r2egym-hard`

Use the exact Python list literal:

```bash
export TRAIN_DATA="['/home/hkang/zthunder_agent/data/harbor/r2egym-trivial','/home/hkang/zthunder_agent/data/harbor/r2egym-easy','/home/hkang/zthunder_agent/data/harbor/r2egym-medium','/home/hkang/zthunder_agent/data/harbor/r2egym-hard']"
export EVAL_DATA="['/home/hkang/zthunder_agent/data/harbor/r2egym-trivial','/home/hkang/zthunder_agent/data/harbor/r2egym-easy','/home/hkang/zthunder_agent/data/harbor/r2egym-medium','/home/hkang/zthunder_agent/data/harbor/r2egym-hard']"
```

## 5. Shared Variables

```bash
cd /home/hkang/zthunder_agent/SkyRL

export JOB_ID=1144
export REPO=/home/hkang/zthunder_agent/SkyRL
export RUN_NAME=r2egym-qwen3-32b-6node-rootless-full-1step-4srv-taobs-20260316_061103

export HEAD_NODE=research-dev-coder-003
export ROLLOUT_NODE=research-dev-coder-008
export TRAINER_NODES_CSV=research-dev-coder-012,research-dev-coder-013,research-dev-coder-014,research-dev-coder-015

export HEAD_IP=172.21.44.54
export ROLLOUT_IP=172.21.44.94
export RAY_PORT=6381
export RAY_ADDRESS=${HEAD_IP}:${RAY_PORT}

export DOCKER_MODE=rootless
export XDG_RUNTIME_DIR=/tmp/xdg-test-hkang
export DOCKER_HOST=unix:///tmp/xdg-test-hkang/docker.sock
export HEAD_NOFILE_SOFT=131072

export TRAIN_DATA="['/home/hkang/zthunder_agent/data/harbor/r2egym-trivial','/home/hkang/zthunder_agent/data/harbor/r2egym-easy','/home/hkang/zthunder_agent/data/harbor/r2egym-medium','/home/hkang/zthunder_agent/data/harbor/r2egym-hard']"
export EVAL_DATA="['/home/hkang/zthunder_agent/data/harbor/r2egym-trivial','/home/hkang/zthunder_agent/data/harbor/r2egym-easy','/home/hkang/zthunder_agent/data/harbor/r2egym-medium','/home/hkang/zthunder_agent/data/harbor/r2egym-hard']"

export LOG_DIR=/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME
export RUN_ARTIFACT_DIR=/data/zy/models/hkang/harbor_runs/$RUN_NAME
```

## 6. Fresh Rootless Docker Prerequisite

If rootless Docker is not already alive on head `003`, run the parent runbook rootless-Docker step first.

This 1-step run reused an already-running dockerd. The successful train log later confirmed:

- current process nofile: `131072`
- rootless dockerd nofile: `131072`
- dockerd pid source: `XDG_RUNTIME_DIR/docker.pid`

Evidence:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-taobs-20260316_061103/launcher_train_driver.log`

## 7. Start Ray

This run did launch a fresh Ray cluster for the 1-step experiment:

```bash
JOB_ID="$JOB_ID" \
HEAD_NODE="$HEAD_NODE" \
TRAINER_NODES_CSV="$TRAINER_NODES_CSV" \
RAY_PORT="$RAY_PORT" \
RAY_START_MODE=block \
bash /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/launch_qwen3_32b_ray_cluster.sh \
  2>&1 | tee "$LOG_DIR/launcher_ray.log"
```

What to wait for:

- `Ray cluster ready at 172.21.44.54:6381`
- `Blocking mode is active`

Evidence:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-taobs-20260316_061103/launcher_ray.log`

## 8. Start The Four Rollout Servers

This step matched the 4-server `TP=2` path:

```bash
srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 \
  -w "$ROLLOUT_NODE" --ntasks=1 --nodes=1 --gres=gpu:8 \
  bash -lc "cd '$REPO' && \
    export RUN_NAME='$RUN_NAME' && \
    export RAY_HEAD_IP='$HEAD_IP' && \
    export ROLLOUT_SERVER_PORTS_CSV=18000,18001,18002,18003 && \
    export TP_SIZE=2 && \
    bash examples/train_integrations/harbor/start_qwen3_32b_external_rollout_servers.sh" \
  2>&1 | tee "$LOG_DIR/launcher_rollout.log"
```

Expected healthy sequence:

- `rollout_a healthy at http://127.0.0.1:18000`
- `rollout_b healthy at http://127.0.0.1:18001`
- `rollout_c healthy at http://127.0.0.1:18002`
- `rollout_d healthy at http://127.0.0.1:18003`
- `External rollout servers ready`

Early `ConnectionRefused` against `127.0.0.1:1800x/metrics` while the servers boot is expected.

Evidence:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-taobs-20260316_061103/launcher_rollout.log`

## 9. Do Not Run The Old Standalone Full Validation Step

This 1-step observability run did **not** use a separate:

- `launcher_validation.log`
- `run_full_run_validation_suite.sh`

Instead, the train driver itself performed only the minimal inline checks it needed:

- `harbor-rootless`
- `head-nofile`
- rollout `/health`
- Ray placement-group probe

This is one of the defining differences from the parent `1-epoch` runbook.

## 10. Start Trainer Monitors

```bash
JOB_ID="$JOB_ID" \
TRAINER_NODES_CSV="$TRAINER_NODES_CSV" \
RUN_NAME="$RUN_NAME" \
bash /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/start_trainer_node_monitors.sh \
  2>&1 | tee "$LOG_DIR/launcher_trainer_monitors.log"
```

This writes per-node monitor outputs under:

- `/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/trainer_monitors/research-dev-coder-012`
- `/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/trainer_monitors/research-dev-coder-013`
- `/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/trainer_monitors/research-dev-coder-014`
- `/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/trainer_monitors/research-dev-coder-015`

Evidence:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-taobs-20260316_061103/launcher_trainer_monitors.log`

## 11. Launch The 1-Step Train Driver

This is the actual launch form for the successful 1-step observability run:

```bash
cd /home/hkang/zthunder_agent/SkyRL

export RUN_NAME_OVERRIDE="$RUN_NAME"
export DOCKER_MODE=rootless
export XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR"
export DOCKER_HOST="$DOCKER_HOST"
export HEAD_NOFILE_SOFT=131072
export RAY_HEAD_IP="$HEAD_IP"
export ROLLOUT_HOST_IP="$ROLLOUT_IP"
export RAY_ADDRESS="$RAY_ADDRESS"
export TRAIN_DATA="$TRAIN_DATA"
export EVAL_DATA="$EVAL_DATA"
export ROLLOUT_SERVER_PORTS_CSV=18000,18001,18002,18003
export ROLLOUT_ENGINES=4
export ROLLOUT_TP_SIZE=2

stdbuf -oL -eL \
  bash /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh full \
    max_train_tasks=64 \
    trainer.resume_mode=none \
  2>&1 | tee "$LOG_DIR/launcher_train_driver.log"
```

Why these overrides matter:

1. `max_train_tasks=64` is what collapsed the run to `1` training step
2. `trainer.resume_mode=none` avoids picking up an old checkpoint
3. `TRAIN_DATA` and `EVAL_DATA` must be the 4-split `r2egym` list literal
4. `ROLLOUT_ENGINES=4` and `ROLLOUT_TP_SIZE=2` must match the actual rollout topology
5. `RUN_NAME_OVERRIDE` must be set so all logs and artifacts align

## 12. What The Train Driver Does Inline

This run did not depend on a separate validation launcher. The train driver itself:

1. raised and checked head `nofile`
2. started the head-side fully-async monitor
3. connected to the existing Ray cluster
4. initially attempted to scrape `http://172.21.44.54:8080/router_state`
5. tolerated early `ConnectionRefused` before ThunderAgent was up
6. checked:
   - rootless Harbor compatibility
   - head/rootless nofile
   - rollout `/health`
   - Ray placement-group viability
7. launched ThunderAgent on `http://172.21.44.54:8080`
8. then started the actual training step

Evidence:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-taobs-20260316_061103/launcher_train_driver.log`
- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-taobs-20260316_061103/thunderagent.log`

## 13. ThunderAgent Observability Outputs

This is the main reason this run exists. The head monitor writes:

- `/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/monitoring/thunderagent_backend_state.tsv`
- `/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/monitoring/thunderagent_program_state.tsv`
- `/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/monitoring/thunderagent_events.tsv`
- `/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/monitoring/trial_progress.tsv`

These feed:

1. backend `reasoning token / acting token` curves
2. pause/resume/release event markers
3. program count curves
4. completed trial curve

The rollout monitor still writes:

- `/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/rollout/monitoring/vllm_metrics.tsv`

## 14. Historical Step Boundary For This Run

The successful 1-step run hit:

- `Started: 'step'` at `2026-03-16 06:35:05 PDT`
- `Finished: 'wait_for_generation_buffer'` at `2026-03-16 06:54:16 PDT`
- `Finished: 'run_training'` at `2026-03-16 06:55:15 PDT`
- `Finished: 'step'` at `2026-03-16 06:55:18 PDT`

Evidence:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-taobs-20260316_061103/launcher_train_driver.log`

Important note:

- after that single training step, the driver continued into its normal eval phase
- the runbook target for this experiment was the completed training step, not “stop before eval”

## 15. Historical Step-End State

At step end:

- `256` training trials had finished
- `result_json=256`
- `exception_txt=4`
- ThunderAgent had released all training programs
- `released_total=256`
- `total_programs=0`

Evidence:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-taobs-20260316_061103/monitoring/trial_progress.tsv`
- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-taobs-20260316_061103/monitoring/thunderagent_program_state.tsv`

## 16. Ignore The Broken Mislaunch Artifact

There is also a file:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-taobs-20260316_061103/launcher_train_driver.log.attempt_hn1_mislaunch`

This belongs to the earlier broken attempt before `/router_state` was attached to the correct outer ThunderAgent entrypoint.

Do not use that attempt as the reproduction reference.

The successful reference is the main:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-taobs-20260316_061103/launcher_train_driver.log`

## 17. Post-Run Analysis Commands

### 17.1 Timeline And Rollout Curves

```bash
/home/hkang/zthunder_agent/SkyRL/.venv/bin/python \
  /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/analyze_qwen3_full_run_timeline.py \
  --run-name "$RUN_NAME" \
  --log-root /home/hkang/zthunder_agent/tmp_logs \
  --output-dir "/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/analysis"
```

### 17.2 ThunderAgent Backend/Program/Completed-Trial Curves

```bash
/home/hkang/zthunder_agent/SkyRL/.venv/bin/python \
  /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/analyze_thunderagent_monitoring.py \
  --log-dir "/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME" \
  --output-dir "/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/analysis"
```

Expected outputs:

- `thunderagent_backend_tokens.png`
- `thunderagent_program_counts.png`
- `skyrl_completed_trials.png`
- `thunderagent_monitoring_report.md`

### 17.3 Fine-Grained Harbor Step Timing

```bash
/home/hkang/zthunder_agent/SkyRL/.venv/bin/python \
  /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/analyze_harbor_step_timing.py \
  --run-dir "/data/zy/models/hkang/harbor_runs/$RUN_NAME" \
  --output-dir "/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/analysis"
```

Expected outputs:

- `trial_step_timing_breakdown.png`
- `trial_step_timing_heatmap.png`
- `trial_step_timing_report.md`

## 18. Clean Reuse Policy

If you want to reuse the same `RUN_NAME`, clear only this run’s state:

```bash
rm -rf "/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME"
rm -rf "/data/zy/models/hkang/harbor_runs/$RUN_NAME"
mkdir -p "/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME"
```

If you want to keep the Slurm allocation alive, do not `scancel "$JOB_ID"`.

## 19. Minimal Reproduction Contract

To reproduce this specific 1-step observability run, do not change these:

1. keep the same `r2egym` 4-split dataset lists
2. keep `4` rollout servers
3. keep `TP=2`
4. keep `trainer.resume_mode=none`
5. keep `max_train_tasks=64`
6. do not use a shortcut wrapper
7. do not insert the old standalone full validation step
8. use the current codebase state where `/router_state` is exposed from:
   - `/home/hkang/zthunder_agent/SkyRL/examples/train/thunder_agent/thunder_agent_router.py`
