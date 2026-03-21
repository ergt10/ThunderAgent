# CodeContests Qwen3-32B 6-Node Rootless Fully Async 5-Step 4-Server mini-swe-agent-50 retry4 `rerunfixfd` Actual Command Runbook

Date: 2026-03-17

This document records the exact correct launch, monitoring, and stop commands used for the last CodeContests experiment:

- `codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633`

This document intentionally excludes the wrong attempts from earlier in the night.

## 1. Run Identity

- Slurm allocation: `1144`
- Head node: `research-dev-coder-003`
- Rollout node: `research-dev-coder-008`
- Trainer nodes:
  - `research-dev-coder-012`
  - `research-dev-coder-013`
  - `research-dev-coder-014`
  - `research-dev-coder-015`
- Head IP: `172.21.44.54`
- Rollout IP: `172.21.44.94`
- Ray port: `6381`
- Ray address: `172.21.44.54:6381`
- Rollout ports:
  - `18000`
  - `18001`
  - `18002`
  - `18003`

Primary log directory:

- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633`

Primary artifact directory:

- `/home/hkang/zthunder_agent/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633`

## 2. Runtime Assumption Used By The Run

Only the repo-local venv was used:

- `/home/hkang/zthunder_agent/SkyRL/.venv`

No runtime patch command was executed during this launch. The launch relied on the repo state already on disk at launch time, including the patched copies of:

- `/home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/launch_qwen3_32b_ray_cluster.sh`
- `/home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/start_qwen3_32b_external_rollout_servers.sh`

## 3. Shared Variables

These were the shared environment values used for the run.

```bash
cd /home/hkang/zthunder_agent/SkyRL

export JOB_ID=1144
export REPO=/home/hkang/zthunder_agent/SkyRL
export RUN_NAME=codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633

export HEAD_NODE=research-dev-coder-003
export ROLLOUT_NODE=research-dev-coder-008
export TRAINER_NODES_CSV=research-dev-coder-012,research-dev-coder-013,research-dev-coder-014,research-dev-coder-015

export HEAD_IP=172.21.44.54
export ROLLOUT_IP=172.21.44.94
export RAY_PORT=6381
export RAY_ADDRESS=${HEAD_IP}:${RAY_PORT}

export DOCKER_MODE=rootless
export XDG_RUNTIME_DIR=/tmp/xdg-mswe50k-hkang
export DOCKER_HOST=unix:///tmp/xdg-mswe50k-hkang/docker.sock
export DOCKER_PIDFILE=/tmp/xdg-mswe50k-hkang/docker.pid
export DOCKER_EXEC_ROOT=/tmp/xdg-mswe50k-hkang/docker-exec
export SCRATCH_ROOT=/tmp/hkang/docker-rootless-mswe50k
export DOCKER_DATA_ROOT=/tmp/hkang/docker-rootless-mswe50k/docker-rootless
export ROOTLESS_DOCKER_START_MODE=background

export HEAD_NOFILE_SOFT=131072
export TRAINER_NOFILE_SOFT=131072
export ROLLOUT_NOFILE_SOFT=131072

export TRAIN_DATA="['/home/hkang/zthunder_agent/data/harbor/CodeContests']"
export EVAL_DATA="$TRAIN_DATA"

export LOG_DIR=/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME
export RUN_ARTIFACT_ROOT=/home/hkang/zthunder_agent
export RUN_ARTIFACT_DIR=$RUN_ARTIFACT_ROOT/$RUN_NAME

export ROLLOUT_SERVER_PORTS_CSV=18000,18001,18002,18003
export ROLLOUT_ENGINES=4
export ROLLOUT_TP_SIZE=2
```

## 4. Head Shell On `003`

These commands were run in the persistent shell on `research-dev-coder-003`.

```bash
cd /home/hkang/zthunder_agent/SkyRL
export PATH=/home/hkang/zthunder_agent/SkyRL/.venv/bin:$HOME/.local/bin:$PATH

export JOB_ID=1144
export REPO=/home/hkang/zthunder_agent/SkyRL
export RUN_NAME=codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633

export HEAD_NODE=research-dev-coder-003
export ROLLOUT_NODE=research-dev-coder-008
export TRAINER_NODES_CSV=research-dev-coder-012,research-dev-coder-013,research-dev-coder-014,research-dev-coder-015

export HEAD_IP=172.21.44.54
export ROLLOUT_IP=172.21.44.94
export RAY_PORT=6381
export RAY_ADDRESS=${HEAD_IP}:${RAY_PORT}

export DOCKER_MODE=rootless
export XDG_RUNTIME_DIR=/tmp/xdg-mswe50k-hkang
export DOCKER_HOST=unix:///tmp/xdg-mswe50k-hkang/docker.sock
export DOCKER_PIDFILE=/tmp/xdg-mswe50k-hkang/docker.pid
export DOCKER_EXEC_ROOT=/tmp/xdg-mswe50k-hkang/docker-exec
export SCRATCH_ROOT=/tmp/hkang/docker-rootless-mswe50k
export DOCKER_DATA_ROOT=/tmp/hkang/docker-rootless-mswe50k/docker-rootless
export ROOTLESS_DOCKER_START_MODE=background

export HEAD_NOFILE_SOFT=131072
export TRAINER_NOFILE_SOFT=131072
export ROLLOUT_NOFILE_SOFT=131072

export TRAIN_DATA="['/home/hkang/zthunder_agent/data/harbor/CodeContests']"
export EVAL_DATA="$TRAIN_DATA"

export LOG_DIR=/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME
export RUN_ARTIFACT_ROOT=/home/hkang/zthunder_agent
export RUN_ARTIFACT_DIR=$RUN_ARTIFACT_ROOT/$RUN_NAME

export ROLLOUT_SERVER_PORTS_CSV=18000,18001,18002,18003
export ROLLOUT_ENGINES=4
export ROLLOUT_TP_SIZE=2

mkdir -p "$LOG_DIR"
```

### 4.1 Start Rootless Docker

```bash
bash /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/start_rootless_docker_for_harbor.sh \
  2>&1 | tee "$LOG_DIR/launcher_rootless_docker.log"
```

### 4.2 Start Ray Head And Trainer Workers

```bash
JOB_ID="$JOB_ID" \
HEAD_NODE="$HEAD_NODE" \
TRAINER_NODES_CSV="$TRAINER_NODES_CSV" \
RAY_PORT="$RAY_PORT" \
RAY_START_MODE=block \
HEAD_NOFILE_SOFT="$HEAD_NOFILE_SOFT" \
TRAINER_NOFILE_SOFT="$TRAINER_NOFILE_SOFT" \
bash /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/launch_qwen3_32b_ray_cluster.sh
```

This command stayed attached to the persistent shell on `003` because `RAY_START_MODE=block` was used.

### 4.3 Launch The Train Driver

This command was run later in the same persistent shell on `003`, after rollout and trainer monitors were already up.

```bash
export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
export RAY_HEAD_IP="$HEAD_IP"
export ROLLOUT_HOST_IP="$ROLLOUT_IP"
export TRAIN_DATA="$TRAIN_DATA"
export EVAL_DATA="$EVAL_DATA"
export ROLLOUT_SERVER_PORTS_CSV=18000,18001,18002,18003
export ROLLOUT_ENGINES=4
export ROLLOUT_TP_SIZE=2
export DOCKER_MODE=rootless
export RUN_NAME_OVERRIDE="$RUN_NAME"

stdbuf -oL -eL \
  bash /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh full \
    max_train_tasks=64 \
    trainer.resume_mode=none \
    harbor_trial_config.agent.name=mini-swe-agent \
    harbor_trial_config.agent.kwargs.max_turns=50 \
  2>&1 | tee "$LOG_DIR/launcher_train_driver.log"
```

## 5. Rollout Shell On `008`

These commands were run in the persistent shell on `research-dev-coder-008`.

```bash
cd /home/hkang/zthunder_agent/SkyRL
export PATH=/home/hkang/zthunder_agent/SkyRL/.venv/bin:$HOME/.local/bin:$PATH

export RUN_NAME=codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633
export RAY_HEAD_IP=172.21.44.54
export ROLLOUT_SERVER_PORTS_CSV=18000,18001,18002,18003
export TP_SIZE=2
export ROLLOUT_NOFILE_SOFT=131072

bash /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/start_qwen3_32b_external_rollout_servers.sh
```

This produced:

- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/rollout/rollout_a.log`
- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/rollout/rollout_b.log`
- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/rollout/rollout_c.log`
- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/rollout/rollout_d.log`

## 6. Trainer Monitor Shell

These commands were run from a shell inside the same Slurm allocation to launch the trainer monitors.

```bash
cd /home/hkang/zthunder_agent/SkyRL
export PATH=/home/hkang/zthunder_agent/SkyRL/.venv/bin:$HOME/.local/bin:$PATH

export JOB_ID=1144
export RUN_NAME=codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633
export TRAINER_NODES_CSV=research-dev-coder-012,research-dev-coder-013,research-dev-coder-014,research-dev-coder-015

JOB_ID="$JOB_ID" \
TRAINER_NODES_CSV="$TRAINER_NODES_CSV" \
RUN_NAME="$RUN_NAME" \
bash /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/start_trainer_node_monitors.sh
```

This produced:

- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/trainer_monitors/research-dev-coder-012`
- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/trainer_monitors/research-dev-coder-013`
- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/trainer_monitors/research-dev-coder-014`
- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/trainer_monitors/research-dev-coder-015`

## 7. Monitoring Commands Actually Used

These are the concrete monitoring commands used against this run.

All examples below assume:

```bash
export RUN_NAME=codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633
export LOG_DIR=/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME
export RUN_ARTIFACT_DIR=/home/hkang/zthunder_agent/$RUN_NAME
export HEAD_IP=172.21.44.54
export JOB_ID=1144
```

### 7.1 Trial Progress

```bash
tail -n 10 "$LOG_DIR/monitoring/trial_progress.tsv"
```

### 7.2 Rollout Metrics

```bash
tail -n 8 "$LOG_DIR/rollout/monitoring/vllm_metrics.tsv"
```

### 7.3 Main Trainer Log

```bash
tail -n 40 "$LOG_DIR/launcher_train_driver.log"
```

### 7.4 Router State Endpoint

```bash
curl -sS --max-time 5 "http://${HEAD_IP}:8080/router_state" | head -c 1000
```

### 7.5 Slurm Step State

```bash
squeue -s -j "$JOB_ID"
```

### 7.6 Key Error Scan In The Main Log

```bash
grep -nE "Started: 'step'|Finished: 'step'|Started: 'sync_weights'|Finished: 'sync_weights'|All outputs are loss masked|has no valid backend|ReadTimeout|ReadError|Too many open files|Errno 24" \
  "$LOG_DIR/launcher_train_driver.log" | tail -n 120
```

### 7.7 Direct Trial File Counts

```bash
find "$RUN_ARTIFACT_DIR/trials_run" -maxdepth 2 -name result.json | wc -l
find "$RUN_ARTIFACT_DIR/trials_run" -maxdepth 2 -name exception.txt | wc -l
find "$RUN_ARTIFACT_DIR/trials_run" -maxdepth 2 -name '*.trajectory.json' | wc -l
```

### 7.8 ThunderAgent Log Tail

```bash
tail -n 120 "$LOG_DIR/thunderagent.log"
```

### 7.9 Specific Program Lifecycle Grep

This was used during debugging of the router state problem.

```bash
grep -n "fb63e6639936494ba81a0744cb7ab512" "$LOG_DIR/thunderagent.log" | tail -n 80
grep -n "fb63e6639936494ba81a0744cb7ab512" "$LOG_DIR/launcher_train_driver.log" | tail -n 80
```

## 8. Outcome Reached By This Launch

This run got through startup correctly, filled the generation buffer, ran the first training update, and then hung during the first `sync_weights` boundary.

The important timestamps from the main log were:

- `2026-03-17 04:56:37 -0700`: `Started: 'step'`
- `2026-03-17 05:20:24 -0700`: `Started: 'sync_weights'`
- `2026-03-17 05:20:29 -0700`: ThunderAgent router began repeated `pause_until_safe` logs
- after that: no further forward progress, repeated `/router_state` scrape timeouts

Main log:

- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/launcher_train_driver.log`

ThunderAgent log:

- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/thunderagent.log`

## 9. Exact Stop Commands Used

### 9.1 Stop The Train Driver

In the persistent train shell on `003`, the stop action used was:

```text
Ctrl-C
```

That returned the shell prompt on `research-dev-coder-003`.

### 9.2 Cancel The Child Steps Without Cancelling The Whole Allocation

These were the exact `scancel` commands used after the train shell returned:

```bash
scancel 1144.698 1144.699 1144.700 1144.701 1144.702 1144.703 1144.704 1144.705 1144.706 1144.707
```

Then the remaining head-side bash step was cancelled with:

```bash
scancel 1144.709
```

### 9.3 Verify Only The Allocation Remains

```bash
squeue -s -j 1144
```

Expected post-stop state for this run:

- `1144.batch` remains
- no child run steps remain

## 10. Files Produced By This Run

Top-level logs:

- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/launcher_rootless_docker.log`
- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/launcher_train_driver.log`
- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/infra-260317_045321.log`
- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/thunderagent.log`

Monitoring:

- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/monitoring`
- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/rollout/monitoring`
- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/trainer_monitors`

Artifacts:

- `/home/hkang/zthunder_agent/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/trials_run`

## 11. What This Document Is

This document is a literal command record for the final `rerunfixfd` launch path:

- rootless Docker command used
- Ray command used
- rollout command used
- trainer monitor command used
- train driver command used
- monitoring commands used
- stop commands used

It does not include the earlier wrong launch attempts that happened before `rerunfixfd`.
