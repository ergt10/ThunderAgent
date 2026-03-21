# R2EGYM Qwen3-32B 6-Node Rootless Fully Async 1-Step 4-Server mini-swe-agent-50 timeout9000 fd131072 `blockdockerd` Actual Command Runbook

Date: 2026-03-17

This document records the exact launch contract and the best reconstructed operational commands for the stopped experiment:

- `r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051`

This is a historical run record. It reflects the repo, helper scripts, and local Harbor dataset tree as they existed at launch time. Later R2EGYM verifier hotfixes are outside the scope of this runbook.

Important precision note:

- Section 4 is the exact saved wrapper command for the historical launch.
- Section 5 is a replay-oriented decomposition reconstructed from that wrapper plus the helper scripts on disk.
- Section 5 is not a verbatim shell-history transcript of every terminal used that night.

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
- Ray address: `172.21.44.54:6381`
- Rollout ports:
  - `18000`
  - `18001`
  - `18002`
  - `18003`

Primary logs:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051`

Primary artifacts:

- `/home/hkang/zthunder_agent/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051`

Launch wrapper saved on disk:

- `/home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/run_r2egym_qwen3_32b_6node_rootless_full_1step_miniswe50_timeout9000_fd131072_blockdockerd_20260317.sh`

## 2. Repo State Assumed At Launch

No runtime patch command was executed during this launch. The run relied on the repo state already on disk, including these helper-script behaviors:

- `examples/train_integrations/harbor/launch_qwen3_32b_ray_cluster.sh`
  - `run_on_node()` uses `srun --exact --gpus-per-node=0`
  - `run_on_node_bg()` uses `srun --exact --gpus-per-node=0`
- `examples/train_integrations/harbor/start_trainer_node_monitors.sh`
  - trainer monitor `srun` uses `--exact`

The run used only the repo-local environment:

- `/home/hkang/zthunder_agent/SkyRL/.venv`

## 3. Shared Variables

These are the exact environment values encoded in the saved launch wrapper.

```bash
cd /home/hkang/zthunder_agent/SkyRL
export PATH=/home/hkang/zthunder_agent/SkyRL/.venv/bin:$HOME/.local/bin:$PATH

export TRAIN_DATA="['/home/hkang/zthunder_agent/data/harbor/r2egym-trivial','/home/hkang/zthunder_agent/data/harbor/r2egym-easy','/home/hkang/zthunder_agent/data/harbor/r2egym-medium','/home/hkang/zthunder_agent/data/harbor/r2egym-hard']"
export EVAL_DATA="$TRAIN_DATA"
export RAY_ADDRESS="172.21.44.54:6381"
export RAY_HEAD_IP="172.21.44.54"
export ROLLOUT_HOST_IP="172.21.44.94"
export ROLLOUT_SERVER_PORTS_CSV="18000,18001,18002,18003"
export ROLLOUT_ENGINES=4
export ROLLOUT_TP_SIZE=2
export DOCKER_MODE=rootless
export RUN_NAME_OVERRIDE="r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051"
export LOG_DIR_OVERRIDE="/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051"
export RUN_ARTIFACT_ROOT="/home/hkang/zthunder_agent"
export SCRATCH_ROOT="/tmp/hkang/r2e1s-0317220051b-train-runtime"
export XDG_RUNTIME_DIR="/tmp/xdg-r2e1s-0317220051b"
export DOCKER_HOST="unix:///tmp/xdg-r2e1s-0317220051b/docker.sock"
export DOCKER_PIDFILE="/tmp/xdg-r2e1s-0317220051b/docker.pid"
export DOCKER_EXEC_ROOT="/tmp/hkang/r2e1s-0317220051b-rootless-exec"
export DOCKER_DATA_ROOT="/scratch/triton_cache/hkang/r2e1s-0317220051b-rootless-data"
export HEAD_NOFILE_SOFT=131072
export ROLLOUT_METRICS_ENDPOINT_SPECS="rollout_a.log=http://172.21.44.94:18000;rollout_b.log=http://172.21.44.94:18001;rollout_c.log=http://172.21.44.94:18002;rollout_d.log=http://172.21.44.94:18003"
export RUN_PREFLIGHT_CHECKS=false
```

## 4. Single-Wrapper Launch Contract

The saved wrapper is the exact experiment contract:

```bash
cd /home/hkang/zthunder_agent/SkyRL
bash /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/run_r2egym_qwen3_32b_6node_rootless_full_1step_miniswe50_timeout9000_fd131072_blockdockerd_20260317.sh
```

Its final `exec` target is:

```bash
stdbuf -oL -eL \
  bash examples/train_integrations/harbor/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh full \
    max_train_tasks=64 \
    trainer.resume_mode=none \
    harbor_trial_config.agent.name=mini-swe-agent \
    harbor_trial_config.agent.kwargs.max_turns=50 \
    harbor_trial_config.agent.override_timeout_sec=9000
```

The allocation-local shell mirrored launcher output into:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051/infra-260317_222013.log`

## 5. Replay Decomposition Of The Wrapper

This is the manual decomposition of the same run contract, matching the helper scripts and environment values used by the wrapper.

It is intended to be operator-friendly and replayable. It should not be treated as stronger evidence than the saved wrapper in Section 4.

### 5.1 Head Shell On `003`

```bash
cd /home/hkang/zthunder_agent/SkyRL
export PATH=/home/hkang/zthunder_agent/SkyRL/.venv/bin:$HOME/.local/bin:$PATH

export JOB_ID=1144
export HEAD_NODE=research-dev-coder-003
export ROLLOUT_NODE=research-dev-coder-008
export TRAINER_NODES_CSV=research-dev-coder-012,research-dev-coder-013,research-dev-coder-014,research-dev-coder-015

export HEAD_IP=172.21.44.54
export ROLLOUT_IP=172.21.44.94
export RAY_PORT=6381
export RAY_ADDRESS=${HEAD_IP}:${RAY_PORT}

export RUN_NAME=r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051
export LOG_DIR=/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME
export RUN_ARTIFACT_ROOT=/home/hkang/zthunder_agent
export RUN_ARTIFACT_DIR=$RUN_ARTIFACT_ROOT/$RUN_NAME

export DOCKER_MODE=rootless
export XDG_RUNTIME_DIR=/tmp/xdg-r2e1s-0317220051b
export DOCKER_HOST=unix:///tmp/xdg-r2e1s-0317220051b/docker.sock
export DOCKER_PIDFILE=/tmp/xdg-r2e1s-0317220051b/docker.pid
export DOCKER_EXEC_ROOT=/tmp/hkang/r2e1s-0317220051b-rootless-exec
export SCRATCH_ROOT=/tmp/hkang/r2e1s-0317220051b-train-runtime
export DOCKER_DATA_ROOT=/scratch/triton_cache/hkang/r2e1s-0317220051b-rootless-data
export DOCKER_LOG_PATH=$LOG_DIR/dockerd_rootless_harbor.log

export HEAD_NOFILE_SOFT=131072
export TRAINER_NOFILE_SOFT=131072
export ROLLOUT_NOFILE_SOFT=131072

export TRAIN_DATA="['/home/hkang/zthunder_agent/data/harbor/r2egym-trivial','/home/hkang/zthunder_agent/data/harbor/r2egym-easy','/home/hkang/zthunder_agent/data/harbor/r2egym-medium','/home/hkang/zthunder_agent/data/harbor/r2egym-hard']"
export EVAL_DATA="$TRAIN_DATA"
export ROLLOUT_SERVER_PORTS_CSV=18000,18001,18002,18003
export ROLLOUT_ENGINES=4
export ROLLOUT_TP_SIZE=2

mkdir -p "$LOG_DIR"
```

### 5.2 Start Rootless Docker On `003`

```bash
DOCKER_NOFILE_SOFT="$HEAD_NOFILE_SOFT" \
ROOTLESS_DOCKER_START_MODE=background \
bash /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/start_rootless_docker_for_harbor.sh
```

Persistent daemon log:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051/dockerd_rootless_harbor.log`

### 5.3 Start Ray Head And Trainer Workers

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

### 5.4 Rollout Shell On `008`

```bash
cd /home/hkang/zthunder_agent/SkyRL
export PATH=/home/hkang/zthunder_agent/SkyRL/.venv/bin:$HOME/.local/bin:$PATH

export RUN_NAME=r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051
export RAY_HEAD_IP=172.21.44.54
export ROLLOUT_SERVER_PORTS_CSV=18000,18001,18002,18003
export TP_SIZE=2
export ROLLOUT_NOFILE_SOFT=131072

bash /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/start_qwen3_32b_external_rollout_servers.sh
```

Per-server logs:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051/rollout/rollout_a.log`
- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051/rollout/rollout_b.log`
- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051/rollout/rollout_c.log`
- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051/rollout/rollout_d.log`

### 5.5 Start Trainer Monitors

```bash
cd /home/hkang/zthunder_agent/SkyRL
export PATH=/home/hkang/zthunder_agent/SkyRL/.venv/bin:$HOME/.local/bin:$PATH

export JOB_ID=1144
export RUN_NAME=r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051
export TRAINER_NODES_CSV=research-dev-coder-012,research-dev-coder-013,research-dev-coder-014,research-dev-coder-015

JOB_ID="$JOB_ID" \
TRAINER_NODES_CSV="$TRAINER_NODES_CSV" \
RUN_NAME="$RUN_NAME" \
bash /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/start_trainer_node_monitors.sh
```

Monitor output root:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051/trainer_monitors`

### 5.6 Launch The Train Driver On `003`

```bash
cd /home/hkang/zthunder_agent/SkyRL
export PATH=/home/hkang/zthunder_agent/SkyRL/.venv/bin:$HOME/.local/bin:$PATH

export TRAIN_DATA="['/home/hkang/zthunder_agent/data/harbor/r2egym-trivial','/home/hkang/zthunder_agent/data/harbor/r2egym-easy','/home/hkang/zthunder_agent/data/harbor/r2egym-medium','/home/hkang/zthunder_agent/data/harbor/r2egym-hard']"
export EVAL_DATA="$TRAIN_DATA"
export RAY_ADDRESS=172.21.44.54:6381
export RAY_HEAD_IP=172.21.44.54
export ROLLOUT_HOST_IP=172.21.44.94
export ROLLOUT_SERVER_PORTS_CSV=18000,18001,18002,18003
export ROLLOUT_ENGINES=4
export ROLLOUT_TP_SIZE=2
export DOCKER_MODE=rootless
export RUN_NAME_OVERRIDE=r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051
export LOG_DIR_OVERRIDE=/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051
export RUN_ARTIFACT_ROOT=/home/hkang/zthunder_agent
export SCRATCH_ROOT=/tmp/hkang/r2e1s-0317220051b-train-runtime
export XDG_RUNTIME_DIR=/tmp/xdg-r2e1s-0317220051b
export DOCKER_HOST=unix:///tmp/xdg-r2e1s-0317220051b/docker.sock
export DOCKER_PIDFILE=/tmp/xdg-r2e1s-0317220051b/docker.pid
export DOCKER_EXEC_ROOT=/tmp/hkang/r2e1s-0317220051b-rootless-exec
export DOCKER_DATA_ROOT=/scratch/triton_cache/hkang/r2e1s-0317220051b-rootless-data
export HEAD_NOFILE_SOFT=131072
export ROLLOUT_METRICS_ENDPOINT_SPECS="rollout_a.log=http://172.21.44.94:18000;rollout_b.log=http://172.21.44.94:18001;rollout_c.log=http://172.21.44.94:18002;rollout_d.log=http://172.21.44.94:18003"
export RUN_PREFLIGHT_CHECKS=false

stdbuf -oL -eL \
  bash /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh full \
    max_train_tasks=64 \
    trainer.resume_mode=none \
    harbor_trial_config.agent.name=mini-swe-agent \
    harbor_trial_config.agent.kwargs.max_turns=50 \
    harbor_trial_config.agent.override_timeout_sec=9000 \
  2>&1 | tee /home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051/launcher_train_driver.log
```

## 6. Monitoring Commands Used

All commands below assume:

```bash
export RUN_NAME=r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051
export LOG_DIR=/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME
export RUN_ARTIFACT_DIR=/home/hkang/zthunder_agent/$RUN_NAME
```

Trial progress:

```bash
tail -n 10 "$LOG_DIR/monitoring/trial_progress.tsv"
```

Harbor Docker startup timeouts:

```bash
grep -n "EnvironmentStartTimeoutError" "$LOG_DIR/launcher_train_driver.log"
```

mini-swe-agent rollout-details postprocess failures:

```bash
grep -n "did not return assistant logprobs/token ids" "$LOG_DIR/launcher_train_driver.log"
```

ThunderAgent backend timeouts:

```bash
grep -n "failed after" "$LOG_DIR/thunderagent.log"
grep -n "httpx.ReadTimeout\\|httpcore.ReadTimeout" "$LOG_DIR/launcher_train_driver.log"
```

Representative stop-state snapshot at the end of this run:

```text
2026-03-17T23:34:03-0700 ... trial_dirs=379 result_json_count=336 exception_txt_count=126 trajectory_json_count=210 completed_trials_count=336
```

Source:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051/monitoring/trial_progress.tsv`

## 7. Stop And Cleanup Commands Used

The run was stopped by cancelling the non-batch Slurm steps only. The allocation itself was intentionally left alive.

```bash
scancel 1144.814 1144.808 1144.809 1144.810 1144.811 1144.812 1144.806 1144.800 1144.801 1144.802 1144.803 1144.804
```

Verification:

```bash
sacct -j 1144.800,1144.801,1144.802,1144.803,1144.804,1144.806,1144.808,1144.809,1144.810,1144.811,1144.812,1144.814 \
  --format=JobID,JobName%40,State,ExitCode --parsable2
```

Observed final state:

```text
1144.800-804  CANCELLED by 243001624
1144.806      CANCELLED by 243001624
1144.808      CANCELLED by 243001624
1144.809-812  CANCELLED by 243001624
1144.814      CANCELLED by 243001624
```

## 8. Files To Preserve

Main driver log:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051/launcher_train_driver.log`

Infra shell capture:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051/infra-260317_222013.log`

ThunderAgent log:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051/thunderagent.log`

Rootless dockerd log:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051/dockerd_rootless_harbor.log`

Head-side monitoring TSVs:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051/monitoring/trial_progress.tsv`
- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051/monitoring/thunderagent_backend_state.tsv`
- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051/monitoring/thunderagent_program_state.tsv`
- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051/monitoring/thunderagent_events.tsv`

Artifacts:

- `/home/hkang/zthunder_agent/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051`
