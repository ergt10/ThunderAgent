# R2EGYM QWEN3-32B 6-Node Rootless Fully Async 1-Epoch 4-Server Repro Runbook

Date: 2026-03-16

This document reconstructs the exact workflow used for the historical run:

- `r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424`

It is based on the real logs under:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424`

This is not the "minimum safe shortcut" path. It is the actual tested path, including the full validation step that was run before the train driver.

Boundary note:

- the exact top-level interactive shell history was not saved
- the commands below are the closest equivalent command chain that matches the observed launcher logs and the effective runtime configuration

## 1. Scope

This runbook covers:

1. dataset preparation for the four `r2egym` splits
2. exact launch order inside an existing 6-node Slurm allocation
3. cleanup of old checkpoints and old run artifacts before reusing the same run name
4. rollout, trainer, and head-side monitoring
5. post-run analysis commands
6. the exact failure signature of the historical run for comparison

## 2. Historical Topology

The historical run used:

- Slurm allocation: `1144`
- batch script: `/home/hkang/zthunder_agent/skyrl.sbatch`
- reservation: `datagen`
- head: `research-dev-coder-003`
- rollout: `research-dev-coder-008`
- trainers: `research-dev-coder-012,research-dev-coder-013,research-dev-coder-014,research-dev-coder-015`
- Ray head address: `172.21.44.54:6381`
- rollout host IP: `172.21.44.94`
- rollout backend URLs:
  - `http://172.21.44.94:18000`
  - `http://172.21.44.94:18001`
  - `http://172.21.44.94:18002`
  - `http://172.21.44.94:18003`

Primary evidence:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424/launcher_ray.log`
- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424/launcher_validation.log`
- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424/launcher_rollout.log`

If Slurm gives you different nodes on a future run, keep the same role split:

- 1 head CPU node
- 1 rollout GPU node
- 4 trainer GPU nodes

and update the node/IP variables below accordingly.

## 3. Historical Effective Configuration

The train driver actually ran with these important settings:

- `trainer.run_name=r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424`
- `trainer.resume_mode=none`
- `trainer.epochs=1`
- `max_train_tasks=null`
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
- `generator.inference_engine.max_num_batched_tokens=8192`

Dataset inputs:

- train:
  - `/home/hkang/zthunder_agent/data/harbor/r2egym-trivial`
  - `/home/hkang/zthunder_agent/data/harbor/r2egym-easy`
  - `/home/hkang/zthunder_agent/data/harbor/r2egym-medium`
  - `/home/hkang/zthunder_agent/data/harbor/r2egym-hard`
- eval:
  - same four directories

Derived training size:

- total valid train tasks: `701`
- `train_batch_size=64`
- `epochs=1`
- number of steps per epoch: `10`
- total training steps: `10`

Primary evidence:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424/launcher_train_driver.log`

## 4. Paths

Model:

- `/data/zy/models/hkang/models/Qwen3-32B`

Datasets:

- `/home/hkang/zthunder_agent/data/harbor/r2egym-trivial`
- `/home/hkang/zthunder_agent/data/harbor/r2egym-easy`
- `/home/hkang/zthunder_agent/data/harbor/r2egym-medium`
- `/home/hkang/zthunder_agent/data/harbor/r2egym-hard`

Run logs:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424`

Run artifacts:

- `/data/zy/models/hkang/harbor_runs/r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424`

Important subpaths:

- trials:
  - `/data/zy/models/hkang/harbor_runs/r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424/trials_run`
- checkpoints:
  - `/data/zy/models/hkang/harbor_runs/r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424/ckpts`
- exports:
  - `/data/zy/models/hkang/harbor_runs/r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424/exports`
- head monitor:
  - `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424/monitoring`
- rollout monitor:
  - `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424/rollout/monitoring`
- trainer monitors:
  - `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424/trainer_monitors`

## 5. Allocation

This runbook assumes you already have a 6-node allocation.

The historical allocation came from:

```bash
sbatch --reservation=datagen /home/hkang/zthunder_agent/skyrl.sbatch
```

Do not run the launch commands below outside an active 6-node allocation.

## 6. Prepare The Four R2EGYM Splits

If the Harbor-format task directories are not already present, prepare them once:

```bash
cd /home/hkang/zthunder_agent/SkyRL

for split in trivial easy medium hard; do
  /home/hkang/zthunder_agent/SkyRL/.venv/bin/python \
    /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/prepare_harbor_dataset.py \
    --dataset "DCAgent/exp-rdb-r2egym-${split}" \
    --output_dir "/home/hkang/zthunder_agent/data/harbor/r2egym-${split}"
done
```

Expected output directories:

- `/home/hkang/zthunder_agent/data/harbor/r2egym-trivial`
- `/home/hkang/zthunder_agent/data/harbor/r2egym-easy`
- `/home/hkang/zthunder_agent/data/harbor/r2egym-medium`
- `/home/hkang/zthunder_agent/data/harbor/r2egym-hard`

## 7. Shared Variables

```bash
cd /home/hkang/zthunder_agent/SkyRL

export JOB_ID=1144
export REPO=/home/hkang/zthunder_agent/SkyRL
export RUN_NAME=r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424

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

## 8. Clean Old Artifacts Before Reusing The Same Run Name

The historical reproduction request explicitly wanted old checkpoints removed before launch.

If you are reusing the same `RUN_NAME`, clean the old run state first:

```bash
rm -rf "$LOG_DIR"
rm -rf "$RUN_ARTIFACT_DIR"
mkdir -p "$LOG_DIR"
```

This removes:

- old checkpoints
- old exports
- old `trials_run`
- old tmp logs under the same run name

## 9. Actual Launch Order

The historical run used this order:

1. start rootless Docker on head `003`
2. start Ray on `003 + 012-015`
3. start 4 rollout servers on `008`
4. run full validation
5. start trainer monitors
6. launch the train driver

This order matters. Do not swap `Ray` and `rootless Docker`, and do not launch the train driver before rollout passes `/health`.

## 10. Start Rootless Docker On The Head Node

```bash
srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 \
  -w "$HEAD_NODE" --ntasks=1 --nodes=1 \
  bash -lc "cd '$REPO' && \
    export XDG_RUNTIME_DIR='$XDG_RUNTIME_DIR' && \
    export DOCKER_HOST='$DOCKER_HOST' && \
    export HEAD_NOFILE_SOFT='$HEAD_NOFILE_SOFT' && \
    export ROOTLESS_DOCKER_START_MODE=block && \
    bash examples/train_integrations/harbor/start_rootless_docker_for_harbor.sh" \
  2>&1 | tee "$LOG_DIR/launcher_rootless_docker.log"
```

Historical socket and runtime paths:

- socket: `/tmp/xdg-test-hkang/docker.sock`
- pidfile: `/tmp/xdg-test-hkang/docker.pid`
- data-root: `/tmp/hkang/docker-rootless/docker-rootless`

Historical confirmation:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424/launcher_rootless_docker.log`

## 11. Start Ray

```bash
JOB_ID="$JOB_ID" \
HEAD_NODE="$HEAD_NODE" \
TRAINER_NODES_CSV="$TRAINER_NODES_CSV" \
RAY_PORT="$RAY_PORT" \
RAY_START_MODE=block \
bash /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/launch_qwen3_32b_ray_cluster.sh \
  2>&1 | tee "$LOG_DIR/launcher_ray.log"
```

Expected result:

- Ray head at `172.21.44.54:6381`
- `harbor_head` resource present
- 4 trainer bundles available on `012-015`

Historical confirmation:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424/launcher_ray.log`

## 12. Start The Four Rollout Servers

This run used:

- 4 rollout servers
- ports `18000,18001,18002,18003`
- `tensor_parallel_size=2`
- GPU groups:
  - `rollout_a`: GPUs `0,1`
  - `rollout_b`: GPUs `2,3`
  - `rollout_c`: GPUs `4,5`
  - `rollout_d`: GPUs `6,7`

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

Important note:

- the rollout monitor starts before all servers are healthy
- early `ConnectionRefused` while `18000-18003` are still booting is expected
- the success condition is the later `rollout_[a-d] healthy at http://127.0.0.1:<port>` lines

Historical confirmation:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424/launcher_rollout.log`
- rollout logs:
  - `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424/rollout/rollout_a.log`
  - `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424/rollout/rollout_b.log`
  - `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424/rollout/rollout_c.log`
  - `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424/rollout/rollout_d.log`

## 13. Run Full Validation

The historical run did run full validation, with `r2egym` as both train and eval dataset spec:

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
TRAIN_DATA="$TRAIN_DATA" \
EVAL_DATA="$EVAL_DATA" \
ROLLOUT_SERVER_URLS='["http://172.21.44.94:18000","http://172.21.44.94:18001","http://172.21.44.94:18002","http://172.21.44.94:18003"]' \
bash /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/run_full_run_validation_suite.sh \
  2>&1 | tee "$LOG_DIR/launcher_validation.log"
```

Historical confirmation:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424/launcher_validation.log`

## 14. Start Trainer Monitors

```bash
JOB_ID="$JOB_ID" \
TRAINER_NODES_CSV="$TRAINER_NODES_CSV" \
RUN_NAME="$RUN_NAME" \
bash /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/start_trainer_node_monitors.sh \
  2>&1 | tee "$LOG_DIR/launcher_trainer_monitors.log"
```

This launches one monitor per trainer node under:

- `/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/trainer_monitors/research-dev-coder-012`
- `/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/trainer_monitors/research-dev-coder-013`
- `/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/trainer_monitors/research-dev-coder-014`
- `/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/trainer_monitors/research-dev-coder-015`

## 15. Launch The Train Driver

This is the exact reproduction command for the historical `r2egym` test:

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
    trainer.resume_mode=none \
  2>&1 | tee "$LOG_DIR/launcher_train_driver.log"
```

Why these overrides matter:

- `TRAIN_DATA` and `EVAL_DATA` must both be the four-directory Python list literal
- `ROLLOUT_SERVER_PORTS_CSV=18000,18001,18002,18003` is required because the script default is still the 2-server path
- `ROLLOUT_ENGINES=4` is required to align the generator config with the 4 rollout backends
- `ROLLOUT_TP_SIZE=2` is required because the historical run was 4 servers x 2-way TP, not 2 servers x 4-way TP
- `trainer.resume_mode=none` is required because the current script default is still `latest`

Expected historical side effects from this command:

- head-side monitor auto-starts in:
  - `/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/monitoring`
- ThunderAgent starts on head and proxies to:
  - `http://172.21.44.94:18000`
  - `http://172.21.44.94:18001`
  - `http://172.21.44.94:18002`
  - `http://172.21.44.94:18003`

## 16. What The Historical Run Actually Did

Observed early sequence:

- `01:30:57 PDT`: train dataset loaded, `701` total valid tasks
- `01:30:57 PDT`: eval dataset limited to `20` tasks
- `01:30:57 PDT`: `Number of steps per epoch: 10`
- `01:33:40 PDT`: `Started: 'step'`
- `01:33:40 PDT`: `Started: 'wait_for_generation_buffer'`
- the run never finished `wait_for_generation_buffer` for step 1

The run died during:

- `step 1`
- phase: `wait_for_generation_buffer`
- buffer progress reached only `46/64`

## 17. Historical Failure Signature

The failure signature of the historical run was:

1. `01:49:50 PDT`
   - Harbor-side program release warning:
   - `Failed to release ThunderAgent program_id=e2bcf55b50304932b62cf5d009800ff5 after 4 attempts (ReadError: ReadError(''))`
2. `01:50:21 PDT`
   - one LLM call failed with:
   - `litellm.InternalServerError: Hosted_vllmException - Server disconnected`
3. `01:51:13 PDT`
   - one trial timed out after `900s`:
   - `r2egym-hard/r2egym-4566`
4. `01:51:48 PDT`
   - main training worker `skyrl_entrypoint pid=1311258` received `SIGABRT`
   - log shows:
     - `*** SIGABRT received`
     - `Fatal Python error: Aborted`
5. after the main worker died, Ray retried the worker, but the retry failed with:
   - `Failed to create placement group with 4 bundles (requiring 32.0 GPUs, 32.0 CPUs total) in 180 seconds`

The important conclusion is:

- the historical run did not die because rollout was saturated
- it did not finish step 1
- the direct kill event was the head-side main worker `SIGABRT`
- the later placement group failure was a follow-on effect after the first worker died

Primary evidence:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424/launcher_train_driver.log`
- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424/thunderagent.log`
- `/data/zy/models/hkang/harbor_runs/r2egym-qwen3-32b-6node-rootless-full-1epoch-4srv-20260316_011424/trials_run/r2egym-4566__WD6Ywag/exception.txt`

## 18. Post-Run Analysis Commands

### 18.1 Timeline And Rollout Metrics

```bash
/home/hkang/zthunder_agent/SkyRL/.venv/bin/python \
  /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/analyze_qwen3_full_run_timeline.py \
  --run-name "$RUN_NAME" \
  --log-root /home/hkang/zthunder_agent/tmp_logs
```

This writes into:

- `/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/analysis`

Including:

- `timeline_overview.png`
- `trainer_memory.png`
- `rollout_memory.png`
- `rollout_kv_usage.png`
- `rollout_prefix_cache_hit_rate.png`
- `rollout_num_requests_running.png`
- `rollout_num_requests_waiting.png`
- `timeline_report.md`

### 18.2 Average Trial Stage Breakdown

```bash
/home/hkang/zthunder_agent/SkyRL/.venv/bin/python \
  /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/analyze_harbor_trial_stage_breakdown.py \
  --run-artifacts-dir "/data/zy/models/hkang/harbor_runs/$RUN_NAME" \
  --output-dir "/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/analysis"
```

This writes:

- `trial_stage_breakdown.png`
- `trial_stage_breakdown_report.md`

Important limitation of the current analysis code:

- it can split `environment_setup`, `agent_setup`, `agent_execution`, `verifier`, `llm_api`, and command wait budget
- it still leaves a residual bucket inside `agent_execution`
- it does not yet provide a perfect per-step, per-part wall-clock decomposition for every tool and every observation exchange

So this runbook fully reproduces the historical run and current monitoring stack, but not the future fine-grained step-timing instrumentation that was discussed later.

## 19. Targeted Cleanup Without Scanceling The Whole Allocation

If this run fails and you want to clean only this run's remnants while keeping the allocation:

1. stop trainer monitor steps
2. stop rollout step on `008`
3. stop Ray on `003,012,013,014,015`
4. stop rootless Docker only on the head
5. do not `scancel "$JOB_ID"`

Historical cleanup philosophy for this workflow:

- target explicit processes or `srun` steps
- keep the parent batch allocation if the user still wants the nodes

## 20. Key Differences From The Default Current Full Path

To reproduce this exact `r2egym` test, do not forget these differences from the default full path:

1. training data is the four-split `r2egym` list, not `CodeContests`
2. eval data is also the four-split `r2egym` list
3. rollout uses 4 servers, not 2
4. rollout TP size is `2`, not `4`
5. `trainer.resume_mode=none` is explicitly required
6. the tested path did include full validation before launch
