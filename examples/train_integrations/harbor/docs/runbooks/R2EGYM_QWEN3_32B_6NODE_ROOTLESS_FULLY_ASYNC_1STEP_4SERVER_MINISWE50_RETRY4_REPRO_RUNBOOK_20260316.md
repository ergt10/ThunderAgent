# R2EGYM QWEN3-32B 6-Node Rootless Fully Async 1-Step 4-Server mini-swe-agent-50 Repro Runbook

Date: 2026-03-16

This document records the exact workflow used for the mini-swe-agent reproduction run:

- `r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112`

This is not a successful training run. It is a stable reproduction of a Harbor integration bug:

- `mini-swe-agent` returns `AgentContext(... metadata=None)`
- Harbor postprocess assumes `metadata["all_messages"]`, `metadata["summarization_count"]`, and `metadata["n_episodes"]` exist
- the run then fails in `harbor_agent_loop()` with:
  - `'NoneType' object is not subscriptable`

The same run also later reproduces a second head-side failure:

- Harbor trial setup eventually hits:
  - `OSError: [Errno 24] Too many open files`
- the failing path is Harbor Docker environment startup and file upload subprocess creation
- this is separate from the `mini-swe-agent metadata=None` mismatch

Per the reproduction goal for this document, ignore the separate `RewardFileNotFoundError` signals. They are present in the same run, but they are not the primary issue tracked here.

Primary logs:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112`

Primary artifacts:

- `/home/hkang/zthunder_agent/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112`

Parent runbook:

- `/home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/R2EGYM_QWEN3_32B_6NODE_ROOTLESS_FULLY_ASYNC_1STEP_4SERVER_TAOBS_REPRO_RUNBOOK_20260316.md`

## 1. Scope

This runbook reproduces exactly these deltas on top of the parent 1-step TA-observability run:

1. agent changed from `terminus-2` to `mini-swe-agent`
2. interaction limit changed from `10` to `50`
3. the run still used:
   - rootless Docker on head `003`
   - 4 rollout servers on `008`
   - `TP=2`
   - `max_train_tasks=64`
   - `resume_mode=none`
   - the same `r2egym` four-split dataset list for train and eval
4. the run captured the same TA observability TSVs and rollout metrics

## 2. Historical Topology

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

## 3. One-Time System Prerequisite

This reproduction required restoring rootless Docker subuid/subgid ranges for user `hkang`.

The missing state was:

- `/etc/subuid` and `/etc/subgid` had no `hkang:` entry

The restored entries were:

```text
hkang:165536:65536
```

Backups used during the restore:

- `/etc/subuid.bak.codex_20260316`
- `/etc/subgid.bak.codex_20260316`

Without this fix, fresh rootless Docker startup on `003` failed with:

- `No subuid ranges found for user 243001624 ("hkang")`

## 4. Shared Variables

```bash
cd /home/hkang/zthunder_agent/SkyRL

export JOB_ID=1144
export REPO=/home/hkang/zthunder_agent/SkyRL
export RUN_NAME=r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112

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

export TRAIN_DATA="['/home/hkang/zthunder_agent/data/harbor/r2egym-trivial','/home/hkang/zthunder_agent/data/harbor/r2egym-easy','/home/hkang/zthunder_agent/data/harbor/r2egym-medium','/home/hkang/zthunder_agent/data/harbor/r2egym-hard']"
export EVAL_DATA="$TRAIN_DATA"

export LOG_DIR=/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME
export RUN_ARTIFACT_ROOT=/home/hkang/zthunder_agent
export ROLLOUT_SERVER_PORTS_CSV=18000,18001,18002,18003
export ROLLOUT_ENGINES=4
export ROLLOUT_TP_SIZE=2
```

## 5. Start Rootless Docker On Head 003

Run from `research-dev-coder-003` inside the Slurm allocation:

```bash
export PATH=/home/hkang/zthunder_agent/SkyRL/.venv/bin:$HOME/.local/bin:$PATH

bash /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/start_rootless_docker_for_harbor.sh \
  2>&1 | tee "$LOG_DIR/launcher_rootless_docker.log"
```

Expected healthy output:

- `Rootless Docker ready`
- `docker_host: unix:///tmp/xdg-mswe50k-hkang/docker.sock`
- `nofile: soft=131072 hard=131072`

Evidence:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/launcher_rootless_docker.log`

## 6. Start Ray

Head:

```bash
JOB_ID="$JOB_ID" \
HEAD_NODE="$HEAD_NODE" \
TRAINER_NODES_CSV="$TRAINER_NODES_CSV" \
RAY_PORT="$RAY_PORT" \
RAY_START_MODE=block \
bash /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/launch_qwen3_32b_ray_cluster.sh \
  2>&1 | tee "$LOG_DIR/launcher_ray_head.log"
```

Workers must remain alive on:

- `research-dev-coder-012`
- `research-dev-coder-013`
- `research-dev-coder-014`
- `research-dev-coder-015`

Expected healthy signal:

- `Ray runtime started.`
- `--block`

Evidence:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/launcher_ray_head.log`

## 7. Start The Four Rollout Servers On 008

Run from `research-dev-coder-008`:

```bash
cd /home/hkang/zthunder_agent/SkyRL
export PATH=/home/hkang/zthunder_agent/SkyRL/.venv/bin:$HOME/.local/bin:$PATH
export RUN_NAME="$RUN_NAME"
export RAY_HEAD_IP="$HEAD_IP"
export ROLLOUT_SERVER_PORTS_CSV=18000,18001,18002,18003
export TP_SIZE=2

bash /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/start_qwen3_32b_external_rollout_servers.sh
```

Expected healthy output:

- `rollout_a healthy`
- `rollout_b healthy`
- `rollout_c healthy`
- `rollout_d healthy`
- `External rollout servers ready`

Per-server logs:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/rollout/rollout_a.log`
- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/rollout/rollout_b.log`
- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/rollout/rollout_c.log`
- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/rollout/rollout_d.log`

## 8. Start Trainer Monitors

```bash
JOB_ID="$JOB_ID" \
TRAINER_NODES_CSV="$TRAINER_NODES_CSV" \
RUN_NAME="$RUN_NAME" \
bash /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/start_trainer_node_monitors.sh
```

Output directory:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/trainer_monitors`

## 9. Launch The Train Driver

Run from the persistent shell on head `003`:

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

This is the exact agent switch:

- `harbor_trial_config.agent.name=mini-swe-agent`
- `harbor_trial_config.agent.kwargs.max_turns=50`

This keeps the run at one step by setting:

- `max_train_tasks=64`

## 10. Monitoring Files To Keep

Main train log:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/launcher_train_driver.log`

ThunderAgent router log:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/thunderagent.log`

Head-side monitoring TSVs:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/monitoring/thunderagent_backend_state.tsv`
- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/monitoring/thunderagent_program_state.tsv`
- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/monitoring/thunderagent_events.tsv`
- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/monitoring/trial_progress.tsv`

Rollout-side metrics and logs:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/rollout/monitoring/vllm_metrics.tsv`
- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/rollout/rollout_a.log`
- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/rollout/rollout_b.log`
- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/rollout/rollout_c.log`
- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/rollout/rollout_d.log`

## 11. What Reproduces Reliably

The run reliably reaches:

- `Initialized weight sync state for policy model and inference engines.`
- `Started: 'sync_weights_to_inference_engines'`
- `Started: 'step'`
- `Started: 'wait_for_generation_buffer'`
- `trial_dirs=256` very early in the step
- all 4 rollout backends receiving load

Primary log locations:

- step start:
  - [launcher_train_driver.log](/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/launcher_train_driver.log#L708)
  - [launcher_train_driver.log](/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/launcher_train_driver.log#L709)
- first successful 256 trial-directory fill:
  - `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/monitoring/trial_progress.tsv`

## 12. Primary Problem Reproduced

The main bug reproduced by this run is:

- `mini-swe-agent` returns `AgentContext(... metadata=None)`
- Harbor generator assumes metadata exists
- Harbor generator then crashes that trajectory with:
  - `'NoneType' object is not subscriptable`

Representative first error in the main log:

- [launcher_train_driver.log](/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/launcher_train_driver.log#L712)

Representative repeat on the second attempt of the same task family:

- [launcher_train_driver.log](/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/launcher_train_driver.log#L1984)

The logged proof is explicit inside the serialized `TrialResult`:

- `agent_result=AgentContext(... rollout_details=None, metadata=None)`
- `verifier_result=VerifierResult(rewards={'reward': 1.0})`

This proves the trial itself can finish and even get a verifier reward, but Harbor still rejects it afterwards because the wrapper did not provide Harbor-required metadata.

Code locations for the broken interface:

- Harbor assumes metadata exists at:
  - [harbor_generator.py](/home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/harbor_generator.py#L574)
  - [harbor_generator.py](/home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/harbor_generator.py#L575)
  - [harbor_generator.py](/home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/harbor_generator.py#L576)

- mini-swe-agent wrapper populates token counts but does not set `context.metadata`:
  - [mini_swe_agent.py](/home/hkang/zthunder_agent/SkyRL/.venv/lib/python3.13/site-packages/harbor/agents/installed/mini_swe_agent.py#L344)

This is the bug to fix first if the goal is to make `mini-swe-agent` usable under the Harbor generator path.

## 13. Second Problem Reproduced

The same run later reproduces a separate head-side failure:

- `OSError: [Errno 24] Too many open files`

This is not the ThunderAgent profiling bug from an earlier retry. In this run, the stack is inside Harbor trial setup:

- `harbor.trial.trial._setup_environment()`
- `harbor.environments.docker.docker._run_docker_compose_command(["build"])`
- `asyncio.create_subprocess_exec(...)`

Representative first occurrence:

- [launcher_train_driver.log](/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/launcher_train_driver.log#L3199)

Representative second-attempt occurrence during agent setup file upload:

- [launcher_train_driver.log](/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/launcher_train_driver.log#L3287)

Relevant code locations:

- [trial.py](/home/hkang/zthunder_agent/SkyRL/.venv/lib/python3.13/site-packages/harbor/trial/trial.py#L470)
- [docker.py](/home/hkang/zthunder_agent/SkyRL/.venv/lib/python3.13/site-packages/harbor/environments/docker/docker.py#L180)
- [docker.py](/home/hkang/zthunder_agent/SkyRL/.venv/lib/python3.13/site-packages/harbor/environments/docker/docker.py#L267)

This second issue should be tracked separately from the `mini-swe-agent metadata=None` interface bug.

## 14. Secondary Observation That Did Not Regress

This run did not reproduce the previous infrastructure failures:

- no `Failed to release ThunderAgent program_id=...`
- no `Errno 24`
- no `Server disconnected`
- no `SIGABRT`

Representative successful release lines:

- [launcher_train_driver.log](/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/launcher_train_driver.log#L713)
- [launcher_train_driver.log](/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/launcher_train_driver.log#L1985)

So the reproduced failure is not a ThunderAgent release-path regression. It is a Harbor wrapper mismatch.

## 15. Useful Trial Artifact Locations

Representative result files:

- `/home/hkang/zthunder_agent/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/trials_run/r2egym-0060__D847W7L/result.json`
- `/home/hkang/zthunder_agent/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/trials_run/r2egym-0060__z6gvtdT/result.json`

Representative exception files:

- `/home/hkang/zthunder_agent/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/trials_run/r2egym-0060__D847W7L/exception.txt`
- `/home/hkang/zthunder_agent/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-retry4-20260316_202112/trials_run/r2egym-0060__z6gvtdT/exception.txt`

These are useful when verifying whether a specific trial failed inside the task, inside verifier, or in Harbor postprocess.

## 16. Minimal Repair Targets

To make this experiment succeed as intended, the first repair target is not infrastructure. It is:

- teach `mini-swe-agent` to populate the Harbor-required metadata fields:
  - `all_messages`
  - `summarization_count`
  - `n_episodes`

Without that, the run remains reproducible but invalid for training, because trials that otherwise finished successfully will still be retried or marked failed by Harbor.

After that, the second repair target is the head-side file-descriptor pressure that eventually trips Harbor Docker subprocess creation with:

- `OSError: [Errno 24] Too many open files`
