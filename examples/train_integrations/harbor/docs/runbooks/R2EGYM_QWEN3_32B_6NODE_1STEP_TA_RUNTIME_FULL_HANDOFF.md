# R2EGYM Qwen3-32B 6-Node 1-Step ThunderAgent Runtime-Full Handoff

This document is the minimum handoff for a new LLM on a new cluster.

Goal:
- reproduce the `R2EGYM + Harbor + fully-async + ThunderAgent + 6-node + 1-step` run
- using branch `ergt10/skyrl-ta-runtime-full`

## 1. Branch And Scope

Use branch:
- `ergt10/skyrl-ta-runtime-full`

This branch already contains:
- vendored ThunderAgent
- SkyRL <-> ThunderAgent integration core
- Harbor integration code
- Harbor launchers, docs, analysis assets
- repo-local Harbor runtime patch assets

Do not use the dirty workspace as the execution source of truth.

## 2. Read These Files First

Read in this order:

1. baseline runbook:
- `examples/train_integrations/harbor/docs/runbooks/R2EGYM_QWEN3_32B_6NODE_ROOTLESS_FULLY_ASYNC_1STEP_4SERVER_MINISWE50_TIMEOUT9000_FD131072_BLOCKDOCKERD_REPLAY_RUNBOOK_20260318.md`

2. delta runbook:
- `examples/train_integrations/harbor/docs/runbooks/R2EGYM_QWEN3_32B_6NODE_ROOTLESS_FULLY_ASYNC_1STEP_4SERVER_MINISWE50_TIMEOUT9000_FD131072_BLOCKDOCKERD_REPLAY_RUNBOOK_20260319.md`

3. actual replay wrapper:
- `examples/train_integrations/harbor/run_r2egym_qwen3_32b_6node_rootless_full_1step_miniswe50_timeout9000_fd131072_blockdockerd_replay.sh`

4. actual launcher called by the wrapper:
- `examples/train_integrations/harbor/run_r2egym_qwen3_32b_6node_rootless_full_1step_miniswe50_timeout9000.sh`

5. Harbor runtime patcher:
- `examples/train_integrations/harbor/ops/apply_harbor_runtime_patches.py`

6. readiness and validation:
- `examples/train_integrations/harbor/run_qwen3_32b_full_readiness_suite.sh`
- `examples/train_integrations/harbor/run_full_run_validation_suite.sh`

Important:
- the `20260319` runbook is not self-contained
- it is a delta relative to `20260318`

## 3. Target Topology

Expected topology:
- 1 head node
- 1 rollout node
- 4 trainer nodes

Expected rollout layout:
- 4 rollout servers
- ports: `18000,18001,18002,18003`
- tensor parallel size: `2`

Expected trainer layout:
- 4 trainer nodes
- 8 GPUs per trainer node

## 4. Required External Assets

These are not provided by the repo:

- Qwen3-32B model directory
- Harbor-formatted R2EGYM datasets
- pre-pulled task Docker images
- rootless Docker prerequisites on the head node

The launcher defaults are cluster-specific and must be overridden on a new cluster.

## 5. Environment Build

This branch does not rely on a `requirements.txt` freeze.

Use:
- `pyproject.toml`
- `uv.lock`

The branch includes:
- ThunderAgent extra
- local editable ThunderAgent dependency

So the environment should be built from the repo root using `uv`, not from an old freeze file.

## 6. Mandatory Harbor Patch Step

Before any readiness check or run, apply the Harbor runtime patches:

```bash
python examples/train_integrations/harbor/ops/apply_harbor_runtime_patches.py --backup
```

This patcher covers three required Harbor site-packages modifications:
- rootless Docker upload patch in `harbor/environments/docker/docker.py`
- shared mini-swe-agent fast path template
- shared Harbor docker compose base template

To verify:

```bash
python examples/train_integrations/harbor/ops/apply_harbor_runtime_patches.py --check
```

## 7. Variables That Must Be Overridden On A New Cluster

Do not trust the wrapper defaults for these:

- `JOB_ID`
- `HEAD_NODE`
- `ROLLOUT_NODE`
- `TRAINER_NODES_CSV`
- `TRAIN_DATA`
- `EVAL_DATA`
- `MODEL_PATH`
- `DOCKER_DATA_ROOT`
- `HARBOR_SHARED_UV_CACHE_HOST_DIR`
- `HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME`
- `HARBOR_SHARED_UV_PYTHON_HOST_DIR`

The shipped defaults still point at the original cluster:
- `research-dev-coder-003`
- `research-dev-coder-008`
- `research-dev-coder-012,013,014,015`
- `/home/hkang/zthunder_agent/data/harbor/...`
- `/scratch/triton_cache/...`

## 8. Fixed Runtime Assumptions To Preserve

Keep these unless the new cluster forces a change:

- `DOCKER_MODE=rootful`
- `DOCKER_INOTIFY_MAX_USER_INSTANCES=8192`
- `HEAD_DOCKER_READY_TIMEOUT_SEC=300`
- `AGENT_TIMEOUT_SEC=9000`
- `MINI_SWE_MODEL_TIMEOUT_SEC=1200`
- `HARBOR_AGENT_MAX_TURNS=20`
- `MAX_TRAIN_TASKS=64`
- `ROLLOUT_ENGINES=4`
- `ROLLOUT_TP_SIZE=2`

Also preserve:
- pre-pulled Docker image reuse
- shared mini-swe-agent tool home
- shared uv cache
- shared uv Python runtime mount

## 9. Required Execution Order

1. Build the Python environment from `pyproject.toml` and `uv.lock`
2. Apply Harbor runtime patches
3. Ensure the selected Docker runtime is available on the head node:
   - current benchmark default: system Docker (`DOCKER_MODE=rootful`)
   - rootless fallback only if the cluster cannot expose a usable system Docker socket:
     `examples/train_integrations/harbor/ops/install_rootless_docker_userland.sh`
   - compose plugin if the cluster image does not already provide it:
     `examples/train_integrations/harbor/ops/install_docker_compose_plugin.sh`
4. Prepare external assets:
   - model path
   - Harbor R2EGYM dataset paths
   - pre-pulled Docker images
5. Run readiness:
   - `examples/train_integrations/harbor/run_qwen3_32b_full_readiness_suite.sh`
6. Run validation:
   - `examples/train_integrations/harbor/run_full_run_validation_suite.sh`
7. Only then launch the replay wrapper:
   - `examples/train_integrations/harbor/run_r2egym_qwen3_32b_6node_rootless_full_1step_miniswe50_timeout9000_fd131072_blockdockerd_replay.sh`

Do not skip readiness and validation on a new cluster.

For the current cross-job benchmark wrapper, keep the canonical stage order
`cleanup-stage all -> prepare -> head -> ray -> rollout -> status -> driver`.
That `driver` action is the durable path: it starts a detached local `srun`
client internally and then waits on
`examples/train_integrations/harbor/ops/wait_harbor_driver_until_terminal.py`.
The current benchmark default is system Docker on the merged/head node
(`DOCKER_MODE=rootful`), not rootless Docker.
Do not use `driver-detach` for the benchmark path.

## 10. What Usually Goes Wrong

- only reading the `20260319` runbook and missing the `20260318` baseline
- forgetting to apply the Harbor runtime patches
- reusing old node names or old data/model paths
- launching before readiness/validation
- not mounting the shared mini-swe-agent / uv cache / uv Python paths
- treating the wrapper defaults as portable across clusters

## 11. Minimal Launch Template

Use this as a starting point and replace every cluster-specific value:

```bash
cd <repo-root>
export PATH="<repo-root>/.venv/bin:$HOME/.local/bin:$PATH"

python examples/train_integrations/harbor/ops/apply_harbor_runtime_patches.py --backup

JOB_ID=<slurm-job-id> \
HEAD_NODE=<head-node> \
ROLLOUT_NODE=<rollout-node> \
TRAINER_NODES_CSV=<trainer-a>,<trainer-b>,<trainer-c>,<trainer-d> \
TRAIN_DATA="['<r2egym-trivial>','<r2egym-easy>','<r2egym-medium>','<r2egym-hard>']" \
EVAL_DATA="$TRAIN_DATA" \
MODEL_PATH=<qwen3-32b-model-dir> \
DOCKER_MODE=rootful \
HARBOR_SHARED_UV_CACHE_HOST_DIR=<shared-uv-cache> \
HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME=<shared-mini-tool-home> \
HARBOR_SHARED_UV_PYTHON_HOST_DIR=<shared-uv-python-dir> \
bash examples/train_integrations/harbor/run_r2egym_qwen3_32b_6node_rootless_full_1step_miniswe50_timeout9000_fd131072_blockdockerd_replay.sh
```

## 12. Success Criteria

The reproduction target is:
- 1 training step
- fully async
- ThunderAgent enabled
- Harbor-enabled task execution
- R2EGYM datasets
- 6-node topology
- pre-pulled image reuse
- mini-swe-agent fast path active
