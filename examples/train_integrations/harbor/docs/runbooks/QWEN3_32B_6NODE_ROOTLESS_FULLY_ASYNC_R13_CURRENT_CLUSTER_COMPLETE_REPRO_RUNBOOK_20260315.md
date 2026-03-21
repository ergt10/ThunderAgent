# QWEN3-32B 6-Node Rootless Fully Async R13 Current-Cluster Complete Repro Runbook

Date: 2026-03-15

This is the exact end-to-end runbook for reproducing the current-cluster R13 rootless Harbor full-async run on:

- head: `research-dev-coder-003`
- rollout: `research-dev-coder-008`
- trainers: `research-dev-coder-012,013,014,015`

It follows the historical Harbor workflow and only keeps the deviations that were actually required on the current cluster:

- use `003/008/012-015` instead of the older secure-cluster nodes
- keep artifacts under `/home/hkang/zthunder_agent/harbor_runs`
- temporarily raise kernel key quota on `003` so rootless Docker can survive Harbor container churn

## 0. Scope

This runbook covers:

1. environment bootstrap
2. model and dataset preparation
3. 6-node Slurm allocation
4. rootless Docker bring-up on the head node
5. rootless Harbor concurrency smoke
6. full readiness / preflight
7. exact R13 full launch
8. monitoring
9. stop / cleanup

All commands below assume:

- repo root: `/home/hkang/zthunder_agent/SkyRL`
- workspace root: `/home/hkang/zthunder_agent`
- current shell host can run `sbatch`, `srun`, `squeue`, `scontrol`

## 1. Historical Constraints

Treat these notes as the execution contract:

- [QWEN3_32B_6NODE_ROOTLESS_FULLY_ASYNC_R12_20STEP_REPRO.md](/home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/QWEN3_32B_6NODE_ROOTLESS_FULLY_ASYNC_R12_20STEP_REPRO.md)
- [QWEN3_32B_6NODE_ROOTLESS_FULLY_ASYNC_R13_FAILURE_NOTES.md](/home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/QWEN3_32B_6NODE_ROOTLESS_FULLY_ASYNC_R13_FAILURE_NOTES.md)

Current-cluster failure notes from the latest R13 attempt are here:

- [QWEN3_32B_6NODE_ROOTLESS_FULLY_ASYNC_R13_FULL_RUN_STOPPED_BEFORE_STEP5_20260315.md](/home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/QWEN3_32B_6NODE_ROOTLESS_FULLY_ASYNC_R13_FULL_RUN_STOPPED_BEFORE_STEP5_20260315.md)

## 2. One-Time Environment Bootstrap

Run from the repo root:

```bash
cd /home/hkang/zthunder_agent/SkyRL

bash examples/train_integrations/harbor/bootstrap_uv_env_from_freeze.sh
```

What this does:

- rebuilds `.venv` from `requirements.txt`
- sanitizes the freeze for `uv`
- reapplies the Harbor rootless Docker upload patch

Sanity check:

```bash
.venv/bin/python examples/train_integrations/harbor/apply_harbor_rootless_patch.py --check
```

Expected result:

- `PATCH_PRESENT .../harbor/environments/docker/docker.py`

## 3. Model and Dataset Preparation

### 3.1 Model

The training launcher expects:

- model path: `/data/zy/models/$USER/models/Qwen3-32B`

Download it if missing:

```bash
cd /home/hkang/zthunder_agent/SkyRL

.venv/bin/hf download Qwen/Qwen3-32B \
  --repo-type model \
  --local-dir /data/zy/models/$USER/models/Qwen3-32B \
  --cache-dir /data/zy/models/hub
```

### 3.2 Harbor datasets

The current launcher expects:

- train data: `/home/hkang/zthunder_agent/data/harbor/CodeContests`
- eval data: `/home/hkang/zthunder_agent/data/harbor/OpenThoughts-TB-dev`

Prepare them:

```bash
cd /home/hkang/zthunder_agent/SkyRL

.venv/bin/python examples/train_integrations/harbor/prepare_harbor_dataset.py \
  --dataset open-thoughts/CodeContests \
  --output_dir /home/hkang/zthunder_agent/data/harbor/CodeContests

.venv/bin/python examples/train_integrations/harbor/prepare_harbor_dataset.py \
  --dataset open-thoughts/OpenThoughts-TB-dev \
  --output_dir /home/hkang/zthunder_agent/data/harbor/OpenThoughts-TB-dev
```

Sanity check:

```bash
test -f /data/zy/models/$USER/models/Qwen3-32B/config.json
test -d /home/hkang/zthunder_agent/data/harbor/CodeContests
test -d /home/hkang/zthunder_agent/data/harbor/OpenThoughts-TB-dev
test -f /home/hkang/zthunder_agent/SkyRL/skyrl/train/utils/templates/qwen3_acc_thinking.jinja2
```

## 4. Reserve the Exact 6-Node Shape

The current allocation file is:

- [skyrl.sbatch](/home/hkang/zthunder_agent/skyrl.sbatch)

It already pins the current working topology:

- `research-dev-coder-[003,008,012-015]`
- `--reservation=datagen`

Submit it:

```bash
cd /home/hkang/zthunder_agent
sbatch skyrl.sbatch
```

Confirm the allocation:

```bash
squeue -u "$USER" -o "%.18i %.10T %.40N"
```

Export the run topology variables once the job is live:

```bash
export JOB_ID=<your_job_id>
export REPO=/home/hkang/zthunder_agent/SkyRL
export HEAD_NODE=research-dev-coder-003
export ROLLOUT_NODE=research-dev-coder-008
export TRAINER_NODES_CSV=research-dev-coder-012,research-dev-coder-013,research-dev-coder-014,research-dev-coder-015
export XDG_RUNTIME_DIR=/tmp/xdg-rootless-r13-full-$JOB_ID
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
```

Resolve the head and rollout IPs from inside the allocation:

```bash
export HEAD_IP="$(srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 -w "$HEAD_NODE" --ntasks=1 --nodes=1 bash -lc \"hostname -I | awk '{print \\\$1}'\")"
export ROLLOUT_IP="$(srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 -w "$ROLLOUT_NODE" --ntasks=1 --nodes=1 bash -lc \"hostname -I | awk '{print \\\$1}'\")"

printf 'HEAD_IP=%s\nROLLOUT_IP=%s\n' "$HEAD_IP" "$ROLLOUT_IP"
```

## 5. Rootless Docker Prerequisites on the Head Node

### 5.1 User-space requirements

On `003`, these must exist:

- `docker`
- `docker compose`
- `slirp4netns`
- `rootlesskit`
- `newuidmap`
- `newgidmap`
- `dockerd-rootless.sh`

Check them:

```bash
srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 -w "$HEAD_NODE" --ntasks=1 --nodes=1 \
  bash -lc 'command -v docker docker-compose >/dev/null 2>&1 || true; docker compose version; command -v slirp4netns rootlesskit newuidmap newgidmap dockerd-rootless.sh'
```

### 5.2 Temporary key-quota expansion on `003`

This was required on the current cluster. Without it, rootless Docker hit:

- `unable to join session keyring`
- `unable to create session key: disk quota exceeded`

Apply this on the head node:

```bash
srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 -w "$HEAD_NODE" --ntasks=1 --nodes=1 \
  bash -lc 'sudo sysctl -w kernel.keys.maxkeys=20000 kernel.keys.maxbytes=25000000'
```

Verify:

```bash
srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 -w "$HEAD_NODE" --ntasks=1 --nodes=1 \
  bash -lc 'sysctl kernel.keys.maxkeys kernel.keys.maxbytes'
```

Expected:

```text
kernel.keys.maxkeys = 20000
kernel.keys.maxbytes = 25000000
```

## 6. Start Rootless Docker on the Head Node

Keep the daemon in a dedicated blocking `srun` step:

```bash
cd "$REPO"

srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 \
  -w "$HEAD_NODE" --ntasks=1 --nodes=1 --cpus-per-task=8 --gres=gpu:0 \
  bash -lc "
    cd '$REPO' &&
    export XDG_RUNTIME_DIR='$XDG_RUNTIME_DIR' &&
    export DOCKER_HOST='$DOCKER_HOST' &&
    export ROOTLESS_DOCKER_START_MODE=block &&
    export SCRATCH_ROOT=/tmp/$USER/docker-rootless-$JOB_ID &&
    export DOCKER_DEFAULT_ADDRESS_POOL_BASE=10.240.0.0/12 &&
    export DOCKER_DEFAULT_ADDRESS_POOL_SIZE=24 &&
    bash examples/train_integrations/harbor/start_rootless_docker_for_harbor.sh
  " &

export ROOTLESS_DOCKER_STEP_PID=$!
```

Quick health check:

```bash
srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 -w "$HEAD_NODE" --ntasks=1 --nodes=1 \
  bash -lc "export DOCKER_HOST='$DOCKER_HOST'; docker info >/dev/null && docker compose version"
```

## 7. Rootless Harbor Concurrency Smoke

Do this before the full readiness sweep. This isolates:

- `Harbor trial -> Docker sandbox -> verifier`

It does not involve:

- rollout engines
- trainer step loop

Run the same high-concurrency smoke that matched the R13 Harbor pressure:

```bash
cd "$REPO"

JOB_ID="$JOB_ID" \
HEAD_NODE="$HEAD_NODE" \
DOCKER_MODE=rootless \
DOCKER_HOST="$DOCKER_HOST" \
TRIAL_COUNT=256 \
MAX_CONCURRENCY=256 \
MAX_FAILURES=0 \
OUTPUT_ROOT=/home/hkang/zthunder_agent/tmp_logs/harbor-rootless-concurrency256-job${JOB_ID} \
bash examples/train_integrations/harbor/run_harbor_docker_concurrency_smoke.sh
```

Expected success artifact:

- `summary.json` under the chosen `OUTPUT_ROOT`

On the current cluster, this step passed only after the key quota increase.

## 8. Full Rootless Readiness / Preflight

This step starts Ray and rollout, verifies cluster placement, and checks Harbor, Docker, storage, model, datasets, and rollout health.

Run it exactly like this:

```bash
cd "$REPO"

JOB_ID="$JOB_ID" \
HEAD_NODE="$HEAD_NODE" \
TRAINER_NODES_CSV="$TRAINER_NODES_CSV" \
ROLLOUT_NODE="$ROLLOUT_NODE" \
DOCKER_MODE=rootless \
XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" \
DOCKER_HOST="$DOCKER_HOST" \
START_ROOTLESS_DOCKER=false \
RUN_HARBOR_ROOTLESS_SMOKE=true \
RUN_HARBOR_DOCKER_CONCURRENCY_SMOKE=false \
START_RAY_CLUSTER=true \
RUN_CLUSTER_VALIDATION=true \
START_ROLLOUT_SERVERS=true \
RUN_TRAINING_SMOKE=false \
bash examples/train_integrations/harbor/run_qwen3_32b_full_readiness_suite.sh
```

Readiness passes only if all of these are true:

- `003/008/012-015` can import `ray`, `harbor`, `ThunderAgent`, `torch`, `vllm`
- every checked GPU node reports `torch.cuda.device_count() == 8`
- rootless Docker is live on `003`
- Harbor verifier smoke passes
- Ray head exposes `{"harbor_head": 1}`
- a real `STRICT_SPREAD` placement group with four `8 CPU + 8 GPU` bundles becomes ready
- rollout node brings up healthy vLLM servers on `18000` and `18001`

## 9. Exact Full R13 Launch

Important:

- `run_qwen3_32b_full_readiness_suite.sh` is a validation pass.
- when it exits, it tears down the Ray and rollout steps that it started for readiness.
- before the real full launch, bring those two services back up as dedicated long-lived blocking steps and keep them alive while the training driver runs.

Bring Ray back up:

```bash
cd "$REPO"

JOB_ID="$JOB_ID" \
HEAD_NODE="$HEAD_NODE" \
TRAINER_NODES_CSV="$TRAINER_NODES_CSV" \
RAY_START_MODE=block \
RAY_PORT=6381 \
bash examples/train_integrations/harbor/launch_qwen3_32b_ray_cluster.sh
```

Bring rollout back up under the same `RUN_NAME` that the full launch will use:

```bash
cd "$REPO"

export RUN_NAME=codecontest-qwen3-32b-6node-rootless-full-r13-64x4x64-stale2-conc256-$(date +%Y%m%d_%H%M%S)
export LOG_ROOT=/home/hkang/zthunder_agent/tmp_logs

srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 \
  -w "$ROLLOUT_NODE" --ntasks=1 --nodes=1 --cpus-per-task=100 --gres=gpu:8 \
  bash -lc "
    cd '$REPO' &&
    export RAY_HEAD_IP='$HEAD_IP' &&
    export RUN_NAME='$RUN_NAME' &&
    export LOG_DIR='$LOG_ROOT/$RUN_NAME/rollout' &&
    export MONITORING_DIR='$LOG_ROOT/$RUN_NAME/rollout/monitoring' &&
    export TENSORBOARD_DIR='$LOG_ROOT/$RUN_NAME/rollout/tensorboard' &&
    export PORT_A=18000 &&
    export PORT_B=18001 &&
    bash examples/train_integrations/harbor/start_qwen3_32b_external_rollout_servers.sh
  "
```

This is the exact full-run shape used on the current cluster:

- `train_batch_size=64`
- `policy_mini_batch_size=64`
- `micro_forward_batch_size_per_gpu=4`
- `micro_train_batch_size_per_gpu=4`
- `n_samples_per_prompt=4`
- `num_parallel_generation_workers=64`
- `max_concurrency=256`
- `trajectories_per_second=2`
- `max_staleness_steps=2`
- `flash_attn=true`
- `max_seq_len=6144`

Use a persistent stdout log for the training driver. This matters because step-level progress is otherwise only visible in the live terminal stream.

```bash
cd "$REPO"

export RUN_NAME=codecontest-qwen3-32b-6node-rootless-full-r13-64x4x64-stale2-conc256-$(date +%Y%m%d_%H%M%S)
export LOG_ROOT=/home/hkang/zthunder_agent/tmp_logs
export RUN_ARTIFACT_ROOT=/home/hkang/zthunder_agent/harbor_runs
mkdir -p "$LOG_ROOT/$RUN_NAME"

srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 \
  -w "$HEAD_NODE" --ntasks=1 --nodes=1 --cpus-per-task=8 --gres=gpu:0 \
  bash -lc "
    cd '$REPO' &&
    export XDG_RUNTIME_DIR='$XDG_RUNTIME_DIR' &&
    export DOCKER_HOST='$DOCKER_HOST' &&
    export DOCKER_MODE=rootless &&
    export RAY_HEAD_IP='$HEAD_IP' &&
    export ROLLOUT_HOST_IP='$ROLLOUT_IP' &&
    export RUN_NAME_OVERRIDE='$RUN_NAME' &&
    export LOG_ROOT='$LOG_ROOT' &&
    export RUN_ARTIFACT_ROOT='$RUN_ARTIFACT_ROOT' &&
    export FULL_TRAIN_BATCH_SIZE=64 &&
    export FULL_POLICY_MINI_BATCH_SIZE=64 &&
    export FULL_MICRO_FORWARD_BATCH_SIZE_PER_GPU=4 &&
    export FULL_MICRO_TRAIN_BATCH_SIZE_PER_GPU=4 &&
    export FULL_N_SAMPLES=4 &&
    export FULL_NUM_PARALLEL_GENERATION_WORKERS=64 &&
    export FULL_MAX_CONCURRENCY=256 &&
    export FULL_TRAJ_PER_SEC=2 &&
    export FULL_MAX_STALENESS_STEPS=2 &&
    export FLASH_ATTN=true &&
    export TRAIN_MAX_SEQ_LEN=6144 &&
    bash examples/train_integrations/harbor/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh full
  " 2>&1 | tee "$LOG_ROOT/$RUN_NAME/launcher-interactive.log"
```

Important output roots for this launch:

- run log root: `/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME`
- artifacts: `/home/hkang/zthunder_agent/harbor_runs/$RUN_NAME`
- trainer monitors: `/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/trainer_monitors`
- rollout logs: `/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/rollout`
- thunderagent log: `/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/thunderagent.log`

## 10. What the Launch Script Starts Automatically

The `full` stage of [run_codecontest_qwen3_32b_6node_rootless_fully_async.sh](/home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh) does all of this automatically:

- verifies `RAY_ADDRESS` and rollout URLs
- reruns the Python preflight before training
- starts the head-local stage-3 monitor
- starts the trainer node monitors
- writes artifacts into:
  - `trials_run`
  - `ckpts`
  - `exports`
- uses:
  - `trainer.ckpt_interval=5`
  - `trainer.hf_save_interval=5`

That means you do not need to manually start trainer monitors during the full launch.

## 11. Live Monitoring

### 11.1 Fast health checks

Artifact growth:

```bash
python3 - <<'PY'
from pathlib import Path
run = Path('/home/hkang/zthunder_agent/harbor_runs') / '<RUN_NAME>'
trials = run / 'trials_run'
print('trajectory_json=%d' % sum(1 for _ in trials.rglob('trajectory.json')))
print('exception_txt=%d' % sum(1 for _ in trials.rglob('exception.txt')))
print('result_json=%d' % sum(1 for _ in trials.rglob('result.json')))
print('ckpt_dirs=%d' % (sum(1 for _ in (run / 'ckpts').glob('global_step_*')) if (run / 'ckpts').exists() else 0))
PY
```

ThunderAgent activity:

```bash
tail -f /home/hkang/zthunder_agent/tmp_logs/<RUN_NAME>/thunderagent.log
```

Trainer GPU memory:

```bash
for f in /home/hkang/zthunder_agent/tmp_logs/<RUN_NAME>/trainer_monitors/research-dev-coder-0{12,13,14,15}/gpu_summary.tsv; do
  echo "=== $f ==="
  tail -n 8 "$f"
done
```

Rollout GPU memory:

```bash
tail -n 8 /home/hkang/zthunder_agent/tmp_logs/<RUN_NAME>/rollout/monitoring/gpu_summary.tsv
```

### 11.2 Rollout telemetry check

After rollout startup and monitor bring-up, these files should become non-empty:

- `monitoring/vllm_kv_cache.log`
- `monitoring/vllm_metrics.tsv`
- `rollout/monitoring/vllm_kv_cache.log`
- `rollout/monitoring/vllm_metrics.tsv`

Quick check:

```bash
for f in \
  /home/hkang/zthunder_agent/tmp_logs/<RUN_NAME>/monitoring/vllm_kv_cache.log \
  /home/hkang/zthunder_agent/tmp_logs/<RUN_NAME>/monitoring/vllm_metrics.tsv \
  /home/hkang/zthunder_agent/tmp_logs/<RUN_NAME>/rollout/monitoring/vllm_kv_cache.log \
  /home/hkang/zthunder_agent/tmp_logs/<RUN_NAME>/rollout/monitoring/vllm_metrics.tsv; do
  echo "=== $f ==="
  wc -c "$f"
  tail -n 5 "$f"
done
```

## 12. Stop the Run but Keep the Allocation

If you want to stop the training stack without cancelling the Slurm job itself, cancel only the non-batch steps:

```bash
squeue -s -j "$JOB_ID" -o "%.18i" | awk 'NR>1 && $1 !~ /\\.batch$/ {print $1}' | xargs -r scancel
```

After a few seconds, verify that only `batch` remains:

```bash
squeue -s -j "$JOB_ID" -o "%.18i %.9P %.10j %.12N"
```

Expected final state:

- only `1139.batch`-like batch step remains

## 13. Cleanup Verification

Verify no residual training / rollout / Ray / rootless Docker processes are still using GPUs:

```bash
python3 - <<'PY'
import subprocess
nodes=['research-dev-coder-003','research-dev-coder-008','research-dev-coder-012','research-dev-coder-013','research-dev-coder-014','research-dev-coder-015']
job_id='<JOB_ID>'
for n in nodes:
    print('===', n, '===')
    cmd=['srun','--jobid',job_id,'--overlap','--overcommit','--immediate=10','-w',n,'--ntasks=1','--nodes=1','bash','-lc',
         "hostname; echo '[compute-apps]'; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true; "
         "echo '[ports]'; ss -ltnp | grep -E ':(6381|8265|8080|18000|18001)\\b' || true; "
         "echo '[procs]'; ps -ef | grep -E 'raylet|gcs_server|vllm_server.py|run_codecontest_qwen3_32b_6node_rootless_fully_async|dockerd-rootless|rootlesskit|start_trainer_node_monitors|ThunderAgent' | grep -v grep || true"]
    r=subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
PY
```

## 14. Optional: Restore the Default Key Quota

Only do this after you are done with rootless Harbor on `003`:

```bash
srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 -w "$HEAD_NODE" --ntasks=1 --nodes=1 \
  bash -lc 'sudo sysctl -w kernel.keys.maxkeys=200 kernel.keys.maxbytes=20000'
```

Do not restore these defaults before the run if you intend to use rootless Docker with Harbor concurrency on the current cluster.

## 15. Known Current-Cluster Failure Signatures

If the run regresses, these are the first patterns to look for:

### Rootless Docker key quota failure

Symptoms:

- `unable to join session keyring`
- `unable to create session key: disk quota exceeded`

First fix:

- reapply the temporary key-quota increase on `003`

### Ray placement failure before step 0

Symptoms:

- placement group timeout
- Ray says no node can satisfy `{'CPU': 8.0, 'GPU': 8.0}`

First fix:

- rerun full readiness and confirm the `STRICT_SPREAD` probe passes

### Harbor setup bug

Symptoms:

- `harbor/agents/terminus_2/tmux_session.py:372`
- `AttributeError: 'NoneType' object has no attribute 'strip'`

Interpretation:

- Harbor error-path bug in `tmux` startup/setup

### Live-but-stalled run

Symptoms:

- `thunderagent.log` stops advancing
- `trajectory/result/exception` counts stop moving
- trainer and rollout GPUs keep holding memory
- sampled `util_gpu_pct` stays at `0`

Interpretation:

- no-forward-progress stall, often around generation-buffer flow rather than hard crash

## 16. Minimal Command Checklist

If you only want the exact command sequence without explanation, this is the shortest safe checklist:

```bash
cd /home/hkang/zthunder_agent/SkyRL
bash examples/train_integrations/harbor/bootstrap_uv_env_from_freeze.sh

.venv/bin/hf download Qwen/Qwen3-32B --repo-type model --local-dir /data/zy/models/$USER/models/Qwen3-32B --cache-dir /data/zy/models/hub
.venv/bin/python examples/train_integrations/harbor/prepare_harbor_dataset.py --dataset open-thoughts/CodeContests --output_dir /home/hkang/zthunder_agent/data/harbor/CodeContests
.venv/bin/python examples/train_integrations/harbor/prepare_harbor_dataset.py --dataset open-thoughts/OpenThoughts-TB-dev --output_dir /home/hkang/zthunder_agent/data/harbor/OpenThoughts-TB-dev

cd /home/hkang/zthunder_agent
sbatch skyrl.sbatch

export JOB_ID=<job_id>
export REPO=/home/hkang/zthunder_agent/SkyRL
export HEAD_NODE=research-dev-coder-003
export ROLLOUT_NODE=research-dev-coder-008
export TRAINER_NODES_CSV=research-dev-coder-012,research-dev-coder-013,research-dev-coder-014,research-dev-coder-015
export XDG_RUNTIME_DIR=/tmp/xdg-rootless-r13-full-$JOB_ID
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
export HEAD_IP="$(srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 -w "$HEAD_NODE" --ntasks=1 --nodes=1 bash -lc \"hostname -I | awk '{print \\\$1}'\")"
export ROLLOUT_IP="$(srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 -w "$ROLLOUT_NODE" --ntasks=1 --nodes=1 bash -lc \"hostname -I | awk '{print \\\$1}'\")"

srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 -w "$HEAD_NODE" --ntasks=1 --nodes=1 bash -lc 'sudo sysctl -w kernel.keys.maxkeys=20000 kernel.keys.maxbytes=25000000'

srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 -w "$HEAD_NODE" --ntasks=1 --nodes=1 --cpus-per-task=8 --gres=gpu:0 \
  bash -lc "cd '$REPO' && export XDG_RUNTIME_DIR='$XDG_RUNTIME_DIR' && export DOCKER_HOST='$DOCKER_HOST' && export ROOTLESS_DOCKER_START_MODE=block && export SCRATCH_ROOT=/tmp/$USER/docker-rootless-$JOB_ID && export DOCKER_DEFAULT_ADDRESS_POOL_BASE=10.240.0.0/12 && export DOCKER_DEFAULT_ADDRESS_POOL_SIZE=24 && bash examples/train_integrations/harbor/start_rootless_docker_for_harbor.sh" &

JOB_ID="$JOB_ID" HEAD_NODE="$HEAD_NODE" DOCKER_MODE=rootless DOCKER_HOST="$DOCKER_HOST" TRIAL_COUNT=256 MAX_CONCURRENCY=256 MAX_FAILURES=0 OUTPUT_ROOT=/home/hkang/zthunder_agent/tmp_logs/harbor-rootless-concurrency256-job${JOB_ID} bash "$REPO/examples/train_integrations/harbor/run_harbor_docker_concurrency_smoke.sh"

JOB_ID="$JOB_ID" HEAD_NODE="$HEAD_NODE" TRAINER_NODES_CSV="$TRAINER_NODES_CSV" ROLLOUT_NODE="$ROLLOUT_NODE" DOCKER_MODE=rootless XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" DOCKER_HOST="$DOCKER_HOST" START_ROOTLESS_DOCKER=false RUN_HARBOR_ROOTLESS_SMOKE=true START_RAY_CLUSTER=true RUN_CLUSTER_VALIDATION=true START_ROLLOUT_SERVERS=true RUN_TRAINING_SMOKE=false bash "$REPO/examples/train_integrations/harbor/run_qwen3_32b_full_readiness_suite.sh"

export RUN_NAME=codecontest-qwen3-32b-6node-rootless-full-r13-64x4x64-stale2-conc256-$(date +%Y%m%d_%H%M%S)
export LOG_ROOT=/home/hkang/zthunder_agent/tmp_logs
export RUN_ARTIFACT_ROOT=/home/hkang/zthunder_agent/harbor_runs
mkdir -p "$LOG_ROOT/$RUN_NAME"

srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 -w "$HEAD_NODE" --ntasks=1 --nodes=1 --cpus-per-task=8 --gres=gpu:0 \
  bash -lc "cd '$REPO' && export XDG_RUNTIME_DIR='$XDG_RUNTIME_DIR' && export DOCKER_HOST='$DOCKER_HOST' && export DOCKER_MODE=rootless && export RAY_HEAD_IP='$HEAD_IP' && export ROLLOUT_HOST_IP='$ROLLOUT_IP' && export RUN_NAME_OVERRIDE='$RUN_NAME' && export LOG_ROOT='$LOG_ROOT' && export RUN_ARTIFACT_ROOT='$RUN_ARTIFACT_ROOT' && export FULL_TRAIN_BATCH_SIZE=64 && export FULL_POLICY_MINI_BATCH_SIZE=64 && export FULL_MICRO_FORWARD_BATCH_SIZE_PER_GPU=4 && export FULL_MICRO_TRAIN_BATCH_SIZE_PER_GPU=4 && export FULL_N_SAMPLES=4 && export FULL_NUM_PARALLEL_GENERATION_WORKERS=64 && export FULL_MAX_CONCURRENCY=256 && export FULL_TRAJ_PER_SEC=2 && export FULL_MAX_STALENESS_STEPS=2 && export FLASH_ATTN=true && export TRAIN_MAX_SEQ_LEN=6144 && bash examples/train_integrations/harbor/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh full" \
  2>&1 | tee "$LOG_ROOT/$RUN_NAME/launcher-interactive.log"
```
