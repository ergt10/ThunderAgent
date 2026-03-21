# R2EGYM Qwen3-32B 6-Node Rootless Fully Async 1-Step 4-Server mini-swe-agent-50 timeout9000 fd131072 `blockdockerd` Replay Runbook

Date: 2026-03-18

This document is the replay-oriented runbook for reproducing the same experiment shape as:

- `r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051`

Unlike the historical runbook, this file is organized around one executable replay wrapper plus explicit preflight assertions.

## 1. Canonical Replay Entry Point

Use this wrapper as the canonical replay command:

- `/home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/run_r2egym_qwen3_32b_6node_rootless_full_1step_miniswe50_timeout9000_fd131072_blockdockerd_replay.sh`

That wrapper delegates to the orchestration script:

- `/home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/run_r2egym_qwen3_32b_6node_rootless_full_1step_miniswe50_timeout9000.sh`

The replay wrapper freezes these historical choices:

- dataset: local four-split R2EGYM Harbor tree
- topology: `003` head, `008` rollout, `012-015` trainers
- rollout: `4` servers, `TP=2`, ports `18000-18003`
- fd soft limits: `131072`
- Harbor agent timeout: `9000s`
- rootless Docker data root on head `/scratch/triton_cache/$USER/...`

## 2. Preconditions

Before launching, all of the following must be true.

### 2.1 Repo And Python Environment

```bash
cd /home/hkang/zthunder_agent/SkyRL
test -x /home/hkang/zthunder_agent/SkyRL/.venv/bin/python
test -x /home/hkang/zthunder_agent/SkyRL/.venv/bin/ray
test -f /home/hkang/zthunder_agent/SkyRL/skyrl/train/utils/templates/qwen3_acc_thinking.jinja2
test -x /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/run_r2egym_qwen3_32b_6node_rootless_full_1step_miniswe50_timeout9000_fd131072_blockdockerd_replay.sh
```

### 2.2 Local Harbor Dataset Tree

```bash
test -d /home/hkang/zthunder_agent/data/harbor/r2egym-trivial
test -d /home/hkang/zthunder_agent/data/harbor/r2egym-easy
test -d /home/hkang/zthunder_agent/data/harbor/r2egym-medium
test -d /home/hkang/zthunder_agent/data/harbor/r2egym-hard
```

If any split is missing, prepare it once:

```bash
cd /home/hkang/zthunder_agent/SkyRL

for split in trivial easy medium hard; do
  /home/hkang/zthunder_agent/SkyRL/.venv/bin/python \
    /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/prepare_harbor_dataset.py \
    --dataset "DCAgent/exp-rdb-r2egym-${split}" \
    --output_dir "/home/hkang/zthunder_agent/data/harbor/r2egym-${split}"
done
```

### 2.3 Slurm Allocation Shape

This replay expects the same six-node shape pinned by:

- `/home/hkang/zthunder_agent/skyrl.sbatch`

That batch file requests:

- nodes: `6`
- ntasks: `6`
- cpus-per-task: `100`
- `gpu:8` on each node
- nodelist: `research-dev-coder-[003,008,012-015]`

Submit a fresh allocation:

```bash
cd /home/hkang/zthunder_agent
sbatch --reservation=datagen /home/hkang/zthunder_agent/skyrl.sbatch
```

Export the new job id:

```bash
export JOB_ID=<new_job_id>
```

Verify the allocation really holds the expected nodes:

```bash
squeue -j "$JOB_ID" -o "%.18i %.10T %.40N"
scontrol show hostnames "$(squeue -j "$JOB_ID" -h -o '%N')"
```

### 2.4 Rootless Docker Prerequisites On The Head Node

The head node must have these available to user `hkang`:

- `docker`
- `docker compose`
- `slirp4netns`
- `rootlesskit`
- `newuidmap`
- `newgidmap`
- `dockerd-rootless.sh`

Check them on `003`:

```bash
export HEAD_NODE=research-dev-coder-003

srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 \
  -w "$HEAD_NODE" --ntasks=1 --nodes=1 --cpus-per-task=1 --gres=gpu:0 \
  bash -lc 'docker compose version && command -v slirp4netns rootlesskit newuidmap newgidmap dockerd-rootless.sh'
```

Check that the head node can write the intended Docker data-root parent:

```bash
srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 \
  -w "$HEAD_NODE" --ntasks=1 --nodes=1 --cpus-per-task=1 --gres=gpu:0 \
  bash -lc 'mkdir -p /scratch/triton_cache/$USER && test -w /scratch/triton_cache/$USER'
```

If rootless startup fails with missing subuid/subgid state, fix that first. This replay runbook assumes that prerequisite is already restored.

## 3. Fixed Replay Topology Variables

Export these once in the shell from which you will launch the replay wrapper:

```bash
export JOB_ID="${JOB_ID:?set JOB_ID first}"
export HEAD_NODE=research-dev-coder-003
export ROLLOUT_NODE=research-dev-coder-008
export TRAINER_NODES_CSV=research-dev-coder-012,research-dev-coder-013,research-dev-coder-014,research-dev-coder-015
```

You do not need to export `HEAD_IP` or `ROLLOUT_IP`; the orchestration script resolves them itself from inside the allocation.

## 4. Recommended Launch Command

Run the replay wrapper from a persistent shell:

```bash
cd /home/hkang/zthunder_agent/SkyRL
export PATH=/home/hkang/zthunder_agent/SkyRL/.venv/bin:$HOME/.local/bin:$PATH

JOB_ID="$JOB_ID" \
bash /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/run_r2egym_qwen3_32b_6node_rootless_full_1step_miniswe50_timeout9000_fd131072_blockdockerd_replay.sh
```

That single command will:

1. start rootless Docker on head `003`
2. start Ray head and four trainer workers
3. start four external rollout servers on `008`
4. wait for rollout `/health`
5. start trainer monitors
6. launch the train driver
7. run post-stop summary and analysis
8. clean up the child Slurm steps it created

## 5. Optional Frozen Overrides

If you want deterministic names and runtime paths instead of auto-generated timestamps, export these before launch:

```bash
export RUN_TS=20260318_000000
export RUN_SHORT_ID=r2e1srp-fixed
export RUN_NAME_OVERRIDE=r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-replay-20260318_000000
```

If you want the replay to reuse the pre-pulled R2EGYM image cache on `/scratch`, export:

```bash
export DOCKER_DATA_ROOT=/scratch/triton_cache/$USER/r2egym-rootless-image-cache
```

## 6. Expected Output Paths

The replay wrapper writes:

- logs:
  - `/home/hkang/zthunder_agent/tmp_logs/<RUN_NAME>`
- Harbor artifacts:
  - `/home/hkang/zthunder_agent/<RUN_NAME>`

Important files under the log root:

- `launcher_rootless_docker.log`
- `launcher_ray.log`
- `launcher_rollout.log`
- `launcher_trainer_monitors.log`
- `launcher_train_driver.log`
- `launcher_summary.log`
- `launcher_analysis.log`
- `thunderagent.log`
- `monitoring/trial_progress.tsv`

## 7. Live Monitoring Commands

Assume:

```bash
export RUN_NAME=<the actual run name printed by the wrapper>
export LOG_DIR=/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME
```

Watch top-level launcher output:

```bash
tail -f "$LOG_DIR/launcher_train_driver.log"
```

Watch trial accumulation:

```bash
tail -f "$LOG_DIR/monitoring/trial_progress.tsv"
```

Watch ThunderAgent request failures:

```bash
grep -n "failed after\\|ReadTimeout" "$LOG_DIR/thunderagent.log"
```

Watch Harbor Docker startup failures:

```bash
grep -n "EnvironmentStartTimeoutError" "$LOG_DIR/launcher_train_driver.log"
```

## 8. Stop Procedure

Preferred stop path:

- press `Ctrl-C` in the wrapper shell

Reason:

- the orchestration script traps `EXIT/INT/TERM`
- it kills its background launcher processes
- it cancels only the Slurm child steps that it created for this replay

If the wrapper shell is gone and you must stop manually, first inspect steps:

```bash
squeue -s -j "$JOB_ID" -o "%.18i %.30j %.12T %.40R"
```

Then cancel only the replay-created steps, not the batch allocation:

```bash
scancel <step_id_1> <step_id_2> ...
```

Do not `scancel "$JOB_ID"` unless you explicitly want to kill the whole six-node allocation.

## 9. Sanity Checks After Stop

Verify the batch allocation still exists but the replay steps are gone:

```bash
sacct -j "$JOB_ID" --format=JobID,JobName%40,State,ExitCode --parsable2
```

Review generated summaries:

```bash
ls -R "/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/post_run_summary"
ls -R "/home/hkang/zthunder_agent/tmp_logs/$RUN_NAME/analysis"
```

## 10. Replay Contract Boundaries

This runbook is designed to be executable on the current cluster, but it does not claim that the replay will be anomaly-free.

It does claim something narrower and more useful:

- the allocation command is explicit
- the topology is explicit
- the environment assertions are explicit
- the replay entry point is one fixed wrapper
- the wrapper expands into one fixed orchestration script already present in the repo

That is the canonical replay contract for this experiment.
