# Qwen3-32B 6-Node Rootless Harbor Full-Async `r12` 20-Step Repro

Date: 2026-03-12

## Scope

This note summarizes the previous successful 20-step full run and gives the exact reproduction procedure using the current repo state.

The run being summarized is:

- `codecontest-qwen3-32b-6node-rootless-full-20step-flash-attn-memtrace-r12`

Direct evidence used in this note:

- config: `/data/zy/models/hkang/harbor_runs/codecontest-qwen3-32b-6node-rootless-full-20step-flash-attn-memtrace-r12/ckpts/global_step_15/trainer_state.pt`
- step summary: `/home/hkang/zthunder_yagent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-20step-flash-attn-memtrace-r12/analysis/step_peak_summary.csv`
- latest checkpoint pointer: `/data/zy/models/hkang/harbor_runs/codecontest-qwen3-32b-6node-rootless-full-20step-flash-attn-memtrace-r12/ckpts/latest_ckpt_global_step.txt`

## What `r12` Actually Achieved

- It finished `20` training steps.
- It did not OOM.
- It used `flash_attn=true`.
- It preserved checkpoints at `global_step_5`, `global_step_10`, and `global_step_15`.
- It was stopped after step `20`, so there is no `global_step_20` checkpoint.

Direct evidence:

- `step_peak_summary.csv` has `20` rows.
- `latest_ckpt_global_step.txt` contains `15`.

## Exact `r12` Config

These values were read directly from `trainer_state.pt`, not inferred from memory.

### Topology

- Harbor head / rootless Docker / Ray head: `research-secure-14`
- Trainer nodes:
  - `research-secure-23`
  - `research-secure-06`
  - `research-secure-18`
  - `research-secure-11`
- External rollout node: `research-secure-17`

### Training

- `trainer.strategy=fsdp2`
- `trainer.flash_attn=true`
- `trainer.train_batch_size=32`
- `trainer.policy_mini_batch_size=32`
- `trainer.micro_forward_batch_size_per_gpu=1`
- `trainer.micro_train_batch_size_per_gpu=1`
- `trainer.algorithm.max_seq_len=6144`
- `trainer.fully_async.max_staleness_steps=1`
- `trainer.fully_async.num_parallel_generation_workers=32`
- `trainer.placement.colocate_all=false`
- `trainer.placement.colocate_policy_ref=true`
- `trainer.placement.policy_num_nodes=4`
- `trainer.placement.policy_num_gpus_per_node=8`
- `trainer.placement.ref_num_nodes=4`
- `trainer.placement.ref_num_gpus_per_node=8`
- `trainer.ckpt_interval=5`
- `trainer.hf_save_interval=5`
- `trainer.eval_before_train=false`
- `trainer.eval_interval=50`

### Rollout

- `generator.n_samples_per_prompt=2`
- `generator.eval_n_samples_per_prompt=1`
- `generator.rate_limit.trajectories_per_second=1`
- `generator.rate_limit.max_concurrency=4`
- `generator.inference_engine.run_engines_locally=false`
- `generator.inference_engine.external_server_urls=['http://172.27.21.28:18000','http://172.27.21.28:18001']`
- `generator.inference_engine.tensor_parallel_size=4`
- `generator.inference_engine.enforce_eager=true`
- `generator.inference_engine.gpu_memory_utilization=0.8`
- `generator.inference_engine.engine_init_kwargs.max_model_len=32768`
- `generator.inference_engine.thunder_agent_mode=tr`

### Other Important Values

- model path: `/data/zy/models/hkang/models/Qwen3-32B`
- served model name: `Qwen3-32B`
- `generator.sampling_params.logprobs=1`
- `generator.sampling_params.temperature=0.3`
- `generator.eval_sampling_params.logprobs=1`
- `generator.eval_sampling_params.temperature=0.0`
- `trainer.algorithm.off_policy_correction.tis_ratio_type=token`
- `trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=2.0`
- `max_train_tasks=640`
- `max_eval_tasks=20`

## Required Preconditions

### 1. Harbor rootless Docker patch must exist

This run was not stock Harbor.

The local virtualenv must contain the rootless verifier upload patch in:

- `/home/hkang/zthunder_yagent/SkyRL/.venv/lib/python3.13/site-packages/harbor/environments/docker/docker.py`

Expected behavior:

- `upload_file()` and `upload_dir()` stream tar into the container
- ownership is normalized with:
  - `--owner=0`
  - `--group=0`
  - `--numeric-owner`

Without this patch, the run regresses to the old rootless verifier failure and does not meaningfully train.

### 2. Model and data paths

Expected paths:

- model: `/data/zy/models/hkang/models/Qwen3-32B`
- train data: `/home/hkang/zthunder_yagent/data/harbor/CodeContests`
- eval data: `/home/hkang/zthunder_yagent/data/harbor/OpenThoughts-TB-dev`

### 3. Ray head must expose `harbor_head`

The entrypoint is head-pinned and requires Ray resource:

- `{"harbor_head": 1}`

Current entrypoint:

- [main_harbor_thunder_agent_fully_async_head_pinned.py](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/entrypoints/main_harbor_thunder_agent_fully_async_head_pinned.py#L19)

If this resource is missing, the driver stays pending and the run never starts.

## Exact Reproduction Procedure

The current launcher defaults no longer match `r12`.

To reproduce `r12` exactly with the current code, keep the current scripts but override the current defaults back to the `r12` values below.

### 1. Reserve the same 6-node shape

Use the same role split:

- `1` CPU node for Harbor + rootless Docker + Ray head
- `4` GPU nodes for trainer
- `1` GPU node for rollout

The original successful nodes were:

- head: `research-secure-14`
- trainer: `research-secure-23`, `research-secure-06`, `research-secure-18`, `research-secure-11`
- rollout: `research-secure-17`

### 2. Start rootless Docker on the head node

Run on the Harbor head node:

```bash
export SCRATCH=/scratch/$USER/scratch
export DOCKER_DATA_ROOT=/scratch/$USER/docker-rootless
mkdir -p "$SCRATCH" "$DOCKER_DATA_ROOT"

export XDG_RUNTIME_DIR=/tmp/xdg-test-$USER
mkdir -p "$XDG_RUNTIME_DIR"
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock

dockerd-rootless.sh \
  --data-root "$DOCKER_DATA_ROOT" \
  --exec-root "$XDG_RUNTIME_DIR/docker-exec" \
  --pidfile "$XDG_RUNTIME_DIR/docker.pid" \
  --host "$DOCKER_HOST" \
  --exec-opt native.cgroupdriver=cgroupfs \
  > "$SCRATCH/dockerd_r12_repro.log" 2>&1 &

docker info
```

### 3. Start the Ray head on the same head node

Use the head node's real IP as `HEAD_IP`.

```bash
export HEAD_IP="$(hostname -I | awk '{print $1}')"

/home/hkang/zthunder_yagent/SkyRL/.venv/bin/ray stop -f || true
/home/hkang/zthunder_yagent/SkyRL/.venv/bin/ray start \
  --head \
  --port=6381 \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --num-gpus=0 \
  --node-ip-address="$HEAD_IP" \
  --resources='{"harbor_head": 1}'
```

### 4. Start the 4 trainer Ray workers

Run the same command on each trainer node.

```bash
export HEAD_IP="<head-node-ip>"
export NODE_IP="$(hostname -I | awk '{print $1}')"

/home/hkang/zthunder_yagent/SkyRL/.venv/bin/ray stop -f || true
/home/hkang/zthunder_yagent/SkyRL/.venv/bin/ray start \
  --address="${HEAD_IP}:6381" \
  --node-ip-address="$NODE_IP" \
  --num-cpus=176 \
  --num-gpus=8
```

Why explicit `--num-cpus=176 --num-gpus=8` is important:

- it avoids the bad registration case where Ray sees the node but cannot place the 8-GPU trainer bundles

### 5. Start the rollout servers on the rollout node

Run on the rollout node:

```bash
cd /home/hkang/zthunder_yagent/SkyRL

export RUN_NAME=codecontest-qwen3-32b-6node-rootless-full-20step-flash-attn-memtrace-r12-repro
export LOG_DIR=/home/hkang/zthunder_yagent/tmp_logs/$RUN_NAME/rollout
export MONITORING_DIR=$LOG_DIR/monitoring
export TENSORBOARD_DIR=$LOG_DIR/tensorboard
export MODEL_PATH=/data/zy/models/hkang/models/Qwen3-32B
export TP_SIZE=4
export PORT_A=18000
export PORT_B=18001
export MAX_MODEL_LEN=32768
export GPU_MEMORY_UTILIZATION=0.8
export NCCL_SOCKET_IFNAME=ens7
export GLOO_SOCKET_IFNAME=ens7

bash examples/train_integrations/harbor/start_qwen3_32b_external_rollout_servers.sh
```

Wait until both are healthy:

```bash
curl -sf http://127.0.0.1:18000/health
curl -sf http://127.0.0.1:18001/health
```

### 6. Launch the exact `r12` training config on the head node

Run on the Harbor head node:

```bash
cd /home/hkang/zthunder_yagent/SkyRL

export RUN_NAME_OVERRIDE=codecontest-qwen3-32b-6node-rootless-full-20step-flash-attn-memtrace-r12-repro
export LOG_ROOT=/home/hkang/zthunder_yagent/tmp_logs
export RAY_ADDRESS="<head-node-ip>:6381"
export ROLLOUT_SERVER_URLS="['http://<rollout-node-ip>:18000','http://<rollout-node-ip>:18001']"
export NCCL_SOCKET_IFNAME=ens7
export GLOO_SOCKET_IFNAME=ens7
export XDG_RUNTIME_DIR=/tmp/xdg-test-$USER
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
export DOCKER_DATA_ROOT=/scratch/$USER/docker-rootless

export FULL_TRAIN_BATCH_SIZE=32
export FULL_POLICY_MINI_BATCH_SIZE=32
export FULL_MICRO_FORWARD_BATCH_SIZE_PER_GPU=1
export FULL_MICRO_TRAIN_BATCH_SIZE_PER_GPU=1
export FULL_N_SAMPLES=2
export FULL_NUM_PARALLEL_GENERATION_WORKERS=32
export FULL_MAX_CONCURRENCY=4
export FULL_TRAJ_PER_SEC=1
export FULL_MAX_STALENESS_STEPS=1
export FLASH_ATTN=true
export TRAIN_MAX_SEQ_LEN=6144

stdbuf -oL -eL bash examples/train_integrations/harbor/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh full 2>&1 | tee /home/hkang/zthunder_yagent/tmp_logs/$RUN_NAME_OVERRIDE/launcher-interactive.log
```

This works because the current launcher still supports all required overrides:

- [run_codecontest_qwen3_32b_6node_rootless_fully_async.sh](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh#L175)
- [run_codecontest_qwen3_32b_6node_rootless_fully_async.sh](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh#L282)

### 7. Monitor until step 20

The previous `r12` run did not stop from inside the launcher. It was monitored externally and stopped after step `20`.

The most reliable direct signal is the active Ray worker log under `/scratch`, not only `launcher-interactive.log`.

Use this loop on the head node:

```bash
while true; do
  rg -a -n "trainer/global_step" /scratch/$USER/**/session_latest/logs/worker-*.err 2>/dev/null | tail -n 5
  sleep 30
done
```

Stop condition for exact `r12` reproduction:

- as soon as you confirm `trainer/global_step: 20`

Do not wait for a `global_step_20` checkpoint if you want exact `r12` behavior.

The original `r12` run stopped after step `20` but before a step-20 save happened.

### 8. Stop the run and preserve artifacts

Exact `r12`-style stop:

- stop the training launcher on the head node
- stop the rollout script on the rollout node
- leave the saved artifacts on disk

Minimal cleanup:

```bash
# head node
pkill -f run_codecontest_qwen3_32b_6node_rootless_fully_async.sh || true

# rollout node
pkill -f skyrl.backends.skyrl_train.inference_engines.vllm.vllm_server || true

# optional Ray cleanup after artifacts are safe
/home/hkang/zthunder_yagent/SkyRL/.venv/bin/ray stop -f || true
```

If you want to keep the Harbor head Docker daemon running for follow-up runs, do not kill `dockerd-rootless.sh`.

## Expected Outputs After A Successful Repro

### Artifacts

- run root:
  - `/data/zy/models/hkang/harbor_runs/<RUN_NAME>`
- checkpoints:
  - `global_step_5`
  - `global_step_10`
  - `global_step_15`
- latest checkpoint pointer:
  - `latest_ckpt_global_step.txt` containing `15`

### Logs

- launcher:
  - `/home/hkang/zthunder_yagent/tmp_logs/<RUN_NAME>/launcher-interactive.log`
- ThunderAgent:
  - `/home/hkang/zthunder_yagent/tmp_logs/<RUN_NAME>/thunderagent.log`
- trainer memory events:
  - `/home/hkang/zthunder_yagent/tmp_logs/<RUN_NAME>/memory_events/*.jsonl`
- head-node monitor:
  - `/home/hkang/zthunder_yagent/tmp_logs/<RUN_NAME>/monitoring`
- rollout logs:
  - `/home/hkang/zthunder_yagent/tmp_logs/<RUN_NAME>/rollout/rollout_a.log`
  - `/home/hkang/zthunder_yagent/tmp_logs/<RUN_NAME>/rollout/rollout_b.log`

## Expected Runtime Shape

These are the main things to expect if the repro is behaving like `r12`:

- initialization takes a long time before step `1`
- train compute per step is under a minute
- end-to-end wall time to step `20` is much longer because Harbor rollout is the bottleneck
- step checkpoints appear only at `5`, `10`, `15`
- no `step 20` checkpoint if you stop exactly like `r12`

## Post-Run Checks

### Confirm step count

```bash
/home/hkang/zthunder_yagent/SkyRL/.venv/bin/python - <<'PY'
import csv
import os
from pathlib import Path
run_name = os.environ['RUN_NAME_OVERRIDE']
p = Path(f'/home/hkang/zthunder_yagent/tmp_logs/{run_name}/analysis/step_peak_summary.csv')
rows = list(csv.DictReader(p.open()))
print(len(rows))
print(rows[0])
print(rows[-1])
PY
```

For a clean repro, the length should be `20`.

### Confirm latest checkpoint pointer

```bash
cat /data/zy/models/hkang/harbor_runs/<RUN_NAME>/ckpts/latest_ckpt_global_step.txt
```

For exact `r12` behavior, this should still be `15`.

### Generate the same trainer-side memory analysis

```bash
/home/hkang/zthunder_yagent/SkyRL/.venv/bin/python \
  /home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/analyze_memory_trace.py \
  --run-dir /home/hkang/zthunder_yagent/tmp_logs/$RUN_NAME_OVERRIDE
```

Expected outputs:

- `/home/hkang/zthunder_yagent/tmp_logs/$RUN_NAME_OVERRIDE/analysis/memory_trace_overview.png`
- `/home/hkang/zthunder_yagent/tmp_logs/$RUN_NAME_OVERRIDE/analysis/memory_trace_summary.md`
- `/home/hkang/zthunder_yagent/tmp_logs/$RUN_NAME_OVERRIDE/analysis/step_peak_summary.csv`

## Main Lessons From `r12`

- `flash_attn=true` was the difference between surviving to step `20` and earlier OOM behavior.
- Rootless Harbor verifier is only stable with the patched local Harbor Docker upload path.
- `r12` was a valid end-to-end run, but it was still conservative:
  - `32/32/1/1`
  - `n_samples=2`
  - `num_parallel_generation_workers=32`
  - `max_concurrency=4`
- The current launcher defaults are now much more aggressive, so exact `r12` reproduction requires explicit overrides.
