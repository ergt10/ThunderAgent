# Qwen3-32B 6-Node Rootless Harbor Full-Async `r10` Run Notes

Date: 2026-03-11

## Scope

This note records the final effective setup, reproducible launch order, observed outcome, and reusable lessons from the real 32B full run:

- Run name: `codecontest-qwen3-32b-6node-rootless-full-r10`
- Goal: `4 trainer + 1 rollout + 1 CPU Harbor head`
- Result: training worked end-to-end through real Harbor + ThunderAgent traffic, reached step `5/301`, then failed with training-side CUDA OOM

This is the follow-up to the earlier rootless Harbor verifier failure note in:

- `examples/train_integrations/harbor/QWEN3_32B_6NODE_ROOTLESS_FULLY_ASYNC_R4_FAILURE_NOTES.md`

## Environment

### Node topology

- Harbor CPU head / rootless Docker / Ray head: `research-secure-14` via job `28692`
- Trainer nodes:
  - `research-secure-23` via job `28283`
  - `research-secure-06` via job `28600`
  - `research-secure-18` via job `28307`
  - `research-secure-11` via job `28595`
- External rollout node: `research-secure-17` via job `28584`

### Storage

All large artifacts were kept off `/home`:

- model weights: `/data/zy/models/hkang/models/Qwen3-32B`
- HF cache: `/data/zy/models/hub`
- run artifacts: `/data/zy/models/hkang/harbor_runs/codecontest-qwen3-32b-6node-rootless-full-r10`
- logs: `/data/zy/models/hkang/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-r10`
- scratch/runtime caches: `/scratch/hkang/skyrl_runtime`

### Required code state

This run did not use a stock repo + stock Harbor install. The following pieces were part of the working setup:

- head-pinned Harbor ThunderAgent entrypoint:
  - `examples/train_integrations/harbor/entrypoints/main_harbor_thunder_agent_fully_async_head_pinned.py`
- external rollout script:
  - `examples/train_integrations/harbor/start_qwen3_32b_external_rollout_servers.sh`
- 32B full launcher:
  - `examples/train_integrations/harbor/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh`
- Harbor rootless Docker upload patch in local site-packages:
  - `.venv/lib/python3.13/site-packages/harbor/environments/docker/docker.py`
  - `upload_file()` and `upload_dir()` stream tar into container with normalized ownership instead of `docker compose cp`

Without that Harbor patch, rootless verifier fails before meaningful training with `AddTestsDirError`.

## Effective config

### Training side

- strategy: `fsdp2`
- trainer nodes: `4`
- GPUs per trainer node: `8`
- total training GPUs: `32`
- `trainer.placement.colocate_all=false`
- `trainer.placement.colocate_policy_ref=true`
- `train_batch_size=32`
- `policy_mini_batch_size=32`
- `micro_forward_batch_size_per_gpu=1`
- `micro_train_batch_size_per_gpu=1`
- `update_epochs_per_batch=1`
- `gradient_checkpointing=true`
- `flash_attn=false`
- `use_sample_packing=false`
- `trainer.algorithm.max_seq_len=6144`
- `max_prompt_length=512`
- `max_generate_length=1024`
- TIS: `token`
- `generator.sampling_params.logprobs=1`

### Rollout side

- external rollout only, not local rollout
- rollout node: `research-secure-17`
- vLLM servers: `2`
- per-server TP: `4`
- ports: `18000`, `18001`
- `gpu_memory_utilization=0.8`
- `max_model_len=32768`
- ThunderAgent mode: `tr`

### Important env

- `RAY_ADDRESS=172.27.31.10:6381`
- `NCCL_SOCKET_IFNAME=ens7`
- `GLOO_SOCKET_IFNAME=ens7`
- `_SKYRL_USE_NEW_INFERENCE=1`
- `DOCKER_HOST=unix:///tmp/xdg-test-$USER/docker.sock`
- `HF_HOME=/data/zy/models`
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## Repro steps

### 1. Start rootless Docker on `research-secure-14`

Run inside job `28692`:

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
  > "$SCRATCH/dockerd_test.log" 2>&1 &
```

### 2. Start external rollout servers on `research-secure-17`

Run inside job `28584`:

```bash
bash examples/train_integrations/harbor/start_qwen3_32b_external_rollout_servers.sh
```

### 3. Launch the 32B full run on `research-secure-14`

Run inside job `28692`:

```bash
export RAY_ADDRESS=172.27.31.10:6381
export NCCL_SOCKET_IFNAME=ens7
export GLOO_SOCKET_IFNAME=ens7
export XDG_RUNTIME_DIR=/tmp/xdg-test-$USER
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock

bash examples/train_integrations/harbor/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh full
```

If `/data/zy/models/hkang/models/Qwen3-32B` is missing, the launcher downloads it automatically from `Qwen/Qwen3-32B` into that path.

## Outcome

What worked:

- 6-node topology was valid
- real Harbor trials ran through ThunderAgent
- external TP=4 rollout on a single 8-GPU node worked
- full-async training progressed to real optimizer steps
- no verifier-side `AddTestsDirError`

Observed training progress:

- step 1 finished at `05:57:45 PDT`
- step 2 finished at `06:08:08 PDT`
- step 3 finished at `06:21:38 PDT`
- step 4 finished at `06:32:01 PDT`
- step 5 finished at `06:45:59 PDT`

Main logs:

- launcher: `/data/zy/models/hkang/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-r10/launcher-interactive.log`
- ThunderAgent: `/data/zy/models/hkang/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-r10/thunderagent.log`

## Failure

The run failed immediately after step 5 with training-side OOM on `research-secure-23`.

Failure chain:

- top-level: `RayTaskError(OutOfMemoryError)`
- worker: `ray::FSDPPolicyWorkerBase.forward_backward_from_staged()`
- device error: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 11.11 GiB`

Important detail:

- this was not a Harbor verifier failure
- not a ThunderAgent routing failure
- not a rootless Docker failure
- not an NCCL connectivity failure

It was a pure training-memory failure during `policy backward`.

## Reusable lessons

- For rootless Harbor, patching verifier uploads away from `docker compose cp` is mandatory.
- External rollout on a dedicated 8-GPU node is workable for 32B with `2 x TP4` vLLM servers.
- The current 32B training shape is still too aggressive for stable long-running FSDP training:
  - `train_batch_size=32`
  - `policy_mini_batch_size=32`
  - `max_seq_len=6144`
  - `policy/ref colocate=true`
  - `flash_attn=false`
- If the next goal is simply to survive to step 20, reduce training memory pressure first before changing topology.

## Recommended next changes

- lower `train_batch_size` and `policy_mini_batch_size`
- consider lowering `trainer.algorithm.max_seq_len`
- consider enabling `flash_attn`
- keep the same 6-node topology unless there is a separate reason to change scheduling
