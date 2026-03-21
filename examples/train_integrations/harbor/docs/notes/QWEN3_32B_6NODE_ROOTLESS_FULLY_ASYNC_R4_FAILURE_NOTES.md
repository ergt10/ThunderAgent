# Qwen3-32B 6-Node Rootless Harbor Full-Async `r4` Failure Notes

Date: 2026-03-11

## Scope

This note records the bring-up, observed failures, environment, repro steps, and reusable lessons from the 6-node Harbor + SkyRL + ThunderAgent run:

- Run name: `codecontest-qwen3-32b-6node-rootless-full-r4`
- Outcome: training infrastructure came up correctly, rollout traffic flowed through ThunderAgent, but the run did not finish step 0 because Harbor verifier repeatedly failed under rootless Docker.

Postmortem status after subsequent fixes:

- Plan A was implemented in Harbor's Docker environment layer.
- The rootless Docker verifier upload failure was reproduced in isolation, patched, and then validated with a real Harbor `Trial.run()` on `code_contests-5277`.
- The original `AddTestsDirError` is no longer the active blocker.
- The full 32B training run has not yet been restarted after this verifier fix.

## Final Status

At stop time:

- The run had entered full-async step `0`
- `Training Step Progress` was still `0/301`
- `Generation Buffer Progress` had reached `24/32`
- No checkpoint had been saved
- Rollout traffic was actively hitting ThunderAgent and the two external vLLM backends
- The launcher was stopped intentionally with:
  - `scancel 28692.123`
  - `scancel 28584.273`

Post-stop verification:

- rollout ports `18000` and `18001` no longer responded
- Ray resource usage dropped to `0/32 GPU`

## Environment

### Node Topology

- Harbor CPU head / rootless Docker / Ray head: `research-secure-14` via job `28692`
- Trainer nodes:
  - `research-secure-23` via job `28283`
  - `research-secure-06` via job `28600`
  - `research-secure-18` via job `28307`
  - `research-secure-11` via job `28595`
- External rollout node: `research-secure-17` via job `28584`

### Storage Layout

All large artifacts were placed off `/home`:

- Model weights: `/data/zy/models/hkang/models/Qwen3-32B`
- HF cache: `/data/zy/models/hub`
- Run artifacts: `/data/zy/models/hkang/harbor_runs/codecontest-qwen3-32b-6node-rootless-full-r4`
- Logs: `/data/zy/models/hkang/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-r4`
- Scratch/runtime caches: `/scratch/hkang/skyrl_runtime`

### Rootless Docker on Harbor Head

The Harbor CPU node used rootless Docker on `research-secure-14`:

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

### Key Runtime Variables

The 32B launcher exported:

- `_SKYRL_USE_NEW_INFERENCE=1`
- `RAY_ADDRESS=172.27.31.10:6381`
- `NCCL_SOCKET_IFNAME=ens7`
- `GLOO_SOCKET_IFNAME=ens7`
- `HARBOR_DOCKER_KEEP_IMAGES=1`
- `HF_HOME=/data/zy/models`
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

See [run_codecontest_qwen3_32b_6node_rootless_fully_async.sh](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh#L97) and [run_codecontest_qwen3_32b_6node_rootless_fully_async.sh](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh#L101).

## Effective Training Shape

The run used:

- `4` trainer nodes
- `1` rollout node
- `policy/ref` colocated
- `2` external rollout engines
- rollout TP size `4`
- full async
- Harbor Docker environments
- ThunderAgent `tr` scheduler mode

The main training config is in [run_codecontest_qwen3_32b_6node_rootless_fully_async.sh](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh#L275).

Important settings:

- `trainer.placement.colocate_all=false`
- `trainer.placement.colocate_policy_ref=true`
- `trainer.placement.policy_num_nodes=4`
- `trainer.placement.ref_num_nodes=4`
- `generator.inference_engine.run_engines_locally=false`
- `generator.inference_engine.external_server_urls=['http://172.27.21.28:18000','http://172.27.21.28:18001']`
- `generator.sampling_params.logprobs=1`
- `trainer.algorithm.off_policy_correction.tis_ratio_type=token`
- `harbor_trial_config.agent.kwargs.record_terminal_session=false`

## What Worked

The following pieces were confirmed working:

1. The 6-node cluster shape itself was valid.
2. The 4 trainer nodes and 1 rollout node could join the same Ray cluster.
3. The rollout node could serve two external TP=4 vLLM backends.
4. ThunderAgent received real Harbor traffic and dispatched requests to both external backends.
5. The trainer reached the beginning of full-async step `0`.
6. NCCL on the selected trainer/rollout nodes was not the blocking issue for this run.

Evidence:

- ThunderAgent access logs were active in [thunderagent.log](/data/zy/models/hkang/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-r4/thunderagent.log)
- The launcher entered `Started: 'step'` and `Started: 'wait_for_generation_buffer'` in [launcher-interactive.log](/data/zy/models/hkang/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-r4/launcher-interactive.log)

## Primary Blocking Issue

The actual blocker was Harbor verifier under rootless Docker.

### Symptom

Harbor repeatedly failed while copying the tests directory into the container:

```text
docker compose ... cp /.../tests/. main:/tests
Error response from daemon: failed to Lchown "/tests" ... lchown /tests: invalid argument
```

That surfaced as:

- `AddTestsDirError: Failed to add tests directory to environment.`

This happened during verification, after the agent had already run.

### Why It Mattered

The run was not deadlocked in NCCL or Ray. It was starved by bad samples:

- Harbor kept producing trajectories that later failed during verification
- those trajectories were masked out with loss mask `[0]`
- full-async generation buffer fill became slow and noisy
- the run never finished filling the first training buffer

At stop time, observed counters were:

- `AddTestsDirError` / `lchown /tests`: `172`
- `Trajectory ... failed (stop_reason=error)`: `54`

Representative log location:

- [launcher-interactive.log](/data/zy/models/hkang/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-r4/launcher-interactive.log)

Representative failure signature:

```text
RuntimeError: Docker compose command failed ...
docker compose ... cp ... tests/. main:/tests
Error response from daemon: failed to Lchown "/tests" ... invalid argument
...
harbor.verifier.verifier.AddTestsDirError: Failed to add tests directory to environment.
```

### Root Cause

`docker compose cp` into the container's `/tests` path is not safe under this rootless Docker setup when Harbor tries to preserve ownership metadata.

The relevant Harbor code path is still a raw `docker compose cp`:

- [docker.py](/home/hkang/zthunder_yagent/SkyRL/.venv/lib/python3.13/site-packages/harbor/environments/docker/docker.py#L288)

So the blocking issue is not Qwen3-32B, not ThunderAgent, and not cross-node training placement. It is Harbor's verifier-side file upload behavior under rootless Docker.

## Plan A Fix And Validation

### Fix

Plan A was to stop using `docker compose cp` for Harbor uploads.

Implemented change:

- Harbor `upload_dir()` and `upload_file()` now stream a tar archive into the running container instead of calling `docker compose cp`
- the tar archive is created with normalized ownership:
  - `--owner=0`
  - `--group=0`
  - `--numeric-owner`

This avoids carrying the host's large cluster UID/GID into the container, which is what caused rootless Docker to fail `lchown("/tests", 243001624, 243001624)`.

Implementation location:

- [docker.py](/home/hkang/zthunder_yagent/SkyRL/.venv/lib/python3.13/site-packages/harbor/environments/docker/docker.py)

Important limitation:

- this patch currently lives in the local `.venv` site-packages
- if Harbor is reinstalled or the virtualenv is rebuilt, this patch must be re-applied

### Isolated Reproducer

A standalone reproducer was added here:

- [repro_rootless_docker_compose_cp_uid_failure.sh](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/repro_rootless_docker_compose_cp_uid_failure.sh)

Validated on:

- `research-secure-14`
- Slurm job `28692`

It reproduces the original failure without Harbor:

- `docker compose cp <src>/. main:/tests`
- fails with `failed to Lchown "/tests" ... invalid argument`

### Harbor-Level Smoke Validation

After the patch, Harbor's own Docker environment abstraction was smoke-tested on `research-secure-14`:

- `DockerEnvironment.upload_dir(..., "/tests")` succeeded
- `DockerEnvironment.upload_file(..., "/solution/renamed.txt")` succeeded
- container-side ownership became `0:0`, not the host UID/GID

### Real Trial Validation

After the patch, a real Harbor trial was executed:

- task: `code_contests-5277`
- agent: `nop`
- node: `research-secure-14`
- rootless Docker

Artifact path:

- [code_contests-5277__rootlessVerifierFix](/data/zy/models/hkang/harbor_trial_smokes/rootless-verifier-fix/code_contests-5277__rootlessVerifierFix)

Outcome:

- verifier successfully uploaded `/tests`
- verifier ran `test.sh`
- trial completed normally
- no `AddTestsDirError`
- no `exception_info`
- final reward was `0.0`, which is expected because `nop` agent never wrote `/app/solution.py`

Result file:

- [result.json](/data/zy/models/hkang/harbor_trial_smokes/rootless-verifier-fix/code_contests-5277__rootlessVerifierFix/result.json)

This is the decisive proof that the original rootless verifier failure has been fixed at the Harbor upload layer.

## Post-Fix Smoke Bring-up

After the verifier fix, the 32B launcher was smoke-tested against the same live 6-node cluster.

### Smoke Config Fix

The first smoke attempt (`codecontest-qwen3-32b-6node-rootless-smoke-r8`) failed immediately with a config mismatch:

- `max_train_tasks=8`
- `train_batch_size=16`

That hit:

```text
AssertionError: dataset should be atleast as large as `train_batch_size` 16, got size 8
```

Fix applied:

- [run_codecontest_qwen3_32b_6node_rootless_fully_async.sh](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh#L200)
- smoke mode now sets `MAX_TRAIN_TASKS="$SCALED_SMOKE_BATCH"` so that `max_train_tasks >= train_batch_size`

### Smoke Run `r9`

Run name:

- `codecontest-qwen3-32b-6node-rootless-smoke-r9`

Observed status:

- the run passed dataset initialization
- ThunderAgent came up correctly and bound to `http://172.27.31.10:8080`
- HarborGenerator initialized successfully with `api_base=http://172.27.31.10:8080/v1`
- full-async dataloader built successfully with:
  - `Length of train_dataloader: 16`
  - `Total training steps: 1`
- trainer actor groups initialized and reached:
  - `Initialized process group for RayActorGroup`
  - `Mesh Ranks: ... world_size=32, dp_size=32`
- Ray resource usage reached and held:
  - `32.0/32.0 GPU`
  - `32.0/880.0 CPU`

Important negative observations:

- no `AddTestsDirError` was observed in the smoke launcher log

Later during the same smoke run, the training path advanced further:

- `init policy/ref/critic models done`
- `Started: 'step'`
- `Started: 'wait_for_generation_buffer'`
- ThunderAgent began serving real Harbor requests:
  - `/v1/chat/completions ... status=200`
- real Harbor trial directories appeared under:
  - [trials_run](/data/zy/models/hkang/harbor_runs/codecontest-qwen3-32b-6node-rootless-smoke-r9/trials_run)
- completed trial results were written with:
  - `"exception_info": null`
  - `reward: 0.0` on observed completed samples

Interpretation:

- the original verifier blocker does not appear to have regressed
- the rootless Harbor verifier fix is now validated not just in isolation and `Trial.run()`, but also on the real 32B training path
- the remaining startup cost is trainer/ref initialization latency, not the old verifier upload failure

## Secondary Issues Encountered

These were real issues during bring-up, but they were not the final blocker.

### 1. Harbor image deletion race under concurrent trials

Problem:

- Harbor trial cleanup used `docker compose down --rmi all`
- concurrent trials with shared task-level images could remove images another trial still needed

Mitigation used:

- `HARBOR_DOCKER_KEEP_IMAGES=1`
- Harbor patched to skip `--rmi all` when that env var is set

Relevant code:

- [docker.py](/home/hkang/zthunder_yagent/SkyRL/.venv/lib/python3.13/site-packages/harbor/environments/docker/docker.py#L245)
- [run_codecontest_qwen3_32b_6node_rootless_fully_async.sh](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh#L106)

### 2. Terminal session recording was incompatible with rootless Docker

Earlier Harbor runs hit `docker compose cp` ownership failures while copying terminal-recording helper scripts.

Mitigation used:

- disabled Harbor terminal recording with `harbor_trial_config.agent.kwargs.record_terminal_session=false`

Relevant code:

- [run_codecontest_qwen3_32b_6node_rootless_fully_async.sh](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh#L351)

This removed one rootless-Docker-specific failure mode, but verifier-side `tests/` upload still remained.

### 3. New inference validator initially disagreed with the 8B training semantics

Problem:

- legacy validation blocked `sampling_params.logprobs` whenever `run_engines_locally=false`
- that conflicted with the new inference path used by ThunderAgent external servers

Fix applied:

- only legacy remote inference remains blocked
- new inference + `external_server_urls` now permits chosen-token logprobs

Relevant code:

- [utils.py](/home/hkang/zthunder_yagent/SkyRL/skyrl/train/utils/utils.py#L381)
- [utils.py](/home/hkang/zthunder_yagent/SkyRL/skyrl/train/utils/utils.py#L401)

This restored 8B-equivalent training semantics:

- `logprobs=1`
- TIS enabled
- ThunderAgent external rollout

### 4. Ray workers initially missed critical environment variables

Problem:

- Ray workers did not automatically inherit enough of the head-node environment
- this affected new inference, Docker control plane, cache paths, and Python import path

Fix applied:

- runtime env propagation was expanded

Relevant code:

- [utils.py](/home/hkang/zthunder_yagent/SkyRL/skyrl/train/utils/utils.py#L523)
- [utils.py](/home/hkang/zthunder_yagent/SkyRL/skyrl/train/utils/utils.py#L594)

### 5. External rollout world size discovery needed a fallback

Problem:

- external servers may not expose all SkyRL control-plane endpoints at bring-up time
- weight-sync init needed `inference_world_size`

Fix applied:

- for external servers, if `get_world_size()` returns `404`, fall back to config-derived world size

Relevant code:

- [worker.py](/home/hkang/zthunder_yagent/SkyRL/skyrl/backends/skyrl_train/workers/worker.py#L418)

### 6. Shutdown path still has a `/pause` mismatch for the external rollout script

After the run was stopped, the launcher hit:

```text
aiohttp.client_exceptions.ClientResponseError: 404 ... url='http://172.27.21.28:18000/pause'
```

This happened during trainer cleanup, not during the main verifier failure loop.

Why:

- the external rollout script uses `skyrl.backends.skyrl_train.inference_engines.vllm.vllm_server`
- that server file exposes custom weight-transfer endpoints, but not `/pause` and `/resume`

Relevant files:

- [start_qwen3_32b_external_rollout_servers.sh](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/start_qwen3_32b_external_rollout_servers.sh#L102)
- [vllm_server.py](/home/hkang/zthunder_yagent/SkyRL/skyrl/backends/skyrl_train/inference_engines/vllm/vllm_server.py#L88)
- [vllm_server_actor.py](/home/hkang/zthunder_yagent/SkyRL/skyrl/backends/skyrl_train/inference_servers/vllm_server_actor.py#L422)

So the cleanup path still needs alignment if external rollout continues to use `vllm_server.py`.

## Reproduction Steps

### 1. Start rootless Docker on the Harbor head node

Use the command block shown above on `research-secure-14`.

### 2. Bring up the live Ray cluster

- head on `research-secure-14`
- trainer workers on `23 / 06 / 18 / 11`
- rollout node on `17`

### 3. Start external rollout servers on `research-secure-17`

Use:

- [start_qwen3_32b_external_rollout_servers.sh](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/start_qwen3_32b_external_rollout_servers.sh)

This launches:

- `http://172.27.21.28:18000`
- `http://172.27.21.28:18001`

### 4. Start the 32B Harbor full-async run on `research-secure-14`

Use:

- [run_codecontest_qwen3_32b_6node_rootless_fully_async.sh](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh)

Representative launcher wrapper:

```bash
export RAY_ADDRESS=172.27.31.10:6381
export RUN_NAME_OVERRIDE=codecontest-qwen3-32b-6node-rootless-full-r4
bash examples/train_integrations/harbor/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh full
```

### 5. Wait for Harbor verifier failures

The blocking repro signature is:

```text
docker compose ... cp ... tests/. main:/tests
Error response from daemon: failed to Lchown "/tests" ... invalid argument
harbor.verifier.verifier.AddTestsDirError: Failed to add tests directory to environment.
```

## Practical Lessons

### 1. Rootless Harbor Docker is not yet verifier-safe

For this workflow, rootless Docker is good enough to:

- build Harbor environments
- run agent containers
- handle training control plane

But it is not yet safe for Harbor verifier file upload via `docker compose cp`.

If Harbor verification must stay enabled, the next serious fix should be:

- replace verifier-side `docker compose cp` with bind mounts or another ownership-safe transfer path

### 2. Keep all heavy storage off `/home`

This part worked and should be preserved:

- weights under `/data/zy/models/...`
- HF cache under `/data/zy/models/...`
- compile/runtime caches under `/scratch/...`

### 3. Keep `record_terminal_session=false` for rootless Harbor

This avoids an earlier, separate `cp` ownership problem and should remain the default in this setup.

### 4. Keep `HARBOR_DOCKER_KEEP_IMAGES=1` for concurrent Harbor runs

This avoids image teardown races between overlapping trials.

### 5. Keep the validator/runtime-env patches if using ThunderAgent external rollout

The following changes were necessary for the 32B topology to even start correctly:

- new inference validator aligned with `logprobs=1` + TIS
- Ray runtime env passthrough for Docker/network/cache/PYTHONPATH
- external-server world-size fallback during weight-sync init

### 6. If external rollout remains external, finish the control-plane API

The external rollout server path should expose the same control-plane endpoints expected by `RemoteInferenceClient`, especially:

- `/pause`
- `/resume`
- possibly any other lifecycle endpoints used in full-async cleanup

## Recommended Next Step

If this experiment is resumed, the most important next change is not trainer placement or NCCL tuning. It is to make Harbor verifier compatible with rootless Docker.

The cleanest next target is:

1. patch Harbor environment upload to avoid `docker compose cp` ownership rewriting for `/tests`
2. keep the current 6-node topology unchanged
3. keep the current 32B training script unchanged except for any cleanup after the verifier fix

## Key Paths

- Launcher log: [launcher-interactive.log](/data/zy/models/hkang/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-r4/launcher-interactive.log)
- ThunderAgent log: [thunderagent.log](/data/zy/models/hkang/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-r4/thunderagent.log)
- Infra log: [infra-260311_033149.log](/data/zy/models/hkang/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-r4/infra-260311_033149.log)
- Run artifact root: [codecontest-qwen3-32b-6node-rootless-full-r4](/data/zy/models/hkang/harbor_runs/codecontest-qwen3-32b-6node-rootless-full-r4)
