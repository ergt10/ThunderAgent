# R2EGYM Qwen3-32B 6-Node Rootless Fully Async 1-Step 4-Server mini-swe-agent Replay Runbook

Date: 2026-03-19

This runbook tracks the live rerun launched after the 2026-03-18 replay exposed Harbor setup instability.

## 1. Primary Differences From `20260318`

Compared with:

- `/home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/R2EGYM_QWEN3_32B_6NODE_ROOTLESS_FULLY_ASYNC_1STEP_4SERVER_MINISWE50_TIMEOUT9000_FD131072_BLOCKDOCKERD_REPLAY_RUNBOOK_20260318.md`

This rerun keeps the same topology and image reuse policy, but changes the Harbor setup path:

- continue reusing pre-pulled Docker images from `/scratch/triton_cache/$USER/r2egym-rootless-image-cache`
- use a shared preinstalled `mini-swe-agent` tool home instead of per-trial online install
- mount the host uv-managed Python runtime into Harbor task containers so the shared `mini` executable remains runnable
- keep `llm_kwargs.timeout=1200`
- set Harbor `max_turns=20`
- persist exact Harbor docker exec commands into each trial under `agent/docker-exec-history.log`
- persist setup request artifacts before execution under `agent/setup/`

## 2. Canonical Entry Point

Launch with:

```bash
cd /home/hkang/zthunder_agent/SkyRL
export PATH=/home/hkang/zthunder_agent/SkyRL/.venv/bin:$HOME/.local/bin:$PATH

JOB_ID=1504 \
bash /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/run_r2egym_qwen3_32b_6node_rootless_full_1step_miniswe50_timeout9000_fd131072_blockdockerd_replay.sh
```

## 3. Fixed Runtime Inputs

- `JOB_ID=1504`
- `HEAD_NODE=research-dev-coder-003`
- `ROLLOUT_NODE=research-dev-coder-008`
- `TRAINER_NODES_CSV=research-dev-coder-012,research-dev-coder-013,research-dev-coder-014,research-dev-coder-015`
- `DOCKER_DATA_ROOT=/scratch/triton_cache/$USER/r2egym-rootless-image-cache`
- `DOCKER_INOTIFY_MAX_USER_INSTANCES=8192`
- `HEAD_DOCKER_READY_TIMEOUT_SEC=300`
- `HARBOR_AGENT_MAX_TURNS=20`
- `AGENT_TIMEOUT_SEC=9000`
- `MINI_SWE_MODEL_TIMEOUT_SEC=1200`

## 4. Shared mini-swe-agent Layout

The rerun prepares these shared host paths on the head node before training starts:

- shared uv cache:
  - `/scratch/triton_cache/$USER/harbor-uv-cache`
- shared preinstalled tool home:
  - `/scratch/triton_cache/$USER/harbor-mini-swe-home`
- shared uv-managed Python runtime:
  - `/home/hkang/zthunder_agent/.local/share/uv/python`

Those paths are mounted into Harbor task containers so setup can short-circuit to:

- `PATH=$HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME/.local/bin:$PATH`
- `mini`

instead of running per-trial package installation.

## 5. Timeout Forensics

If Harbor setup still times out, inspect the timed-out trial directory for:

- `agent/setup/requested-command.txt`
- `agent/setup/requested-env.txt`
- `agent/setup/install.sh`
- `agent/docker-exec-history.log`

Those files should identify the exact command Harbor sent into Docker for the setup path.

## 6. Live Run Notes

- `2026-03-19 01:07:16 PDT`: initialized rerun runbook before launch; pending shared tool-home smoke and full-run launch.
- `2026-03-19 01:09:00 PDT`: head-node smoke passed for shared preinstalled tool home. Verified `mini` runs from `/scratch/triton_cache/hkang/harbor-mini-swe-home/.local/bin/mini` with `HOME=/scratch/triton_cache/hkang/harbor-mini-swe-home` and shared uv cache `/scratch/triton_cache/hkang/harbor-uv-cache`.
- `2026-03-19 01:09:32 PDT`: launched replay wrapper with `JOB_ID=1504`, `DOCKER_INOTIFY_MAX_USER_INSTANCES=8192`, and `HARBOR_AGENT_MAX_TURNS=20`.
- `2026-03-19 01:09:32 PDT`: actual run name is `r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-20260319_010932`.
- `2026-03-19 01:10 PDT`: rootless Docker startup reached `Loading containers: start.` and then became slow to answer `docker info`; no new Harbor setup activity has started yet.
- `2026-03-19 01:11:36 PDT`: rootless dockerd reached `Daemon has completed initialization` and opened `/tmp/xdg-r2e1srp-0319010932/docker.sock`, but the wrapper had already hit `wait_for_head_docker 120` and canceled step `1504.57`. This attempt never entered Harbor.
- `2026-03-19 01:13 PDT`: patched the launcher to use `HEAD_DOCKER_READY_TIMEOUT_SEC=300` for the next rerun.
- `2026-03-19 01:14:39 PDT`: launched the next rerun with run name `r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-20260319_011439`.
- `2026-03-19 01:16:18 PDT`: rootless dockerd completed initialization on `research-dev-coder-003` and exposed `/tmp/xdg-r2e1srp-0319011439/docker.sock` within the extended 300-second window.
- `2026-03-19 01:16:31 PDT`: Ray cluster came up cleanly on all 5 training nodes; no placement or resource failures at bring-up.
- `2026-03-19 01:16:30 PDT` onward: rollout servers started on `research-dev-coder-008`; early monitor `Connection refused` lines were just pre-health probes before vLLM finished loading.
- `2026-03-19 01:22:39 PDT`: all four rollout servers passed `/health`; wrapper moved past the rollout barrier and started trainer monitors, ThunderAgent watchdog, and the training driver.
- `2026-03-19 01:23:00 PDT`: `launcher_train_driver.log` shows ThunderAgent router live at `http://172.21.44.54:8080`, HarborGenerator initialized against `/home/hkang/zthunder_agent/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-20260319_011439/trials_run`, and `Total training steps: 1`.
- `2026-03-19 01:24-01:26 PDT`: training is still in actor/process-group bring-up; monitoring TSVs remain header-only and `trials_run/` is still empty, so generation buffer progress is still `0` at this point.
- `2026-03-19 01:29 PDT`: trainer bring-up is now clearly in model-load, not Harbor setup. `trainer_monitors/*/gpu_processes.tsv` shows both `ray::FSDPPolicyWorkerBase.init_model` and `ray::FSDPRefWorkerBase.init_model` active on all trainer GPUs, while `gpu_summary.tsv` shows roughly `74 GiB` used and `100%` utilization. `trials_run/` is still empty, so generation buffer remains `0` until trainer-side model initialization finishes.
- `2026-03-19 01:37:30 PDT`: trainer finished `sync_weights_to_inference_engines` and entered `wait_for_generation_buffer`; the run is no longer stuck in trainer init.
- `2026-03-19 01:39-01:40 PDT`: Harbor trial creation is live. `trials_run/` now contains `256` trial directories and the first `trajectory.json` files have appeared, so generation buffer progress is at least `2/64`.
- `2026-03-19 01:40 PDT`: the shared preinstalled `mini-swe-agent` fast path did **not** trigger. First setup outputs still show `apt-get` and `uv tool install`, not `USING_SHARED_PREINSTALLED_MINI_SWE_AGENT=...`.
- `2026-03-19 01:40 PDT`: root cause of the fast-path miss is configuration, not Docker image reuse. `docker-compose-base.yaml` mounts the shared tool home, but the container execution path only injects `DEBIAN_FRONTEND=noninteractive`; `HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME` and related variables are not present inside the setup container process, so `install.sh` falls through to the online-install branch.
- `2026-03-19 01:45-01:46 PDT`: patched `docker-compose-base.yaml` so the `main` service now carries `HARBOR_SHARED_UV_CACHE_ENV_DIR`, `HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME`, `HARBOR_SHARED_UV_PYTHON_ENV_DIR`, `HARBOR_MINI_SWE_AGENT_GIT_REF`, and `HARBOR_MINI_SWE_AGENT_UV_OFFLINE` as container environment variables, not just mounted paths.
- `2026-03-19 01:53 PDT`: compose-level validation passed. `docker compose ... config` on the real task environment now renders the expected `HARBOR_*` variables into `services.main.environment`.
- `2026-03-19 01:53 PDT`: head-node shell validation passed. Running the rendered `install.sh` with `HOME` on `research-dev-coder-003`, `PATH=/usr/bin:/bin`, and `HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME=/scratch/triton_cache/hkang/harbor-mini-swe-home` now prints `USING_SHARED_PREINSTALLED_MINI_SWE_AGENT=/scratch/triton_cache/hkang/harbor-mini-swe-home` and exits without touching `apt-get` or `uv tool install`.
- `2026-03-19 01:54 PDT`: validation-specific rootless Docker smoke against the shared `DOCKER_DATA_ROOT` was inconclusive because that daemon again hung in `Loading containers: start.` before the compose trial could run. This did not block the patch validation itself; the env-injection fix was validated independently of the dirty rootless data-root state.
- `2026-03-19 01:53:24 PDT`: launched a fresh full-run attempt with run name `r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-20260319_015324`, but rootless Docker again stalled in `Loading containers: start.` and `docker info` never got past the `Server:` header.
- `2026-03-19 01:55-01:56 PDT`: inspected the shared rootless `DOCKER_DATA_ROOT` and found heavy stale runtime state: `932` `containers/` entries, `899` `containerd runtime.v2.task/moby` entries, `932` `image/overlay2/layerdb/mounts` entries, and a `4.0M` `network/files/local-kv.db`.
- `2026-03-19 01:56 PDT`: backed up only the stale container/network state to `/scratch/triton_cache/hkang/r2egym-rootless-state-backups/20260319_015646` and recreated empty `containers/`, `containerd/.../runtime.v2.task/moby/`, and `image/overlay2/layerdb/mounts/` directories. Image-layer data under the shared rootless cache was left intact.
- `2026-03-19 01:57:02 PDT`: launched the next full-run attempt with run name `r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-20260319_015702`.
- `2026-03-19 01:57:04 PDT`: state cleanup fixed rootless Docker bring-up. The daemon reached `Loading containers: done.`, completed initialization, and opened `/tmp/xdg-r2e1srp-0319015702/docker.sock`.
- `2026-03-19 02:28 PDT`: the current active run (`r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-20260319_020225`) confirmed the fast-path miss is caused by env loss between `run_codecontest_qwen3_32b_6node_rootless_fully_async.sh` and `ray::skyrl_entrypoint`. The parent `run_codecontest...` process still has `HARBOR_SHARED_*` and `HARBOR_MINI_SWE_AGENT_*`, but `ray::skyrl_entrypoint` only retains `HARBOR_DOCKER_KEEP_IMAGES=1`. A live trial container (`r2egym-0145__raevkwm`) sees defaults instead of the intended shared settings: `HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME=/tmp/harbor-mini-swe-home`, `HARBOR_MINI_SWE_AGENT_GIT_REF=`, and `HARBOR_MINI_SWE_AGENT_UV_OFFLINE=0`. `mini` resolves from `/root/.local/bin/mini`, confirming per-trial local install is still active.
- `2026-03-19 02:29 PDT`: live progress snapshot after the above confirmation: `trial_dirs=256`, `trajectory_json=35`, `setup_stdout=182`, `fastpath_hits=0`, `exception_txt=0`, `result_json=0`. ThunderAgent `/health` is still healthy, but rootless Docker continues to log container cleanup churn (`Container failed to exit within 10s of kill`, `shim disconnected`, `cleaning up dead shim`).
- `2026-03-19 02:34 PDT`: `trajectory.json` count is not a reliable proxy for generation-buffer readiness. At this point the run has `trajectory_json=85`, but ThunderAgent has only emitted `14` `/programs/release` events and Harbor has written only `5` `result.json` files. `launcher_train_driver.log` still shows only `Started: 'wait_for_generation_buffer'`, so the trainer appears to be waiting on released/completed samples rather than raw trajectory-file count.
- `2026-03-19 03:43 PDT`: patched `skyrl/train/utils/utils.py` so Ray runtime env passthrough now preserves `HARBOR_SHARED_UV_CACHE_*`, `HARBOR_SHARED_MINI_SWE_TOOL_*`, `HARBOR_SHARED_UV_PYTHON_*`, `HARBOR_MINI_SWE_AGENT_GIT_REF`, and `HARBOR_MINI_SWE_AGENT_UV_OFFLINE`.
- `2026-03-19 03:44 PDT`: helper-level validation passed. Calling `prepare_runtime_environment()` with the launcher env now returns all eight Harbor fast-path variables with the intended shared host paths and `HARBOR_MINI_SWE_AGENT_UV_OFFLINE=1`.
- `2026-03-19 03:45 PDT`: Ray runtime-env validation passed. A local Ray remote function launched with the returned `runtime_env["env_vars"]` sees the same Harbor fast-path variables inside the worker process.
- `2026-03-19 03:47 PDT`: real container validation passed against task `r2egym-0145` using the prebuilt image from its `task.toml`. Running the current `install-mini-swe-agent.sh` inside the task container printed `USING_SHARED_PREINSTALLED_MINI_SWE_AGENT=/scratch/triton_cache/hkang/harbor-mini-swe-home` followed by `INSTALL_SUCCESS`, confirming the fast path now short-circuits before `apt-get` or `uv tool install`.
- `2026-03-19 03:50 PDT`: canceled the validation-only Slurm child step `1504.242` that held the smoke rootless Docker daemon. Post-cleanup check shows only `1504.batch` remains active; no `fastpath-smoke` processes are left behind.
