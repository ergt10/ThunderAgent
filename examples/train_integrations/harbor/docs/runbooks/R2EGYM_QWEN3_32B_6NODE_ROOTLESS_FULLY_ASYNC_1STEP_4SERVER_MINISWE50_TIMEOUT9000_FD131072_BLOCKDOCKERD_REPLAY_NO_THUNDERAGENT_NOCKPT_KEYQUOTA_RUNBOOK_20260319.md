# R2EGYM Qwen3-32B 6-Node Rootless Fully Async 1-Step 4-Server mini-swe-agent Replay Runbook (No ThunderAgent, No Checkpoint, Temporary Key Quota)

Date: 2026-03-19

This runbook tracks the live rerun launched from the validated `20260319` no-ThunderAgent replay path, with one additional temporary head-node system change required by the last failed attempt:

- keep ThunderAgent disabled
- keep checkpoint / HF export disabled
- keep the shared preinstalled `mini-swe-agent` fast path
- keep reusing pre-pulled Docker images from the shared rootless cache
- temporarily raise head-node `kernel.keys.maxkeys/maxbytes` before launch

## 1. Parent Contracts

This run follows the exact topology and launcher contract from:

- `/home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/R2EGYM_QWEN3_32B_6NODE_ROOTLESS_FULLY_ASYNC_1STEP_4SERVER_MINISWE50_TIMEOUT9000_FD131072_BLOCKDOCKERD_REPLAY_RUNBOOK_20260319.md`
- `/home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/R2EGYM_QWEN3_32B_6NODE_ROOTLESS_FULLY_ASYNC_1STEP_4SERVER_MINISWE50_TIMEOUT9000_FD131072_BLOCKDOCKERD_REPLAY_NO_THUNDERAGENT_NOCKPT_RUNBOOK_20260319.md`

The only added runtime delta for this rerun is the temporary key-quota expansion on the head node:

- `kernel.keys.maxkeys=20000`
- `kernel.keys.maxbytes=25000000`

## 2. Canonical Entry Point

Launch with:

```bash
cd /home/hkang/zthunder_agent/SkyRL
export PATH=/home/hkang/zthunder_agent/SkyRL/.venv/bin:$HOME/.local/bin:$PATH

JOB_ID=1731 \
RUN_TS=<yyyymmdd_hhmmss> \
bash /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/run_r2egym_qwen3_32b_6node_rootless_full_1step_miniswe50_timeout9000_fd131072_blockdockerd_replay_no_thunderagent_nockpt_keyquota.sh
```

## 3. Fixed Runtime Inputs

- `JOB_ID=1731`
- `HEAD_NODE=research-dev-coder-003`
- `ROLLOUT_NODE=research-dev-coder-008`
- `TRAINER_NODES_CSV=research-dev-coder-012,research-dev-coder-013,research-dev-coder-014,research-dev-coder-015`
- `DOCKER_MODE=rootless`
- `DOCKER_DATA_ROOT=/scratch/triton_cache/$USER/r2egym-rootless-image-cache`
- `DOCKER_INOTIFY_MAX_USER_INSTANCES=8192`
- `HEAD_DOCKER_READY_TIMEOUT_SEC=300`
- `AGENT_TIMEOUT_SEC=9000`
- `MINI_SWE_MODEL_TIMEOUT_SEC=1200`
- `HARBOR_AGENT_MAX_TURNS=20`
- `MAX_TRAIN_TASKS=64`
- `SKYRL_DISABLE_THUNDERAGENT=1`
- `THUNDERAGENT_WATCHDOG_ENABLED=0`
- `CKPT_INTERVAL=-1`
- `HF_SAVE_INTERVAL=-1`
- `KERNEL_KEYS_MAXKEYS=20000`
- `KERNEL_KEYS_MAXBYTES=25000000`

## 4. Expected Fast Path

The rerun should still hit the validated shared-install fast path:

- shared uv cache: `/scratch/triton_cache/$USER/harbor-uv-cache`
- shared preinstalled tool home: `/scratch/triton_cache/$USER/harbor-mini-swe-home`
- shared uv-managed Python runtime: `/home/hkang/zthunder_agent/.local/share/uv/python`

Success markers:

- setup prints `USING_SHARED_PREINSTALLED_MINI_SWE_AGENT=...`
- setup avoids `apt-get`, `curl https://astral.sh`, and `uv tool install ... mini-swe-agent`
- trial containers resolve `mini` from the shared mounted tool home, not `/root/.local/bin/mini`

## 5. Live Notes

- `2026-03-19 20:08-20:09 PDT`: initialized runbook and validated the new wrapper. `bash -n` passed for the new key-quota wrapper and the inherited no-ThunderAgent/no-ckpt wrapper.
- `2026-03-19 20:09 PDT`: fast-path validation passed on `research-dev-coder-003`. Running `install-mini-swe-agent.sh` with the shared tool-home env printed `USING_SHARED_PREINSTALLED_MINI_SWE_AGENT=/scratch/triton_cache/hkang/harbor-mini-swe-home` followed by `INSTALL_SUCCESS`.
- `2026-03-19 20:09 PDT`: Ray runtime-env passthrough validation also passed. `prepare_runtime_environment(SkyRLTrainConfig())` preserved all Harbor fast-path variables, including `HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME=/scratch/triton_cache/hkang/harbor-mini-swe-home` and `HARBOR_MINI_SWE_AGENT_UV_OFFLINE=1`.
- `2026-03-19 20:09:46 PDT`: launched `run_r2egym_qwen3_32b_6node_rootless_full_1step_miniswe50_timeout9000_fd131072_blockdockerd_replay_no_thunderagent_nockpt_keyquota.sh` with `JOB_ID=1731` and `RUN_TS=20260319_200946`.
- `2026-03-19 20:09:47 PDT`: actual run name is `r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-20260319_200947`.
- `2026-03-19 20:09 PDT`: the wrapper applied the temporary head-node key quota successfully: `kernel.keys.maxkeys = 20000` and `kernel.keys.maxbytes = 25000000`.
- `2026-03-19 20:10 PDT`: rootless Docker bring-up on `research-dev-coder-003` completed successfully and exposed `unix:///tmp/xdg-r2e1srp-0319200947/docker.sock` over the shared pre-pulled image cache at `/scratch/triton_cache/hkang/r2egym-rootless-image-cache`.
- `2026-03-19 20:10 PDT`: Ray cluster bring-up completed cleanly on `003,012-015`; autoscaler status shows 5 active nodes, 32 GPUs total, and no pending demands or recent failures.
- `2026-03-19 20:10-20:11 PDT`: rollout startup is in progress on `research-dev-coder-008`. `launcher_rollout.log` only shows early `Connection refused` metric scrapes so far; `launcher_train_driver.log` has not been created yet, which means the wrapper is still waiting on the rollout `/health` barrier.
- `2026-03-19 20:11 PDT`: `trials_run/` is still absent, so Harbor trial creation and generation-buffer filling have not started yet.
- `2026-03-19 20:11:57 PDT`: `rollout_a` finished route registration, `Application startup complete`, and passed both local and remote `/health` checks. `launcher_rollout.log` recorded `rollout_a healthy at http://127.0.0.1:18000`.
- `2026-03-19 20:12 PDT`: `rollout_b` started its normal V1 engine initialization and model loading. `rollout_c` and `rollout_d` remain pending behind the sequential startup barrier, so `launcher_train_driver.log` is still absent and `trials_run/` is still empty.
- `2026-03-19 20:13:16 PDT`: `rollout_b` completed route registration, `Application startup complete`, and passed both local and remote `/health` checks. `launcher_rollout.log` recorded `rollout_b healthy at http://127.0.0.1:18001`.
- `2026-03-19 20:13 PDT`: `rollout_c` has now started its own V1 engine initialization and model-loading path. `rollout_d` is still waiting behind the sequential startup barrier; `launcher_train_driver.log` is still absent and Harbor has not started creating trials yet.
- `2026-03-19 20:14:36 PDT`: `rollout_c` completed route registration, `Application startup complete`, and passed both local and remote `/health` checks. `launcher_rollout.log` recorded `rollout_c healthy at http://127.0.0.1:18002`.
- `2026-03-19 20:14 PDT`: `rollout_d` has now entered V1 engine initialization. Until it reaches `/health`, `launcher_train_driver.log` remains absent and `trials_run/` remains empty.
- `2026-03-19 20:15:56 PDT`: `rollout_d` reached `/health`; `launcher_rollout.log` now records `rollout_d healthy at http://127.0.0.1:18003` and prints the full external rollout server map.
- `2026-03-19 20:15:56 PDT`: the wrapper crossed the rollout barrier and created `launcher_train_driver.log`. The first driver lines show head-node nofile configured to `131072` and the fully-async monitor starting. Harbor trial creation has still not started at this point, so `trials_run/` remains absent while trainer bring-up begins.
- `2026-03-19 20:16:27 PDT`: driver-side runtime-env export confirms the Harbor fast-path variables survived into Ray workers for this run, including `HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME=/scratch/triton_cache/hkang/harbor-mini-swe-home` and `HARBOR_MINI_SWE_AGENT_UV_OFFLINE=1`.
- `2026-03-19 20:16:48 PDT`: the no-ThunderAgent path is confirmed live. `launcher_train_driver.log` shows `HTTP Inference: Created internal router over external servers`, `HarborGenerator initialized`, and `Total training steps: 1`.
- `2026-03-19 20:17:03 PDT`: trainer bring-up entered process-group initialization (`Initializing process group for RayActorGroup`). `trials_run/` now exists, but still contains `0` trial directories. Trainer GPUs have only single-digit MiB allocated so far, consistent with very early init rather than model-loaded state.
- `2026-03-19 20:17:34 PDT`: the first RayActorGroup process-group initialization completed successfully and printed the 32-rank mesh layout.
- `2026-03-19 20:17:40-20:18:09 PDT`: a second RayActorGroup initialization pass also completed. During this window, some workers auto-corrected `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` from `enp53s0f0np0` to `enp55s0f0np0` on nodes where the original interface name was absent. The run kept moving after the correction.
- `2026-03-19 20:18 PDT`: trainer-side checkpoint loading began in earnest. `infra-260319_201627.log` now contains many `Loading checkpoint shards` progress lines, while trainer GPU memory rose from a few MiB into the `2.5-3.0 GiB` range. This is still far from the fully loaded `~70+ GiB` steady state, so Harbor trial creation remains at `0`.
- `2026-03-19 20:18 PDT`: two recurrent noises are present but not yet fatal: `Failed to establish connection to the metrics exporter agent` from many Ray workers, and monitoring `404` on `/router_state` because this run intentionally disables ThunderAgent and therefore does not expose the TA-specific endpoint.
- `2026-03-19 20:22 PDT`: trainer GPU memory climbed further into the `6.4-7.2 GiB` range across nodes `012-015`, with most GPUs pegged at `100%` utilization. `trials_run/` still contains `0` trial directories, so Harbor generation has still not started.
- `2026-03-19 20:23 PDT`: `infra-260319_201627.log` is still actively growing. Checkpoint-load progress has reached `11/17` on the leading workers, but per-shard time has stretched into the `30-40s` range, much slower than the earlier successful replay.
- `2026-03-19 20:23 PDT`: the current no-ThunderAgent wrapper does not actually disable end-of-run saves. `launcher_train_driver.log` prints `save_final_checkpoint_at_end: true` and `save_final_hf_model_at_end: true`, so `CKPT_INTERVAL=-1` and `HF_SAVE_INTERVAL=-1` are not sufficient on their own for this launcher path. This does not explain the current slow startup, but it will need a wrapper fix before relying on this path for a clean no-save rerun.
- `2026-03-19 20:26 PDT`: trainer checkpoint loading has now reached `17/17` across the visible workers. The slow shard-loading phase appears to have completed, but `trials_run/` is still `0` and the driver has not yet logged `wait_for_generation_buffer` or any Harbor trial creation, so there is at least one more post-load synchronization stage before generation begins.
- `2026-03-19 20:29 PDT`: trainer GPU residency has now climbed into the `17-20 GiB` range, and `gpu_processes.tsv` still shows `ray::FSDPPolicyWorkerBase.init_model` on every trainer GPU. So `17/17` only marks shard reads completing; the trainer remains stuck inside the larger `init_model` path and has not yet returned control to the training loop.
- `2026-03-19 20:29 PDT`: one rank on `research-dev-coder-013` dropped to very low utilization (`~2%`) while sitting at `~20.7 GiB`, which suggests post-load skew between ranks rather than a clean transition into the next training phase.
- `2026-03-19 20:29:35 PDT`: the slow trainer bring-up finally cleared. `launcher_train_driver.log` recorded `init policy/ref/critic models done`.
- `2026-03-19 20:29:36 PDT`: the run entered `init_weight_sync_state`, completed it in `0.81s`, and immediately started `sync_weights_to_inference_engines`.
- `2026-03-19 20:30:32 PDT`: initial weight sync to the inference engines completed in `56.02s`. The driver then started the actual training `step` and entered `wait_for_generation_buffer`.
- `2026-03-19 20:32 PDT`: Harbor generation is now live. `trials_run/` jumped to `256` trial directories, and the driver log contains multiple successful ATIF trajectory conversions under `trials_run/.../agent/trajectory.json`.
- `2026-03-19 20:34 PDT`: live filesystem counts show `256` trial directories, `19` `trajectory.json`, `18` `result.json`, and `0` `exception.txt`. The generation-buffer phase is healthy so far; the main caveat is that `monitoring/trial_progress.tsv` lags the real filesystem state by a couple of minutes.
- `2026-03-19 20:37 PDT`: live counts advanced to `37` `trajectory.json`, `34` `result.json`, and still `0` `exception.txt`. Buffer fill is progressing at a healthy clip.
- `2026-03-19 20:35-20:37 PDT`: even on this no-ThunderAgent path, Harbor is still executing `_best_effort_release_program(...)` and logging `Failed to release ThunderAgent program_id=... after 4 attempts (ReadTimeout)`. These are warnings rather than fatal errors right now, but they confirm the no-TA wrapper still leaves a ThunderAgent-style release path active in cleanup.
- `2026-03-19 20:39 PDT`: live counts reached `43` `trajectory.json`, `43` `result.json`, and `0` `exception.txt`. The generation buffer is still filling, but the pace has slowed relative to the prior two-minute window.
- `2026-03-19 20:37-20:39 PDT`: ThunderAgent-style release warnings became much denser. This has not yet produced trial exceptions, but it is the leading suspect for why buffer growth is no longer as smooth as the initial burst.
- `2026-03-19 20:41 PDT`: live counts reached `48` `trajectory.json`, `48` `result.json`, and still `0` `exception.txt`. The run remains healthy in the sense that useful trajectories are still arriving, but the buffer tail is clearly slower than the first burst.
- `2026-03-19 20:41 PDT`: the driver is still in `wait_for_generation_buffer`; no downstream phase transition has been logged yet. The strongest live symptom remains the dense stream of `_best_effort_release_program(... ReadTimeout)` warnings on the Harbor side.
- `2026-03-19 20:44 PDT`: live counts reached `51` `trajectory.json`, `51` `result.json`, and `0` `exception.txt`. Useful outputs are still arriving, but the tail has slowed further to roughly `+3` trajectories over the last two-minute window.
- `2026-03-19 20:44 PDT`: `wait_for_generation_buffer` is still the active phase. The no-TA wrapper continues to leak time into `_best_effort_release_program(... ReadTimeout)` warnings, which now dominate the live driver log.
- `2026-03-19 20:46 PDT`: live counts reached `56` `trajectory.json`, `56` `result.json`, and still `0` `exception.txt`. The tail remains slow, but not dead; roughly `+5` trajectories arrived over the last two-minute window.
- `2026-03-19 20:48-20:49 PDT`: live counts reached `58` `trajectory.json`, `58` `result.json`, and still `0` `exception.txt`. The driver still has not logged a phase transition out of `wait_for_generation_buffer`, whose last explicit marker remains `2026-03-19 20:30:32 Started: wait_for_generation_buffer`.
- `2026-03-19 20:48-20:49 PDT`: `_best_effort_release_program(... ReadTimeout)` warnings reached `50` total in the driver log. The last ten minutes show repeated bursts at `20:41-20:49`, with especially dense warnings at `20:41`, `20:42`, and `20:47`, making the no-TA cleanup path the main live suspect for the slow generation-buffer tail.
- `2026-03-19 20:55-20:56 PDT`: filesystem counts climbed to `97` then `108` `trajectory.json`, with matching `93` then `103` `result.json`, and still `0` `exception.txt`. However, parsing the latest progress-bar lines from `launcher_train_driver.log` showed the true generation-buffer state was only `2/64`, with `buffer qsize=0`.
- `2026-03-19 20:56 PDT`: grouping the landed trajectories by `task_name` explains the mismatch between raw trajectory count and generation-buffer progress. The `108` trajectories cover only `54` unique tasks; only `3` tasks have reached `4` trajectories, `13` have reached `3`, and `22` remain singletons. This is consistent with the trainer waiting on prompt-group completeness rather than raw trajectory volume.
- `2026-03-19 20:58-21:00 PDT`: generation-buffer fill finally accelerated. Live progress advanced from `3/64` to `10/64` while filesystem counts rose from `127/122` to `154/152` for `trajectory/result`, still with `0` exceptions.
- `2026-03-19 21:02-21:04 PDT`: generation-buffer progress continued to `14/64` and then `15/64`, with filesystem counts reaching `206` `trajectory.json`, `180` `result.json`, and `0` `exception.txt`.
- `2026-03-19 21:04 PDT`: regrouping by `task_name` now fully explains the live `15/64` barrier. All `64` tasks have at least one trajectory, `63` have at least two, `54` have at least three, but only `25` have reached four trajectories; on the stricter `result.json` side, only `15` tasks have reached four finalized attempts.
- `2026-03-19 21:02-21:04 PDT`: the Docker cleanup tail is regressing again, though not yet fatally. `launcher_train_driver.log` now contains repeated `docker compose down --volumes --remove-orphans` warnings ending in `cannot stop container ... tried to kill container, but did not receive an exit event` for environments such as `r2egym-0370`, `r2egym-0851`, and `r2egym-0469`.
