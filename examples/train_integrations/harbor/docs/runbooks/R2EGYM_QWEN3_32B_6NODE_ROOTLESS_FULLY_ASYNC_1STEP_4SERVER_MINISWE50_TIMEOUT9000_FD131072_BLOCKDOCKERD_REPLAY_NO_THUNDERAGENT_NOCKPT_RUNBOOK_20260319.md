# R2EGYM Qwen3-32B 6-Node Rootless Fully Async 1-Step 4-Server mini-swe-agent Replay Runbook (No ThunderAgent, No Checkpoint)

Date: 2026-03-19

This runbook tracks the live rerun launched from the successful `20260319` replay path, but with two explicit deviations required for this experiment:

- do not use ThunderAgent for routing or scheduling
- do not save any training checkpoints or HF exports

## 1. Parent Runbook

This run uses the same 6-node topology, rootless Docker image cache, Harbor task setup fast path, and 1-step R2EGYM split configuration as:

- `/home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/R2EGYM_QWEN3_32B_6NODE_ROOTLESS_FULLY_ASYNC_1STEP_4SERVER_MINISWE50_TIMEOUT9000_FD131072_BLOCKDOCKERD_REPLAY_RUNBOOK_20260319.md`

Only these runtime deltas are intentional:

- `SKYRL_DISABLE_THUNDERAGENT=1`
- `THUNDERAGENT_WATCHDOG_ENABLED=0`
- `CKPT_INTERVAL=-1`
- `HF_SAVE_INTERVAL=-1`

## 2. Canonical Entry Point

Launch with:

```bash
cd /home/hkang/zthunder_agent/SkyRL
export PATH=/home/hkang/zthunder_agent/SkyRL/.venv/bin:$HOME/.local/bin:$PATH

JOB_ID=1731 \
RUN_TS=20260319_192317 \
bash /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/run_r2egym_qwen3_32b_6node_rootless_full_1step_miniswe50_timeout9000_fd131072_blockdockerd_replay_no_thunderagent_nockpt.sh
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

## 4. Live Notes

- `2026-03-19 19:23 PDT`: initialized no-ThunderAgent/no-ckpt replay runbook from the `20260319` replay contract. Verified the active 6-node allocation is `1731` on `003,008,012-015`.
- `2026-03-19 19:23 PDT`: local validation passed for the opt-in launcher changes: shell syntax is clean and `SKYRL_DISABLE_THUNDERAGENT=1` flips the head-pinned entrypoint away from `HarborThunderAgentFullyAsyncExp`.
- `2026-03-19 19:23:53 PDT`: launched `run_r2egym_qwen3_32b_6node_rootless_full_1step_miniswe50_timeout9000_fd131072_blockdockerd_replay_no_thunderagent_nockpt.sh` with `JOB_ID=1731` and `RUN_TS=20260319_192317`.
- `2026-03-19 19:24 PDT`: the launcher printed one head-node `srun ... Exited with exit code 1` line during rootless Docker bring-up, but the daemon log itself continued normally through `Loading containers: done.`, `Daemon has completed initialization`, and `API listen on /tmp/xdg-r2e1srp-0319192353/docker.sock`.
- `2026-03-19 19:24 PDT`: Ray cluster bring-up completed cleanly on `003,012-015`; autoscaler status shows `5` active nodes, `32` GPUs total, and `1.0 harbor_head`.
- `2026-03-19 19:24 PDT`: rollout startup is in progress on `008`. Early `Connection refused` lines in `launcher_rollout.log` are still only pre-health metric scrapes; `launcher_train_driver.log` has not started yet.
- `2026-03-19 19:26 PDT`: direct rollout inspection shows only `rollout_a` has begun so far. It is in normal vLLM model-load, with `Loading safetensors checkpoint shards` progressing through `11/17`. `rollout_b/c/d` logs are still empty, so the wrapper remains blocked on the 4-backend `/health` barrier and the train driver has not started yet.
- `2026-03-19 19:28-19:29 PDT`: `rollout_a` completed application startup and served `/health 200`. `rollout_b` then started and quickly finished shard loading; it moved into the usual compile/finalization phase. `rollout_c/d` still had not started by the end of this window, so the launcher had not yet reached the train-driver step.
- `2026-03-19 19:31-19:32 PDT`: rollout startup stayed sequential as designed. By this point `rollout_a`, `rollout_b`, and `rollout_c` had all reached `/health 200`; `rollout_d` had only just entered `EngineCore_DP0` initialization, so the wrapper was still waiting on the final backend before starting the train driver.
- `2026-03-19 19:33-19:34 PDT`: all four rollout backends reached `/health 200`, so the wrapper advanced into the train driver. The no-ThunderAgent path is confirmed by the driver log line `HTTP Inference: Created internal router over external servers` from `skyrl.train.entrypoints.main_base`, not from any ThunderAgent entrypoint. The log directory also has no `thunderagent.log`.
- `2026-03-19 19:34 PDT`: the actual head-node launch command includes `trainer.ckpt_interval=-1` and `trainer.hf_save_interval=-1`, confirming this rerun will not save training checkpoints or HF exports.
- `2026-03-19 19:36-19:41 PDT`: Harbor has still not started creating trials yet; `trials_run` remains empty with `trial_dirs=0`, `trajectory.json=0`, `result.json=0`, and `exception.txt=0`. The current bottleneck is still trainer bring-up, not Harbor setup or rollout health.
- `2026-03-19 19:40 PDT`: trainer-node inspection on `research-dev-coder-012` shows all 8 GPUs occupied at about `81559 MiB / 81559 MiB` each while utilization remains `0%`. The active processes are still `ray::FSDPPolicyWorkerBase.init_model` and `ray::FSDPRefWorkerBase.init_model`, so the run is still inside model initialization / reference-worker initialization.
- `2026-03-19 19:40-19:41 PDT`: the train-driver log is still only up to repeated `Initialized process group for RayActorGroup` lines, while the infra log continues to advance and shows checkpoint shard loading reaching `17/17`. This indicates progress is still happening in the worker startup path, but the driver has not yet reached `wait_for_generation_buffer`.
- `2026-03-19 19:41 PDT`: monitoring still emits `Failed to scrape http://172.21.44.54:8080/router_state: HTTPError 404` because this run intentionally disabled ThunderAgent; the local internal HTTP router does not expose the TA-specific `/router_state` endpoint. This is monitoring noise, not a generation failure.
- `2026-03-19 19:43 PDT`: a fresh sample still shows `trial_dirs=0`, `trajectory.json=0`, `result.json=0`, and `exception.txt=0`. The train-driver log file keeps changing, but there are still no `sync_weights_to_inference_engines` or `wait_for_generation_buffer` markers.
- `2026-03-19 19:45 PDT`: deeper infra-log inspection shows the worker startup is uneven rather than fully finished. The current log contains `10` occurrences of `Loading checkpoint shards: 100%|██████████| 17/17`, but also live tail lines with workers only at `13/17`, plus smaller counts at `14/17`, `15/17`, and `16/17`. So trainer bring-up is still incomplete across part of the worker set, which explains why Harbor trial generation has not started yet.
- `2026-03-19 19:45 PDT`: cross-node trainer sampling now shows active GPU compute on all four trainer nodes, with most GPUs at `100%` utilization and the per-node `init_model` process count still at `17`. Compared with the successful parent replay, this run is materially slower to exit trainer initialization, but it is still making forward progress inside shard loading rather than sitting in a fully idle post-load stall.
- `2026-03-19 19:47 PDT`: another 2-minute sample still shows `trial_dirs=0` and no `sync_weights_to_inference_engines` or `wait_for_generation_buffer` markers in the train-driver log. However, the infra-log count of `Loading checkpoint shards: 100%|██████████| 17/17` has increased from `10` to `26`, with the log mtime still advancing. The run remains bottlenecked in trainer initialization, but the shard-loading side is still moving.
