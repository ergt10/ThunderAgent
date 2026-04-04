# R2EGYM Qwen3-32B 6-Node Rootless Fully Async 1-Step 4-Server mini-swe-agent Cross-Job Replay Runbook

Date: 2026-03-21

This runbook tracks the live cross-job replay derived from:

- `/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/docs/runbooks/R2EGYM_QWEN3_32B_6NODE_1STEP_TA_RUNTIME_FULL_HANDOFF.md`
- `/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/docs/runbooks/R2EGYM_QWEN3_32B_6NODE_ROOTLESS_FULLY_ASYNC_1STEP_4SERVER_MINISWE50_TIMEOUT9000_FD131072_BLOCKDOCKERD_REPLAY_RUNBOOK_20260319.md`

The main difference from the historical 6-node replay is operational, not logical:

- nodes are spread across multiple Slurm jobs
- `research-secure-06` is the dedicated head/rootless-Docker node
- `research-secure-17` is the dedicated rollout node
- trainers are `research-secure-18`, `research-secure-21`, `research-secure-30`, `research-secure-01`
- the original 6-node wrappers are left untouched
- launch is driven by a new cross-job wrapper that preserves the documented `1 head + 1 rollout + 4 trainer` topology

## 1. Canonical Entry Point

Launch from the isolated execution worktree:

```bash
cd /data/zy/models/hkang/run_worktrees/skyrl-r2e-clean-20260321

MERGED_NODE=research-secure-06 \
MERGED_JOB_ID=28860 \
ROLLOUT_NODE=research-secure-17 \
ROLLOUT_JOB_ID=28911 \
TRAINER_NODE_SPECS='research-secure-18:28860,research-secure-21:28860,research-secure-30:28882,research-secure-01:28837' \
PREPULL_R2EGYM_IMAGES=false \
HEAD_RAY_TMP_DIR=/scratch/$USER/raytmp-head \
TRAINER_RAY_TMP_DIR_ROOT=/scratch/$USER/raytmp \
bash examples/train_integrations/harbor/run_r2egym_qwen3_32b_5node_rootless_full_1step_miniswe50_timeout9000_fd131072_cross_job_merged_head_rollout.sh
```

Even though the filename still says `5node`, this wrapper was extended to support:

- dedicated rollout node via `ROLLOUT_NODE` and `ROLLOUT_JOB_ID`
- 4 trainer nodes across multiple jobs
- rootless runtime-state cleanup without deleting the shared image cache
- short Ray temp dirs to avoid AF_UNIX path length failures

## 2. Fixed Runtime Inputs

- head node: `research-secure-06`
- rollout node: `research-secure-17`
- trainer nodes:
  - `research-secure-18`
  - `research-secure-21`
  - `research-secure-30`
  - `research-secure-01`
- Slurm jobs:
  - `28860` -> `research-secure-[06,18,21]`
  - `28882` -> `research-secure-30`
  - `28837` -> `research-secure-01`
  - `28911` -> `research-secure-17`
- model path: `/data/zy/models/hkang/models/Qwen3-32B`
- datasets:
  - `/home/hkang/zthunder_yagent/data/harbor/r2egym-trivial`
  - `/home/hkang/zthunder_yagent/data/harbor/r2egym-easy`
  - `/home/hkang/zthunder_yagent/data/harbor/r2egym-medium`
  - `/home/hkang/zthunder_yagent/data/harbor/r2egym-hard`
- rootless Docker cache reuse:
  - `DOCKER_DATA_ROOT=/scratch/triton_cache/$USER/r2egym-rootless-image-cache`
- Ray temp dirs:
  - head: `/scratch/$USER/raytmp-head`
  - workers: `/scratch/$USER/raytmp`

## 3. Local Wrapper / Workspace Notes

- the user explicitly requested that entire Slurm jobs must never be canceled
- all cleanup in this replay is limited to:
  - local wrapper processes
  - individual `srun` child steps
  - targeted `ray stop -f`
  - targeted rootless Docker pid cleanup on the head node
- execution is isolated in:
  - `/data/zy/models/hkang/run_worktrees/skyrl-r2e-clean-20260321`
- code/docs are still maintained in:
  - `/home/hkang/zthunder_yagent/SkyRL`

## 4. Live Notes

- `2026-03-21 20:45 PDT` to `21:00 PDT`: determined that the original documented topology must stay logically equivalent to `1 head + 1 rollout + 4 trainer`, but cross-job scheduling is acceptable. Decided to preserve the original 6-node wrappers and create a separate cross-job launcher.
- `2026-03-21 20:45 PDT`: found that `research-secure-06` could not start rootless Docker because `/etc/subuid` and `/etc/subgid` for `hkang` contained an overlapping subordinate range covering the real UID/GID.
- `2026-03-21 20:45 PDT`: repaired only the `hkang` entries on `research-secure-06` to match the healthy nodes:
  - `/etc/subuid` -> `hkang:100000:1048576`
  - `/etc/subgid` -> `hkang:100000:1048576`
- `2026-03-21 20:45 PDT`: backups created on `research-secure-06`:
  - `/etc/subuid.bak.hkang_r2e_20260321_204532`
  - `/etc/subgid.bak.hkang_r2e_20260321_204532`
- `2026-03-21 21:07 PDT`: a long image pre-pull was allowed to continue on `research-secure-06` against the shared rootless image cache. This filled the cache substantially and made a second full repull unnecessary.
- `2026-03-21 22:23 PDT`: the first split-topology relaunch failed before training because Ray head temp paths under `/scratch/triton_cache/$USER/...` exceeded AF_UNIX socket path limits.
- `2026-03-21 22:28 PDT`: patched the cross-job wrapper to force short Ray temp roots, but the next relaunch still failed because worker nodes inherited the head temp root and one worker could not create `/scratch/triton_cache/...`.
- `2026-03-21 22:30 PDT`: corrected the head Ray temp root to `/scratch/$USER/raytmp-head`, keeping worker roots under `/scratch/$USER/raytmp`.
- `2026-03-21 22:34 PDT`: discovered two stale long-lived steps inside job `28860`:
  - `28860.108` on `research-secure-18`
  - `28860.109` on `research-secure-21`
  These steps blocked new trainer `srun` creation and were removed without canceling the parent job.
- `2026-03-21 22:39 PDT`: relaunched the split topology with run name:
  - `r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-crossjob-split-20260321_223910`
- `2026-03-21 22:39 PDT`: rootless Docker on `research-secure-06` came up successfully and reused the shared image cache.
- `2026-03-21 22:40 PDT`: Ray head and all 4 trainer workers came up successfully.
- `2026-03-21 22:40 PDT`: early `Connection refused` lines in `launcher_rollout.log` matched the historical pattern from the 2026-03-19 replay runbook: they were initially indistinguishable from normal pre-health rollout startup noise.
- `2026-03-21 22:42 PDT`: direct rollout log inspection showed the first actual failure was not health-barrier delay. `rollout_a.log` failed during Qwen3 model initialization with:
  - `ModuleNotFoundError: No module named 'flash_attn_2_cuda'`
  - downstream `RuntimeError: Engine core initialization failed`
- `2026-03-21 22:42 PDT`: confirmed the isolated `.venv` contained the Python package `flash_attn` but not the compiled extension module `flash_attn_2_cuda`.
- `2026-03-21 22:47 PDT`: stopped only the failed cross-job run processes and node-local child steps; did not cancel any parent Slurm job.
- `2026-03-21 22:48 PDT`: installed `pip` into the isolated `.venv` via `python -m ensurepip --upgrade` so a direct wheel install could be used instead of `uv pip` resolution.
- `2026-03-21 22:49 PDT`: replaced the broken `flash-attn` install with the official wheel matching the live environment:
  - `https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`
- `2026-03-21 22:49 PDT`: post-install validation succeeded when importing in the same order the runtime effectively needs:
  - `import torch`
  - `import flash_attn_2_cuda`
- `2026-03-21 22:50 PDT`: rollout CUDA runtime smoke is being rerun against `research-secure-17` before the next full relaunch.
- `2026-03-21 22:51 PDT`: rollout CUDA runtime smoke passed on `research-secure-17`. The smoke log reported:
  - `torch 2.9.1+cu128`
  - `CUDA_RUNTIME_SMOKE_OK`
  - all 4 spawn workers returned `fa3_supported=true`
- `2026-03-21 22:52 PDT`: launched the next full split-topology attempt with run name:
  - `r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-crossjob-split-20260321_225208`
- `2026-03-21 22:54 PDT`: rootless Docker on `research-secure-06` and the full Ray cluster came up cleanly again.
- `2026-03-21 22:55 PDT`: rollout startup matched the historical healthy pattern from the 2026-03-19 replay notes: early `Connection refused` lines were still just pre-health metric scrapes.
- `2026-03-21 22:57 PDT`: `rollout_a` progressed past the prior failure point. It now reports:
  - `Using FLASH_ATTN attention backend`
  - `Loading safetensors checkpoint shards`
  This confirms the `flash_attn_2_cuda` fix is active in the real rollout path, not just in the standalone smoke test.
- `2026-03-21 23:01 PDT`: `rollout_a` continued normal model-load progress and reached `14/17` safetensors shards without emitting any new traceback or module-import failure. `rollout_b/c/d` were still empty, so the wrapper remained blocked on the rollout health barrier.
- `2026-03-21 23:03 PDT`: the split run still failed before training. `rollout_a` finished model load and CUDA graph capture, then crashed in `skyrl/backends/skyrl_train/inference_engines/vllm/vllm_server.py` with:
  - `RuntimeError: Cannot add middleware after an application has started`
  This was a vLLM server startup-order incompatibility on the active `vllm==0.16.0` path, not a GPU runtime or model-load failure.
- `2026-03-21 23:07 PDT` to `23:16 PDT`: repinned the isolated replay `.venv` back to the handoff/replay baseline stack used by this experiment shape:
  - `torch 2.8.0+cu128`
  - `torchaudio 2.8.0`
  - `torchvision 0.23.0`
  - `xformers 0.0.32.post1`
  - `vllm 0.10.2`
- `2026-03-21 23:17 PDT`: replaced the previously installed `torch2.9` `flash-attn` wheel with the matching `torch2.8` official wheel:
  - `https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`
- `2026-03-21 23:18 PDT`: post-repin validation succeeded in the isolated replay environment:
  - `import torch`
  - `import vllm`
  - `import xformers`
  - `import flash_attn_2_cuda`
  all completed successfully under:
  - `torch 2.8.0+cu128`
  - `vllm 0.10.2`
- `2026-03-21 23:18 PDT`: rollout CUDA runtime smoke passed again on `research-secure-17` after the environment repin. The smoke log reported:
  - `torch 2.8.0+cu128`
  - `CUDA_RUNTIME_SMOKE_OK`
  - all 4 spawn workers returned `fa3_supported=true`
- `2026-03-21 23:20 PDT`: cleaned only the stale child steps from the failed `22:54 PDT` split run on `research-secure-06,18,21,30,01`; parent jobs `28860`, `28882`, `28837`, and `28911` were left running.
- `2026-03-21 23:21 PDT`: launched a fresh split-topology attempt with run name:
  - `r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-crossjob-split-vllm0102-20260321_232155`
- `2026-03-21 23:24 PDT`: rootless runtime-state cleanup ran again on `research-secure-06` against the shared Docker cache:
  - previous runtime state was already mostly empty
  - `rootless_runtime_state_cleanup=APPLIED`
  - backup root: `/scratch/triton_cache/hkang/r2egym-rootless-state-backups/20260321_232254`
- `2026-03-21 23:24:56 PDT`: rootless Docker completed initialization on `research-secure-06` and exposed `/run/user/243001624/docker.sock`.
- `2026-03-21 23:25 PDT`: Ray head and all 4 trainer workers came up again on the cross-job topology.
- `2026-03-21 23:25 PDT`: rollout monitor started on `research-secure-17`; early `Connection refused` lines remained limited to the expected pre-health scrape phase.
- `2026-03-21 23:27 PDT`: `rollout_a` on the repinned `vllm==0.10.2` path progressed through V1 engine bring-up:
  - `Initializing a V1 LLM engine (v0.10.2)`
  - NCCL and TP rank assignment succeeded
  - `Worker_TP0/1` were created via `spawn`
- `2026-03-21 23:28 PDT`: `rollout_a` moved into real model loading:
  - `Starting to load model`
  - `Loading model from scratch`
  - `Using Flash Attention backend on V1 engine`
  - `Loading safetensors checkpoint shards: 2/17`
  This confirms the `vllm==0.10.2` relaunch has already passed the earlier `Cannot add middleware after an application has started` failure point.
- `2026-03-21 23:33:54 PDT`: `rollout_a` fully completed startup on the repinned stack:
  - server routes were registered
  - `Application startup complete`
  - local `/health` returned `200 OK`
  - follow-on `/metrics` scrapes also returned `200 OK`
- `2026-03-21 23:34 PDT`: after `rollout_a` became healthy, the wrapper advanced to `rollout_b`. Early `Connection refused` lines for `18001-18003` remained expected pre-health probes, not a crash signal.
- `2026-03-21 23:36 PDT`: `rollout_b` reached the same real bring-up path as `rollout_a`:
  - `Initializing a V1 LLM engine (v0.10.2)`
  - TP/NCCL init succeeded
  - `Using FlashInfer for top-p & top-k sampling`
  - `Starting to load model`
  - `Using Flash Attention backend on V1 engine`
- `2026-03-21 23:39 PDT`: `rollout_b` is still steadily loading shards and has reached at least `12/17` without any traceback, middleware error, OOM, or NCCL failure. `rollout_c` and `rollout_d` have not started yet because the wrapper is still waiting for `rollout_b` to pass `/health`.
- `2026-03-21 23:39 PDT`: Harbor runtime patch validation was rechecked in the isolated replay environment and all three required site-packages patches are still present:
  - `harbor/environments/docker/docker.py`
  - `harbor/agents/installed/install-mini-swe-agent.sh.j2`
  - `harbor/environments/docker/docker-compose-base.yaml`
- `2026-03-21 23:39 PDT`: fast-path prerequisites are still aligned with the 2026-03-19 replay fixes:
  - repo code still preserves `HARBOR_SHARED_*` and `HARBOR_MINI_SWE_AGENT_*` through Ray runtime-env preparation
  - the shared `mini-swe-agent` tool home on the head node was refreshed successfully under `/scratch/triton_cache/hkang/harbor-mini-swe-home`
  - no Harbor trial container exists yet in this run, so there is not yet any live `USING_SHARED_PREINSTALLED_MINI_SWE_AGENT=...` evidence or fallback evidence (`apt-get`, `uv tool install`) to inspect

## 5. Current Status

At the time of this note:

- head/rootless Docker path on `research-secure-06` is working
- cross-job Ray cluster launch path is working
- the current live `vllm==0.10.2` relaunch has:
  - rootless Docker up on `research-secure-06`
  - Ray cluster up across `06,18,21,30,01`
  - `rollout_a` fully healthy on `18000`
  - `rollout_b` actively loading Qwen3-32B shards on `18001`
  - `rollout_c` and `rollout_d` not started yet
- the `vllm==0.16.0` startup-order failure is no longer the active blocker; the current gating item is the normal sequential rollout health barrier
- Harbor fast-path has not been exercised yet in this run because:
  - `launcher_train_driver.log` has not appeared yet
  - no `trials_run/` content exists yet
  - there is no live trial `install.sh` / setup stdout to inspect
- next step is:
  - wait for `rollout_b/c/d` to start in sequence and pass `/health`
  - confirm the wrapper clears the rollout barrier and starts `launcher_train_driver.log`
  - keep monitoring `launcher_rootless_docker.log`, `launcher_ray.log`, `launcher_rollout.log`, `launcher_train_driver.log`, `thunderagent.log`, and `monitoring/trial_progress.tsv`
