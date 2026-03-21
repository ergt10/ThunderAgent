# QWEN3 32B 6-Node Current Cluster Error Ledger

Date: 2026-03-13
Job: `1135`
Topology:
- `research-dev-coder-003`: Harbor head + Docker + Ray head
- `research-dev-coder-012` to `research-dev-coder-015`: trainer nodes
- `research-dev-coder-016`: rollout node

This file tracks each distinct error encountered while migrating the historical Harbor experiment to the current cluster. New unique failures should be appended here before more retries are attempted.

## Error 1: Ray head `harbor_head` resource JSON was malformed

Signature:
- `ERR utils.py:1163 -- --resources={harbor_head: 1} is not a valid JSON string`
- Ray head appeared to start, but the GCS port never came up for workers.

Root cause:
- The `--resources` argument in `launch_qwen3_32b_ray_cluster.sh` was assembled through nested `bash -lc` quoting and lost the JSON quotes around `harbor_head`.

Fix:
- Build the JSON string with Python (`json.dumps`) and pass it as `--resources "$HARBOR_HEAD_RESOURCES"`.

Patched file:
- `examples/train_integrations/harbor/launch_qwen3_32b_ray_cluster.sh`

Status:
- Fixed.

## Error 2: Ray `detached` mode was invalid under Slurm for readiness

Signature:
- Ray head and workers reported successful startup, then workers failed with:
- `Failed to connect to GCS at address 172.21.44.54:6381`
- Head port was no longer listening after the short `srun` step exited.

Root cause:
- In `detached` mode, the `srun` step finished and Slurm cleaned up the Ray daemons in that step's cgroup.

Fix:
- Readiness now uses `READINESS_RAY_START_MODE=block` by default so the service-bearing `srun` steps stay alive during validation.

Patched file:
- `examples/train_integrations/harbor/run_qwen3_32b_full_readiness_suite.sh`

Status:
- Fixed.

## Error 3: CPU-only validation probes inherited GPU requirements

Signature:
- During cluster validation, simple checks failed with:
- `srun: error: Unable to create step for job 1135: Requested nodes are busy`
- This happened while rollout and Ray worker steps were already consuming node GPUs.

Root cause:
- Helper `run_on_node` calls used `srun` without explicitly setting `--gres=gpu:0`, so Slurm treated the probes as GPU-consuming steps.

Fix:
- All CPU-only helper probes now use `--gres=gpu:0`.

Patched files:
- `examples/train_integrations/harbor/run_qwen3_32b_full_readiness_suite.sh`
- `examples/train_integrations/harbor/run_full_run_validation_suite.sh`

Status:
- Fixed.

## Error 4: `vllm_server.py` used stale imports for `vllm==0.10.2`

Signature:
- Rollout logs on `016` showed:
- `ModuleNotFoundError: No module named 'vllm.utils.argparse_utils'`
- `ModuleNotFoundError: No module named 'vllm.utils.system_utils'`

Root cause:
- The server entrypoint assumed an older vLLM module layout than the one installed in the recreated `.venv` (`vllm==0.10.2`).

Fix:
- Added compatibility fallbacks:
  - `FlexibleArgumentParser` from `vllm.utils` when `vllm.utils.argparse_utils` is absent.
  - `set_ulimit` from `vllm.utils` when `vllm.utils.system_utils` is absent.

Patched file:
- `skyrl/backends/skyrl_train/inference_engines/vllm/vllm_server.py`

Status:
- Fixed.

## Error 5: Ray worker port collisions on trainer nodes

Signature:
- Ray worker startup on `012` failed with:
- `ValueError: Ray component worker_ports is trying to use a port number 16521 that is used by other components.`

Root cause:
- Ray was allowed to choose dashboard-agent and runtime-env ports randomly, and those could land inside the worker port range.

Fix:
- Explicitly pinned:
  - `--dashboard-agent-listen-port`
  - `--dashboard-agent-grpc-port`
  - `--runtime-env-agent-port`
  - `--min-worker-port`
  - `--max-worker-port`

Patched file:
- `examples/train_integrations/harbor/launch_qwen3_32b_ray_cluster.sh`

Status:
- Fixed.

## Error 6: Rollout vLLM defaulted to `fork` and hit CUDA init failure

Signature:
- Rollout logs on `016` showed repeated:
- `Failed to get device capability ... Error 802: system not yet initialized`
- Rollout ports `18000` and `18001` never became healthy.

Root cause:
- In the current `vllm==0.10.2`, worker multiprocessing defaults to `fork`.
- On this rollout path, that conflicted with CUDA initialization order during engine startup.

Fix:
- Export `VLLM_WORKER_MULTIPROC_METHOD=spawn` by default in the external rollout server launcher.

Patched file:
- `examples/train_integrations/harbor/start_qwen3_32b_external_rollout_servers.sh`

Status:
- Fixed in script, pending live verification.

## Error 7: Readiness validated rollout before rollout became healthy

Signature:
- Readiness reported rollout port failures on `18000` and `18001` immediately after starting the rollout script.
- Rollout logs showed model loading still in progress rather than a hard startup crash.

Root cause:
- `run_qwen3_32b_full_readiness_suite.sh` started rollout in the background and only did `sleep 5` before entering cluster validation.
- The historical experiment docs explicitly require waiting for both `/health` endpoints before proceeding.

Fix:
- Replace the fixed sleep with an actual health-wait loop that polls `http://127.0.0.1:18000/health` and `http://127.0.0.1:18001/health` on the rollout node.

Patched file:
- `examples/train_integrations/harbor/run_qwen3_32b_full_readiness_suite.sh`

Status:
- Fixed in script, pending live verification.

## Error 8: Concurrent rollout bring-up on one 8-GPU node remained unstable

Signature:
- Both rollout logs entered model initialization, then never reached healthy ports.
- Both logs later emitted repeated `cudaGetDeviceCount ... Error 802: system not yet initialized`.

Root cause:
- The launcher started both TP=4 rollout servers concurrently on the same 8-GPU node.
- On the current cluster/runtime combination, concurrent bring-up appears to race CUDA initialization across the two server groups.

Fix:
- Serialize rollout startup: wait for `rollout_a` health before starting `rollout_b`.
- Extend cleanup to kill rollout server PIDs if health never comes up.

Patched file:
- `examples/train_integrations/harbor/start_qwen3_32b_external_rollout_servers.sh`

Status:
- Mitigation kept, but not the root cause.
- A later single-server `rollout_a` smoke on `016` still failed with the same `cudaGetDeviceCount ... Error 802`, so serialization alone does not resolve rollout startup on this cluster.

## Error 9: External rollout lacked the training path's runtime stability env defaults

Signature:
- External rollout continued to fail in vLLM engine startup with `cudaGetDeviceCount ... Error 802`.
- The main training runtime already injects a broader set of vLLM/NCCL stabilizing env vars that the external rollout launcher was not exporting.

Root cause:
- `start_qwen3_32b_external_rollout_servers.sh` did not mirror the runtime env used by the stable training path.

Fix:
- Export conservative defaults in the external rollout launcher:
  - `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true`
  - `VLLM_ALLOW_INSECURE_SERIALIZATION=1`
  - `VLLM_USE_V1=1`
  - `VLLM_ENABLE_V1_MULTIPROCESSING=0`
  - `NCCL_CUMEM_ENABLE=0`
  - `NCCL_P2P_DISABLE=1`
  - `NCCL_SHM_DISABLE=1`
  - `OMP_NUM_THREADS=1`

Patched file:
- `examples/train_integrations/harbor/start_qwen3_32b_external_rollout_servers.sh`

Status:
- Fixed in script, pending live verification.

## Known environmental blocker: rootless Docker is not currently available for `hkang`

Signature:
- Rootless installer failed with:
- `Could not find records for the current user hkang from /etc/subuid`

Root cause:
- The cluster image does not currently provide subordinate UID/GID mappings for `hkang` in `/etc/subuid` and `/etc/subgid`.

Current workaround:
- Preflight and readiness are being validated with `DOCKER_MODE=rootful` on `003`.

Status:
- Not fixed in user space.

## Error 10: Slurm step residue made later probes fail with `Requested nodes are busy`

Signature:
- New validation attempts failed immediately with:
- `srun: error: Unable to create step for job 1135: Requested nodes are busy`
- `scontrol show step 1135` showed a leftover step:
- `StepId=1135.617 ... State=RUNNING ... NodeList=research-dev-coder-016 ... gres/gpu=8`

Root cause:
- Earlier smoke tests left a service-bearing `srun` step alive on `016`, so later probes could not allocate that node.
- This was compounded by opening too many short-lived exec sessions instead of reusing one session and explicitly checking Slurm step state before retrying.

Fix:
- Before any new rollout or validation retry, check `scontrol show step 1135`.
- If a non-batch step is still running and it belongs to the previous probe, terminate that probe's process rather than starting another overlapping attempt.
- Avoid delayed polling shells; prefer one live session and direct step-state inspection.

Operational rule:
- Treat `Requested nodes are busy` as a step-residue failure first, not as a new rollout/runtime failure.
- A retry is invalid until `scontrol show step 1135` shows only `1135.batch`.

Status:
- Fixed operationally for the current retry path. Must be enforced on every subsequent probe.

## Error 11: Native `vllm` TP=4 on `016` fails before health, independent of SkyRL rollout

Signature:
- A direct `vllm.entrypoints.openai.api_server` smoke on `research-dev-coder-016` timed out without ever serving `/health`.
- The log showed repeated:
- `Failed to get device capability ... Error 802: system not yet initialized`
- The first concrete worker traceback was:
- `vllm.v1.attention.backends.flash_attn -> get_flash_attn_version -> torch.cuda.get_device_capability -> torch._C._cuda_init()`

Root cause:
- This failure reproduces without Harbor, Ray, or the SkyRL external rollout wrapper.
- The current remaining blocker is therefore in native `vllm` worker startup on `016`, specifically during CUDA initialization in the worker process while probing FlashAttention support.

Evidence:
- Smoke log: `/home/hkang/zthunder_agent/tmp_logs/pure_vllm_tp4_smoke_1773394234/server.log`

Fix:
- Not fixed yet.
- Next diagnostic split should be:
  - plain `torch.cuda` single-process sanity on `016`
  - plain Python multiprocessing `spawn` plus `torch.cuda.get_device_capability`
  - then, if those pass, a `vllm` backend/worker-path workaround

## Error 28: Harbor `no-network` fallback diverged from `r12` and broke tmux bootstrap

Signature:
- Training reached `Training Step Progress: 0/10` and entered `wait_for_generation_buffer`.
- Harbor then logged repeated:
  - `Installing tmux from source...`
  - `Failed to install tmux from source`
- Trial failures were masked as:
  - `AttributeError: 'NoneType' object has no attribute 'strip'`
  - from `harbor/agents/terminus_2/tmux_session.py:372`

Root cause:
- A local mitigation introduced `HARBOR_DOCKER_DISABLE_PROJECT_NETWORK=1`.
- That switched Harbor trial containers to:
  - `harbor/environments/docker/docker-compose-no-network.yaml`
  - `network_mode: none`
- This diverged from the documented `r12` path.
- With networking disabled, the trial container could not fetch or install tmux during Terminus-2 setup.
- Harbor then masked the real setup failure with a bad `stderr.strip()` call.

Why this was a repeatable operator mistake:
- The historical reproduction note `QWEN3_32B_6NODE_ROOTLESS_FULLY_ASYNC_R12_20STEP_REPRO.md` existed and should have been treated as the primary contract.
- The mitigation was introduced before re-checking the live path against that note.

Fix:
- Revert the 10-step wrapper default so `HARBOR_DOCKER_DISABLE_PROJECT_NETWORK` is off unless explicitly requested.
- Treat the historical run note as the first comparison point before adding any new Harbor/Docker fallback.

Patched files:
- `examples/train_integrations/harbor/run_qwen3_32b_full_10step_capture.sh`
- `.codex/skills/harbor-full-run-preflight/SKILL.md`

Status:
- Fixed in launcher defaults and workflow guidance.
- Any follow-up retry must start from the documented path, not the `no-network` fallback.

Operational rule:
- Do not spend more time changing Harbor or rollout wrapper code until this native `vllm` failure on `016` is either fixed or explicitly worked around.

Status:
- Open.

## Error 12: `research-dev-coder-016` fails plain single-process `torch.cuda` initialization

Signature:
- Reusable smoke script `check_rollout_cuda_runtime_smoke.sh` failed on `016`.
- Logged:
- `torch.cuda.device_count() == 4`
- `torch.cuda.is_available() == false`
- `torch.cuda.get_device_name()` / `torch.cuda.get_device_capability()` then failed with:
- `RuntimeError: Unexpected error from cudaGetDeviceCount() ... Error 802: system not yet initialized`

Root cause:
- The rollout node failure is below `vllm`.
- On `016`, even plain single-process `torch.cuda` initialization is not clean under the current `srun --gres=gpu:4` runtime path.

Evidence:
- Smoke log: `/home/hkang/zthunder_agent/tmp_logs/rollout-cuda-runtime-smoke-1773394611/rollout_cuda_runtime_smoke.log`

Fix:
- Not fixed yet.
- Next split should compare the same smoke on a known-good trainer node to determine whether this is specific to `016` or to the recreated Python/CUDA stack on all nodes.

Operational rule:
- Do not attempt full-run rollout startup on `016` until this smoke passes.
- Treat any later rollout error on `016` as downstream noise unless this check is green first.

Status:
- Open.

## Error 13: The new CUDA multiprocessing smoke initially used stdin, which breaks Python `spawn`

Signature:
- The first control run on `012` showed healthy single-process CUDA, then all spawned workers failed with:
- `FileNotFoundError: [Errno 2] No such file or directory: '/home/hkang/zthunder_agent/SkyRL/<stdin>'`

Root cause:
- The diagnostic script launched Python via `python - <<'PY' ...`.
- Python multiprocessing `spawn` requires a real `__main__` file path; it cannot re-import code that came from stdin.

Fix:
- The smoke script now writes the diagnostic program to a temporary file on the target node and runs that file, so spawned workers can import `__main__` correctly.

Patched file:
- `examples/train_integrations/harbor/check_rollout_cuda_runtime_smoke.sh`

Status:
- Fixed.

## Error 14: The CUDA smoke script briefly masked failures due to incorrect shell exit-code capture

Signature:
- The diagnostic output clearly showed `srun` task exit code `1` and a failed smoke log, but the outer script still exited successfully.

Root cause:
- The script used `if ! run_smoke; then smoke_rc=$?; fi`.
- Inside that branch, `$?` was the status of the negated test, not the original `run_smoke` failure code.

Fix:
- Replace the negated form with:
- `if run_smoke; then smoke_rc=0; else smoke_rc=$?; fi`

Patched file:
- `examples/train_integrations/harbor/check_rollout_cuda_runtime_smoke.sh`

Status:
- Fixed.

## Error 15: `srun` task failure output was not sufficient as a pass/fail signal for the diagnostic wrapper

Signature:
- The wrapper printed `srun: error: ... task 0: Exited with exit code 1` and a failing smoke log, yet the outer shell still surfaced `RC=0`.

Root cause:
- Relying on the outer command status alone was not robust enough for this wrapper path.
- The diagnostic needed its own explicit success condition instead of inferring success from the transport layer.

Fix:
- The Python smoke now prints a success sentinel line: `CUDA_RUNTIME_SMOKE_OK`.
- The shell wrapper only returns success if that sentinel is present in the log; otherwise it returns failure even if the outer `srun` wrapper looks successful.

Patched file:
- `examples/train_integrations/harbor/check_rollout_cuda_runtime_smoke.sh`

Status:
- Fixed.

## Error 16: The wrapper needed a final log-level gate, not just internal status plumbing

Signature:
- Even after adding the success sentinel inside the smoke body, the wrapper path still surfaced `RC=0` in outer verification.

Root cause:
- The most reliable source of truth for this diagnostic is the emitted log, not intermediate shell status propagation.

Fix:
- Add a final gate at the end of the wrapper:
- if the log does not contain an exact `CUDA_RUNTIME_SMOKE_OK` line, the script exits nonzero.

Patched file:
- `examples/train_integrations/harbor/check_rollout_cuda_runtime_smoke.sh`

Status:
- Fixed.

## Error 17: `research-dev-coder-016` has failed `nvidia-fabricmanager`, matching the CUDA 802 rollout failure

Signature:
- Node comparison under the same `srun --gres=gpu:4` path showed:
- `research-dev-coder-012`: `systemctl is-active nvidia-fabricmanager` -> `active`
- `research-dev-coder-016`: `systemctl is-active nvidia-fabricmanager` -> `failed`
- On `016`, `systemctl status nvidia-fabricmanager` reports:
- `Active: failed (Result: exit-code) since Sun 2026-03-01 23:55:36 PST`

Root cause:
- The currently best-supported explanation for `016` is a node-level NVIDIA fabric manager failure.
- This matches the observed symptom pattern:
  - `nvidia-smi` still works
  - device nodes are present
  - but CUDA runtime initialization inside PyTorch/vLLM fails with `cudaGetDeviceCount Error 802: system not yet initialized`
- On H100/NVSwitch nodes, that is consistent with a broken fabric-manager path.

Evidence:
- Node diff collected on 2026-03-13 during `012` vs `016` comparison.

Fix:
- Not fixable in user space.
- `016` needs node-level repair or replacement before it can be trusted as rollout.

Operational rule:
- Treat `016` as unhealthy for CUDA runtime workloads until `nvidia-fabricmanager` is back to `active` and the reusable CUDA smoke passes.

Status:
- Open.

## Error 18: `research-dev-coder-006` is not a viable rollout/training node in the current state

Signature:
- `check_rollout_cuda_runtime_smoke.sh` on `006` failed with:
- `torch.cuda.is_available() == false`
- `torch.cuda.device_count() == 0`
- `Can't initialize NVML`
- Node-side quick probe also showed:
- `nvidia-smi -L` emitted `Unable to determine the device handle for gpu 0000:3F:00.0: Unknown Error`

Root cause:
- `006` has a node-level GPU runtime problem distinct from `016`.
- Fabric Manager is running, but at least one GPU device handle is already broken from `nvidia-smi`, and CUDA runtime initialization collapses to zero visible devices for PyTorch.

Fix:
- Not fixable in user space.
- Do not use `006` in the training allocation.

Operational rule:
- Exclude `006` from this experiment until node health is repaired.

Status:
- Open.

## Error 19: Full readiness initially deadlocked on its own rollout CUDA smoke ordering

Signature:
- During full readiness on `1139`, rollout servers were started on `008` first.
- Cluster validation then attempted to run `check_rollout_cuda_runtime_smoke.sh`, which needs `--gres=gpu:4`, and failed with:
- `srun: error: Unable to create step for job 1139: Requested nodes are busy`

Root cause:
- The readiness orchestration started the rollout services before the reusable rollout CUDA smoke.
- Once both rollout servers were up, the rollout node had no free GPUs left for the diagnostic step.

Fix:
- Run the reusable rollout CUDA runtime smoke earlier in `run_qwen3_32b_full_readiness_suite.sh`, before rollout server startup.
- When full readiness later calls `run_full_run_validation_suite.sh`, pass `SKIP_ROLLOUT_CUDA_RUNTIME_SMOKE=true` so validation does not redundantly re-run the same GPU-consuming smoke after rollout is already live.

Patched file:
- `examples/train_integrations/harbor/run_qwen3_32b_full_readiness_suite.sh`

Status:
- Fixed.

## Error 20: Harbor patch checker only recognized the old local upload patch and falsely failed after the new remote Harbor patch landed

Signature:
- Full readiness failed at `Harbor rootless patch` even though the updated `.venv` Harbor smoke already passed on `003`.
- The checker raised:
- `RuntimeError: Unexpected Harbor docker.py imports block; aborting patch.`

Root cause:
- `apply_harbor_rootless_patch.py` only knew how to detect the original local patch shape:
  - import of `io` and `tarfile`
  - `_build_tar_from_file/_build_tar_from_dir`
  - the old `PATCH_MARKER`
- The new remote Harbor patch rewrote `docker.py` differently:
  - `_stream_tar_to_container`
  - inline `docker compose` command construction
  - tar streaming with `--owner=0 --group=0 --numeric-owner`
- Harbor itself was fine; only the checker was stale and produced a false negative.

Fix:
- Teach `apply_harbor_rootless_patch.py` to treat either patch shape as already patched.
- `is_patch_present()` now returns true for:
  - the original local patch marker
  - or the new remote Harbor patch markers in `docker.py`

Patched file:
- `examples/train_integrations/harbor/apply_harbor_rootless_patch.py`

Status:
- Fixed.

## Error 21: `Rollout startup` could report `PASS` even when the rollout launcher had already exited

Signature:
- Full readiness printed:
- `srun: error: research-dev-coder-008: task 0: Exited with exit code 7`
- but still followed with:
- `Rollout servers are healthy on research-dev-coder-008`
- `PASS: Rollout startup`
- Later cluster validation then found both rollout ports closed.

Root cause:
- The readiness wrapper only waited for `/health` on the two rollout ports.
- It did not also require the background rollout launcher process to still be alive.
- That allowed a transient or stale health check to mask a launcher failure.

Fix:
- Thread the rollout launcher PID into `wait_for_rollout_health()`.
- Fail immediately if the launcher exits before health or exits during a post-health dwell window.
- Re-check health after the dwell window so a transient pass does not count.
- Also pass explicit `PORT_A` and `PORT_B` into the rollout startup srun wrapper.

Patched file:
- `examples/train_integrations/harbor/run_qwen3_32b_full_readiness_suite.sh`

Status:
- Fixed.

## Error 22: `Training smoke` could still report `PASS` even when the training entrypoint emitted a preflight failure

Signature:
- Full readiness printed:
- `[preflight] FAIL: ...`
- `srun: error: research-dev-coder-003: task 0: Exited with exit code 1`
- but the outer wrapper still concluded:
- `PASS: Training smoke`

Root cause:
- The readiness wrapper trusted the transport layer too much for this path.
- For the training smoke, the reliable failure signal is the emitted preflight marker in the captured output, not only the wrapper exit plumbing.

Fix:
- Add a dedicated `run_training_smoke_check()` wrapper.
- Capture the full output from the head-node smoke run.
- Fail if the command exits nonzero or if the output contains a `^[preflight] FAIL:` marker.

Patched file:
- `examples/train_integrations/harbor/run_qwen3_32b_full_readiness_suite.sh`

Status:
- Fixed.

## Error 23: Ray preflight could block for 60 seconds waiting for `node_ip_address.json`

Signature:
- Full validation stalled in preflight while repeatedly printing:
- `Can't find a node_ip_address.json file from /tmp/ray/session_...`
- It only failed after the full 60 second wait inside Ray's local node bootstrap.

Root cause:
- Under `ray==2.51.1`, the connection path used by `ray.init(address=...)` can wait for a local `node_ip_address.json` unless `_node_ip_address` is explicitly supplied.
- In this environment the session directory was created, but the node IP cache file was not, so preflight burned 60 seconds before failing.

Fix:
- Compute the local node IP from Ray's perspective and pass it explicitly via the hidden `_node_ip_address` kwarg to `ray.init(...)`.

Patched file:
- `examples/train_integrations/harbor/preflight_qwen3_32b_6node_cluster.py`

Status:
- Fixed.

## Error 24: Validation preflight was running on the login node instead of the Ray head node

Signature:
- Direct preflight from the current shell showed:
- `This node has an IP address of 172.21.44.6, but we cannot find a local Raylet with the same address`
- followed by:
- `Failed to get the system config from raylet because it is dead`

Root cause:
- `run_full_run_validation_suite.sh` executed the Python preflight locally from the control shell.
- In this session the control shell lives on the login node, not on `research-dev-coder-003`.
- That made Ray create a local client-side node on the wrong host instead of running the placement-group probe from the actual cluster head.

Fix:
- Run the final Python preflight through `run_on_node "$HEAD_NODE"` so it executes on the Ray head node inside the Slurm allocation.
- Keep the arguments and environment explicitly forwarded into that `srun` wrapper.

Patched file:
- `examples/train_integrations/harbor/run_full_run_validation_suite.sh`

Status:
- Fixed.

## Error 25: Worker-side socket IFNAME repair could select `lo`, leading NCCL to connect to `127.0.0.1`

Signature:
- `Training smoke` failed after model load with:
- `socketPollConnect: connect to 127.0.0.1<...> returned Connection refused`
- and the same run logged:
- `NCCL_SOCKET_IFNAME='enp53s0f0np0' is not present on this node; overriding with 'lo'`

Root cause:
- The worker-side interface repair path called `detect_socket_ifname.sh` with `self._master_addr`.
- When the target IP routed back to the local host, `ip route get` returned `dev lo`.
- The repair path accepted `lo` as a valid distributed socket interface and wrote it into `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`.
- That poisoned multi-node NCCL rendezvous and caused peers to attempt connections to `127.0.0.1`.

Fix:
- Reject `lo`, `docker*`, `br-*`, and `veth*` as valid results in `detect_socket_ifname.sh`.
- Add a second guard in `worker.py` so even if the detector returns an unusable interface, it is not propagated into distributed env vars.

Patched files:
- `examples/train_integrations/harbor/detect_socket_ifname.sh`
- `skyrl/backends/skyrl_train/workers/worker.py`

Status:
- Fixed; needs smoke rerun.

## Error 26: Training smoke could fail at checkpoint save because the default run artifact root was on a full `/data` filesystem

Signature:
- `Training smoke` completed model init and real ThunderAgent traffic, then failed during checkpoint save with:
- `OSError: [Errno 122] Disk quota exceeded`
- The traceback came from:
- `fsdp_strategy.py -> save_checkpoint() -> fsspec local close()`

Root cause:
- The launch script defaulted `RUN_ARTIFACT_ROOT` to `/data/zy/models/$USER/harbor_runs`.
- On the current cluster `/data` is already full (`19T used, 0 available`), so the first policy checkpoint write failed even though training and rollout were otherwise healthy.

Fix:
- Immediate remediation was to stop the run and delete the stale checkpoint tree under:
- `/data/zy/models/hkang/harbor_runs/codecontest-qwen3-32b-6node-rootless-smoke-flash-attn-memtrace/ckpts`
- That reclaimed the blocked `/data` space without touching the model directory.
- Per user preference, do not redirect artifacts to `/home`; keep the launch default on `/data` for now and manage checkpoint footprint there.

Patched file:
- None. A temporary `/home` redirection change was reverted.

Status:
- Space cleanup done on `/data`; rerun still needed before full run.

## Error 27: Harbor trial startup exhausted Docker's default bridge address pools

Signature:
- Full 10-step run failed to start new Harbor environments on `research-dev-coder-003`.
- `docker compose up -d` returned:
- `all predefined address pools have been fully subnetted`

Root cause:
- On rootful/system Docker, Harbor's default compose flow creates one project-scoped bridge network per trial environment.
- This full run launched enough concurrent `CodeContests` trials that Docker's default address pools were exhausted even without large numbers of stale historical networks.
- During the failure window, `003` had `29` live `code_contests-*` compose containers and `29` matching networks from the current run itself.

Fix:
- Added `HARBOR_DOCKER_DISABLE_PROJECT_NETWORK=1` support in Harbor's docker backend for Dockerfile-only tasks.
- When enabled, Harbor appends `docker-compose-no-network.yaml` instead of creating a per-trial compose bridge network.
- Exported this env var through SkyRL's Ray runtime env passthrough and the reusable 10-step wrapper.

Patched files:
- `.venv/lib/python3.13/site-packages/harbor/environments/docker/docker.py`
- `skyrl/train/utils/utils.py`
- `examples/train_integrations/harbor/run_qwen3_32b_full_10step_capture.sh`

Status:
- Patched; current failed run still needs to be stopped and relaunched so the new Harbor env var takes effect.

## Error 29: Rootless Harbor with aggressive full-run concurrency can saturate the kernel key quota and fail container init with `unable to create session key`

Signature:
- On the rootless Docker path, Harbor trial startup no longer hit `all predefined address pools have been fully subnetted`.
- The replacement failure was:
- `unable to join session keyring: unable to create session key: disk quota exceeded`
- The error surfaced from:
- `docker compose up -d`
- `runc create failed`
- `OCI runtime create failed`

Direct evidence:
- Trial exceptions under:
- `/home/hkang/zthunder_agent/harbor_runs/codecontest-qwen3-32b-6node-full-10step-20260313_081537/trials_run/*/exception.txt`
- Example failing environment:
- `code_contests-0004__QW3uQtT`
- Head-node key quota probe on `research-dev-coder-003` during the run showed:
- `/proc/sys/kernel/keys/maxkeys = 200`
- `/proc/key-users` for `hkang` at failure time:
- `243001624:   200 200/200 200/200 14000/20000`

Root cause:
- The current aggressive full-run parameters create up to `256` concurrent Harbor trial environments.
- Under rootless Docker on this cluster, each container startup consumes session-key/keyring quota on the Harbor head node.
- Once the per-user kernel key quota (`maxkeys=200`) is saturated, new container starts fail inside `runc` before the main process can come up.
- This is distinct from filesystem free space; it is a kernel key quota limit that happens to report as `disk quota exceeded`.

What still worked:
- The rootless address-pool fix was effective:
- the same run created `256` Harbor trial directories without any recurrence of `fully subnetted`.
- Some trials completed successfully and dumped trajectories.
- The training driver eventually filled the first generation buffer and completed `step 1`, despite high trial failure volume.

Mitigation status:
- No user-space fix has been applied yet.
- Raising `kernel.keys.maxkeys` would require a system-level change on the head node.
- Any future user-space workaround must preserve the documented Harbor networking path and avoid reintroducing the `no-network` divergence.

## Error 30: The readiness script could false-pass rootless Docker health by starting the daemon in a short-lived `srun` step

Signature:
- A rootless readiness attempt reported:
- `PASS: Harbor rootless verifier smoke`
- and then later `Readiness suite completed` with `failures: 0`
- but the advertised Docker socket was already dead after the script moved on.

Root cause:
- `run_qwen3_32b_full_readiness_suite.sh` originally started rootless Docker via a short-lived `run_on_node` shell.
- That `srun` step exited after `dockerd-rootless.sh` returned control, so the daemon lifetime was not tied to a live Slurm step.
- Subsequent checks could observe a stale socket path and produce misleading PASS output.

Fix:
- Start rootless Docker in `block` mode inside a dedicated long-lived `srun` step, the same way the rollout and Ray helpers are kept alive.
- Add an explicit `wait_for_head_docker_ready()` probe that runs `docker info` against the exact `DOCKER_HOST` that later Harbor checks will use.
- Thread the resolved `READINESS_DOCKER_HOST` through Harbor smoke, cluster validation, and training smoke so the top-level suite checks the real live socket instead of a hard-coded fallback.

Patched file:
- `examples/train_integrations/harbor/run_qwen3_32b_full_readiness_suite.sh`

Status:
- Fixed and verified by rerunning the full rootless readiness sweep against a live rootless daemon step on `research-dev-coder-003`.

## Error 31: Rootless Harbor was recoverable only after expanding the head node's kernel key quotas

Signature:
- With the rootless address-pool issue already resolved, Harbor trial startup still failed on `research-dev-coder-003` with:
- `unable to join session keyring: unable to create session key: disk quota exceeded`
- `/proc/keys` showed many `hkang`-owned `_ses.<containerid>` keyrings and `/proc/key-users` was stuck at `200/200`.

Direct evidence:
- Attempting to `invalidate` the `_ses.<containerid>` keyrings changed their flags from `I--Q---` to `I--Q--i`, but `keyctl reap -v` still reported `0 keys reaped`.
- That proved the objects were found, but quota would not be released quickly enough for rootless Harbor recovery.

Mitigation applied:
- On `research-dev-coder-003`, temporarily raise:
- `kernel.keys.maxkeys = 20000`
- `kernel.keys.maxbytes = 25000000`

Verification:
- Rootless Docker startup succeeded again on `research-dev-coder-003`.
- Harbor-only rootless concurrency smoke then passed at all tested levels:
- `16/16`
- `32/32`
- `64/64`
- `128/128`
- `256/256`
- Summaries were written under:
- `/tmp/harbor-rootless-concurrency16-after-sysctl/summary.json`
- `/tmp/harbor-rootless-concurrency32-after-sysctl/summary.json`
- `/tmp/harbor-rootless-concurrency64-after-sysctl/summary.json`
- `/tmp/harbor-rootless-concurrency128-after-sysctl/summary.json`
- `/tmp/harbor-rootless-concurrency256-after-sysctl/summary.json`

What this means:
- The stale `_ses.<containerid>` objects were not cleaned up, but they also no longer blocked new rootless container starts.
- This is a practical recovery path for the current cluster state, not proof that the underlying keyring lifecycle issue is gone.

Final readiness result after the fix:
- With the raised key quotas active on `research-dev-coder-003`, the rootless `R13` full-run preflight passed on:
- head: `research-dev-coder-003`
- trainers: `research-dev-coder-012,013,014,015`
- rollout: `research-dev-coder-008`
- The final readiness run reported:
- `PASS: Harbor rootless verifier smoke`
- `PASS: Rollout startup`
- `PASS: Cluster validation`
- `Readiness suite completed`
- `failures: 0`

Patched file:
- None for the quota change itself; this mitigation was applied via `sysctl` on the head node.

Status:
- Mitigated and verified on the current `1139` allocation.
