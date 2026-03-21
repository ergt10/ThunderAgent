# R2EGYM Qwen3-32B 6-Node Rootless Fully Async 1-Step 4-Server mini-swe-agent-50 timeout9000 fd131072 `blockdockerd` Discovered Issues

Date: 2026-03-17

Run:

- `r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051`

Logs:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051`

Artifacts:

- `/home/hkang/zthunder_agent/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051`

Stop-state snapshot from `monitoring/trial_progress.tsv`:

- `trial_dirs=379`
- `result_json_count=336`
- `exception_txt_count=126`
- `trajectory_json_count=210`
- `completed_trials_count=336`

## 1. Harbor Docker Environment Start Timeout

### Symptom

At stop time there were `126` trial exception files, and all `126` were:

- `harbor.trial.trial.EnvironmentStartTimeoutError`

Representative file:

- `/home/hkang/zthunder_agent/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051/trials_run/r2egym-0072__LQQL98m/exception.txt`

Representative failure:

```text
harbor.trial.trial.EnvironmentStartTimeoutError: Environment start timed out after 600.0 seconds
```

### Precise Failing Step

The representative stack shows:

1. `harbor.trial.trial._start_environment_with_retry()` wraps environment startup in `asyncio.wait_for(..., timeout=600.0)`
2. `harbor.environments.docker.docker.start()` enters `docker compose build`
3. the await inside `process.communicate()` is cancelled when the 600-second environment-build timeout fires

Representative local code paths:

- `.../site-packages/harbor/trial/trial.py`
  - `_start_environment_with_retry()`
- `.../site-packages/harbor/environments/docker/docker.py`
  - `start()`
  - `_run_docker_compose_command(["build"])`

### Why This Happened

R2EGYM tasks already carry a prebuilt image identifier in `environment/workspace/metadata.json`, but Harbor did not use it for this run.

Representative task:

- `/home/hkang/zthunder_agent/data/harbor/r2egym-easy/r2egym-0072/task.toml`
- `/home/hkang/zthunder_agent/data/harbor/r2egym-easy/r2egym-0072/environment/workspace/metadata.json`
- `/home/hkang/zthunder_agent/data/harbor/r2egym-easy/r2egym-0072/environment/Dockerfile`

Observed state:

- `task.toml` contains agent and verifier timeouts, but no `docker_image` field
- `metadata.json` contains:
  - `"docker_image": "namanjain12/orange3_final:6e8d153bf7c20e9aa2cc598ac584ca83cbd3f118"`
- `environment/Dockerfile` still says:
  - `FROM namanjain12/orange3_final:6e8d153bf7c20e9aa2cc598ac584ca83cbd3f118`

Harbor Docker startup uses:

- `_use_prebuilt = not force_build and self.task_env_config.docker_image`

Because `task_env_config.docker_image` was not populated from the R2EGYM task tree, Harbor treated these tasks as build-required and ran:

- `docker compose build`
- `docker compose down --remove-orphans`
- `docker compose up -d`

for each environment.

Under this run's concurrency, rootless dockerd on the head node showed repeated cleanup distress:

- `Container failed to exit within 10s of kill - trying direct SIGKILL`

Representative log:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051/dockerd_rootless_harbor.log`

### Impact

- Trials never reach agent execution
- No verifier runs for those trials
- Harbor retries startup, but repeated timeouts continue to consume wall time and Docker control-plane capacity

### Current Direction

The most likely fix direction is to stop rebuilding R2EGYM task environments during Harbor startup:

- propagate the task's prebuilt `docker_image` into Harbor task environment config
- or teach the R2EGYM Harbor preparation path to prefer the image directly instead of rebuilding from the local `Dockerfile`

## 2. mini-swe-agent Missing Rollout Details In Harbor Postprocess

### Symptom

`launcher_train_driver.log` contains `210` warnings of the form:

```text
did not return assistant logprobs/token ids despite collect_rollout_details=True
```

Representative lines:

- `/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051/launcher_train_driver.log`

### Why This Is A Different Bug From `metadata=None`

This is not the old `metadata=None` failure.

Current mini-swe-agent wrapper behavior:

- it now fills `context.metadata`
- it still does not fill Harbor rollout details with assistant logprobs and completion token ids

Representative local code:

- `.../site-packages/harbor/agents/installed/mini_swe_agent.py`
  - `context.metadata = _build_harbor_metadata_from_messages(...)`

Current Harbor generator expectation:

- `examples/train_integrations/harbor/harbor_generator.py`
  - if `collect_rollout_details=True` and either assistant logprobs or completion token ids are missing, it raises

Representative message:

```text
Harbor trial for trajectory ... did not return assistant logprobs/token ids despite collect_rollout_details=True.
```

### Impact

- The underlying Harbor trial may have run
- The trajectory is discarded during Harbor postprocess
- The sample is masked out before training can use it

### Code Fix Applied Locally On 2026-03-18

The local Harbor adapter has now been patched in:

- `/home/hkang/zthunder_agent/SkyRL/.venv/lib/python3.13/site-packages/harbor/agents/installed/mini_swe_agent.py`

What changed:

- `MiniSweAgent` now reads `collect_rollout_details` from Harbor agent kwargs
- when enabled, it adds `logprobs=True` to mini-swe-agent `model_kwargs`
- for the hosted-vLLM path, it also adds `extra_body.return_token_ids=True`
- after the run, it now converts each assistant raw response into Harbor `context.rollout_details` using:
  - `response.prompt_token_ids`
  - `choices[0].provider_specific_fields.token_ids`
  - `choices[0].logprobs.content[*].logprob`

The helper was also smoke-tested locally with a synthetic response object and produces Harbor-shaped rollout details.

### Remaining Validation

This still needs one live rerun to verify that the hosted-vLLM backend actually returns non-null:

- `prompt_token_ids`
- `choices[0].provider_specific_fields.token_ids`
- `choices[0].logprobs`

If those fields come back populated, this issue is resolved for future runs.

## 3. ThunderAgent `httpx.ReadTimeout`

### Symptom

This run recorded `7` ThunderAgent request timeouts:

- `6` on backend `http://172.21.44.94:18003`
- `1` on backend `http://172.21.44.94:18000`

Representative `thunderagent.log` lines:

```text
2026-03-17 22:54:44,830 ... program_id=202fc0bb1ae54b0a831eaf857b299fd1 backend=http://172.21.44.94:18003 failed after 900012.6ms
2026-03-17 22:56:18,600 ... program_id=4c2465ef12cd496eb8a0822f6c8e1aeb backend=http://172.21.44.94:18003 failed after 900002.2ms
2026-03-17 22:59:06,853 ... program_id=1692fdb60a834be68ba7bb0b2636f28e backend=http://172.21.44.94:18000 failed after 902679.3ms
2026-03-17 23:29:31,590 ... program_id=6617c2151d3b45cd91ddcec0d61e7e23 backend=http://172.21.44.94:18003 failed after 900004.5ms
2026-03-17 23:29:55,292 ... program_id=a486fd10a80640019165f0fec33ae412 backend=http://172.21.44.94:18003 failed after 900005.5ms
2026-03-17 23:30:22,994 ... program_id=ad7751caf4a14320b2db81da678f6225 backend=http://172.21.44.94:18003 failed after 900002.2ms
2026-03-17 23:32:25,653 ... program_id=78922fdf940343b29d92a27b6ea7034f backend=http://172.21.44.94:18003 failed after 900017.3ms
```

The main driver log also contains `7` `httpcore.ReadTimeout` mentions and `7` `httpx.ReadTimeout` mentions.

### High-Level Cause

This run used one stable `program_id` per Harbor trial attempt. mini-swe-agent sent that same `program_id` to ThunderAgent on every model turn.

Relevant local behavior:

- local `litellm` defaults the caller-side timeout to `600s` when no timeout is supplied
- ThunderAgent router uses an `httpx.AsyncClient(timeout=900.0)`
- ThunderAgent forwarded these calls through the non-streaming `client.post(...)` path
- ThunderAgent does not enforce a single in-flight request per `program_id`
- ThunderAgent also does not cancel an already forwarded backend request when the same `program_id` is later resumed elsewhere

The result is:

1. mini-swe-agent waits less time than ThunderAgent does
2. mini-swe-agent can retry the same logical model call with the same `program_id`
3. the old backend request keeps running inside ThunderAgent
4. a newer request with the same `program_id` can also be accepted
5. the stale request later hits ThunderAgent's own `900s` timeout

Representative mapped sample:

- `program_id=1692fdb60a834be68ba7bb0b2636f28e`
- trial: `r2egym-0145__ZKAXEZt`
- trajectory `api_calls=40`
- ThunderAgent log counts for that `program_id`:
  - `40` responses with `status=200`
  - `1` `failed after ...`
  - `1` final `status=400`

That is consistent with one extra stale request surviving after the logical 40 model turns.

### Current Recorded Mitigation

The current mitigation to carry forward in notes is:

- raise the mini-swe-agent-side model timeout to `1200s`

The purpose of that change is to make the agent-side caller wait longer than ThunderAgent's own `900s` backend timeout, so mini-swe-agent does not retry the same logical call early and create overlapping same-`program_id` requests.

This is a recorded fix direction, not yet validated by a fresh rerun.
