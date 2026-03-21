# CodeContests 8xH100 Docker Fully-Async Full Run `r5` ThunderAgent 50-Step Notes

Date: 2026-03-09

## Scope

This run was a full-dataset Harbor + SkyRL + ThunderAgent validation run using the same fully-async CodeContests setup as the Harbor full run, but with ThunderAgent correctly wired into the inference path.

Run name:

- `codecontest-8xh100-docker-fully-async-full-r5-thunderagent-tr-metrics-profile`

Key paths:

- Run dir: `/scratch/hkang/codecontest-8xh100-docker-fully-async-full-r5-thunderagent-tr-metrics-profile`
- Main log dir: `/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-fully-async-full-r5-thunderagent-tr-metrics-profile`
- ThunderAgent log: `/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-fully-async-full-r5-thunderagent-tr-metrics-profile/thunderagent.log`
- Main trainer worker log: `/scratch/hkang/skyrl_runtime/ray_tmp/ray/session_latest/logs/worker-639bfda08477f6d534b571eb9d281f650ed1be45965ab6ec6ebe0233-01000000-3586768.err`

## What Was Validated

This run was meant to answer the following questions:

1. Is Harbor actually sending traffic through ThunderAgent rather than bypassing it?
2. Is ThunderAgent receiving stable per-trajectory `program_id` values rather than collapsing everything into `default`?
3. Is ThunderAgent running in scheduler mode rather than plain proxy mode?
4. Does Harbor release programs at trial end?
5. Does the earlier Harbor rollout-details bug (`Missing completion token ids`) stay fixed under ThunderAgent?

## Effective Configuration Differences vs Plain Harbor

Compared with the plain Harbor full run, this run additionally enabled:

- `_SKYRL_USE_NEW_INFERENCE=1`
- ThunderAgent entrypoint: `examples.train_integrations.harbor.entrypoints.main_harbor_thunder_agent_fully_async`
- `generator.inference_engine.thunder_agent_mode=tr`
- `generator.inference_engine.thunder_agent_metrics_enabled=true`
- `generator.inference_engine.thunder_agent_profile_enabled=true`

Code-side integration assumptions for this run:

- Harbor passes a stable per-trial `program_id`
- Harbor still passes `session_id`, but routing correctness depends on `program_id`
- Harbor best-effort calls ThunderAgent `/programs/release` when a trial ends
- Harbor uses the inference client proxy URL, so traffic really flows through ThunderAgent

## Final Outcome

The integration worked.

Evidence:

- ThunderAgent started in `mode=tr`
- ThunderAgent started the scheduler loop
- ThunderAgent started metrics monitoring for all 4 backends
- Harbor requests were logged in `thunderagent.log`
- Requests carried non-default `program_id`
- All 4 backends `8000/8001/8002/8003` received traffic
- Harbor emitted release calls and ThunderAgent logged `Released and removed program: ...`
- The run reached trainer `global_step = 50`
- `Missing completion token ids` did not reappear

Representative ThunderAgent startup lines:

- `ThunderAgentRouter configured with 4 servers, port=8080, mode=tr`
- `Started scheduler loop (interval=5.0s)`
- `Started metrics monitoring for http://172.27.28.185:8000`
- `Started metrics monitoring for http://172.27.28.185:8001`
- `Started metrics monitoring for http://172.27.28.185:8002`
- `Started metrics monitoring for http://172.27.28.185:8003`

## Final Stats

Trainer / checkpoint state:

- Final observed trainer step: `50`
- Stop mode: external stop at target step `50`
- Latest fully saved checkpoint: `45`
- Latest fully saved export: `45`
- Reason latest saved step is `45`: stop was triggered after step `50` was reached, before the next save point completed

Trial / ThunderAgent counts:

- Trial directories created: `428`
- `result.json` count: `409`
- `exception.txt` count: `20`
- Trial dirs without `result.json`: `19`
- ThunderAgent access-log request count: `2156`
- ThunderAgent release count: `414`
- Unique non-default `program_id` count in ThunderAgent log: `414`
- `program_id=default` count: `0`

Observed error counters:

- `Missing completion token ids`: `0`
- `Agent execution timed out`: `0`
- `Invalid JSON: Invalid \\escape`: `34`

Exception breakdown:

- `harbor.llms.base.ContextLengthExceededError`: `13`
- `CancelledError`: `7`

Async behavior at the stop boundary:

- `staleness` at step `50`: `2`

## Interpretation

### 1. ThunderAgent was actually in the loop

This was not a fake "import only" run.

The decisive evidence is:

- `thunderagent.log` contains real `/v1/chat/completions` access logs
- those access logs include unique per-trial `program_id`
- no request fell back to `program_id=default`
- traffic was distributed across all 4 backend servers

### 2. Harbor program lifecycle wiring worked

Harbor did not just pass `program_id`; it also released programs.

Evidence:

- `Released and removed program: ...` appears throughout `thunderagent.log`
- release count reached `414`

The release count is not expected to match `result.json` exactly, because:

- some trial directories contain both `result.json` and `exception.txt`
- the run was externally stopped at step `50`
- stop boundary left `19` trial directories without final `result.json`

### 3. The Harbor rollout-details fix held

The earlier Harbor bug around sparse rollout details did not reappear here.

Evidence:

- `Missing completion token ids = 0`

So for this run, ThunderAgent integration did not regress the Harbor rollout-details fix.

### 4. Remaining real issues were elsewhere

The main residual issues in this run were:

- `Invalid JSON: Invalid \\escape` still appeared (`34`)
- `ContextLengthExceededError` appeared (`13`)
- some in-flight work was interrupted by the forced stop (`CancelledError = 7`)

These are separate from the ThunderAgent routing/program-id integration question.

## Stop / Cleanup

This run was intentionally stopped at trainer step `50`.

Stop sequence:

- `SIGINT` sent when step `50` was observed
- `SIGTERM` sent 30 seconds later because the main process had not exited yet

After the stop, cleanup was performed:

- Harbor/ThunderAgent/vLLM training processes stopped
- Ray stopped with `ray stop --force`
- Docker containers removed
- Docker images removed
- Docker build cache removed

Final Docker state after cleanup:

- Images: `0`
- Containers: `0`
- Local Volumes: `0`
- Build Cache: `0`

## Bottom Line

This `r5` full run validates that the corrected ThunderAgent integration is working as intended:

- Harbor traffic really goes through ThunderAgent
- ThunderAgent runs in `tr` scheduler mode
- Harbor sends stable per-trial `program_id`
- Harbor releases programs
- ThunderAgent sees real multi-backend traffic
- the previous Harbor rollout-details bug stays fixed

The integration can be considered functionally correct for further experiments.
