# CodeContests Qwen3-32B 6-Node Rootless Fully Async 5-Step 4-Server mini-swe-agent-50 retry4 `rerunfixfd` Discovered Issues

Date: 2026-03-17

Run name:

- `codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633`

Primary logs:

- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/launcher_train_driver.log`
- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/thunderagent.log`
- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/monitoring/trial_progress.tsv`
- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/rollout/monitoring/vllm_metrics.tsv`

Primary artifacts:

- `/home/hkang/zthunder_agent/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/trials_run`

This document only records problems confirmed from the final `rerunfixfd` launch path.

## 1. Confirmed Run Outcome

The run started correctly, filled one generation buffer, started the first training update, and then hung at the first `sync_weights` boundary.

The critical timeline from the main log was:

- `2026-03-17 04:56:37 -0700`: `Started: 'step'`
- `2026-03-17 04:56:37 -0700`: `Started: 'wait_for_generation_buffer'`
- `2026-03-17 05:20:12 -0700`: `Finished: 'wait_for_generation_buffer', time cost: 1415.15s`
- `2026-03-17 05:20:12 -0700`: `All outputs are loss masked`
- `2026-03-17 05:20:24 -0700`: `Started: 'sync_weights'`
- there is no matching `Finished: 'sync_weights'`
- there is no matching `Finished: 'step'`

At the monitoring level, the run froze with the last trial-progress counts:

```text
2026-03-17T05:46:38-0700 ... 523 267 264 132 267
```

Interpreted as:

- `trial_dirs = 523`
- `result_json_count = 267`
- `exception_txt_count = 264`
- `trajectory_json_count = 132`
- `completed_trials_count = 267`

The rollout servers were idle after the freeze, while the head-side router became unresponsive.

## 2. Trial Result Breakdown

Completed `result.json` files under `trials_run`:

- `267`

Exception breakdown:

- `253` `AgentTimeoutError`
- `11` `RuntimeError`
- `3` `no_exception`

The `3` no-exception trials were:

- `/home/hkang/zthunder_agent/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/trials_run/code_contests-0032__F4HVJgr/result.json`
- `/home/hkang/zthunder_agent/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/trials_run/code_contests-0094__dasxCXT/result.json`
- `/home/hkang/zthunder_agent/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/trials_run/code_contests-0153__tr9GNBz/result.json`

These `3` trials still failed to produce trainable outputs because Harbor postprocessing later rejected them for missing assistant logprobs and token ids.

## 3. Problem 1: Agent Setup Failures Before Solving Began

There were `11` final `RuntimeError` trial outcomes. All `11` had the same final error shape:

- `Agent setup failed with exit code 6`

Representative failing trial:

- `/home/hkang/zthunder_agent/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/trials_run/code_contests-0336__S33XRq7/result.json`

Representative setup stdout:

- `/home/hkang/zthunder_agent/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/trials_run/code_contests-0336__S33XRq7/agent/setup/stdout.txt`

Representative log content:

```text
curl: (6) Could not resolve host: github.com
```

Confirmed count of setup stdout files that contain that exact DNS failure:

- `10`

Important detail:

- this failure happens during `agent setup`
- it occurs before the sample enters the actual multi-turn solve loop
- these samples are bad because the agent never became runnable

The main log contained `44` occurrences of:

- `Agent setup failed with exit code 6`

That number is larger than `11` because the same trajectory can retry and emit multiple warnings before the final `result.json` is written.

## 4. Problem 2: The Majority Of Samples Timed Out At The Trial Level

The dominant failure mode was:

- `253` `AgentTimeoutError`

Representative final error text:

```text
Agent execution timed out after 900.0 seconds
```

This is not a single-request timeout. It is the timeout on the full Harbor `agent_execution` phase for one sample:

- all LLM calls
- all tool calls
- all waiting
- all retries
- all internal loop overhead

Why this matters:

- a sample can look active for many minutes
- but if a few of its internal request rounds stall badly enough, the whole sample uses up the `900s` budget
- Harbor then turns the whole trial into `AgentTimeoutError`

This is why the high `AgentTimeoutError` count is evidence of execution-path instability, not merely "the model solved the problem incorrectly"

## 5. Problem 3: Router-To-Backend HTTP Read Failures During Execution

The main log contained:

- `76` `httpx.ReadTimeout`
- `76` `httpcore.ReadTimeout`
- `2` `httpx.ReadError`
- `2` `httpcore.ReadError`

These errors occurred inside the ThunderAgent request-forwarding path:

- `/home/hkang/zthunder_agent/SkyRL/ThunderAgent/ThunderAgent/scheduler/vllm_request_processor.py`

The concrete failing line is the non-streaming backend post:

```python
resp = await client.post(url, json=payload)
```

What these errors mean in concrete network terms:

- `ReadTimeout`: the router successfully issued the HTTP request to the rollout backend, but timed out while waiting to read the response
- `ReadError`: the router successfully issued the HTTP request, but the response stream broke while it was being read

These are not:

- DNS lookup failures
- connection-refused errors
- local parser errors
- model correctness errors

They are backend read-path failures during live request execution.

Representative stack in the main log:

- `examples/train/thunder_agent/thunder_agent_router.py`
- `ThunderAgent/ThunderAgent/scheduler/router.py`
- `ThunderAgent/ThunderAgent/scheduler/vllm_request_processor.py`
- `httpx/_client.py`
- `httpcore/_async/http11.py`

The errors surfaced through the router's `/v1/chat/completions` handler while the router was forwarding Harbor's LLM calls to rollout servers.

## 6. Problem 4: `Program ... has no valid backend`

The main log contained:

- `30` occurrences of `Program ... has no valid backend`

This error comes from:

- `/home/hkang/zthunder_agent/SkyRL/ThunderAgent/ThunderAgent/scheduler/router.py`

The exact code path is the "normal existing program" branch:

```python
backend = self.backends.get(state.backend_url)
if not backend:
    logger.error(f"Program {program_id} has no valid backend")
    return False
```

The concrete meaning is:

- ThunderAgent still has a request carrying this `program_id`
- but its in-memory `state.backend_url` no longer points to a valid backend entry

This is not a vague routing failure. It means the router lost the valid backend pointer for a live program.

### 6.1 Concrete Program Timeline

The clearest example is:

- `program_id=fb63e6639936494ba81a0744cb7ab512`

From:

- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/thunderagent.log`

This same program had successful accesses on backend `18000` with these latencies:

- `116193.8 ms`
- `31273.3 ms`
- `30758.1 ms`
- `33622.8 ms`
- `40989.6 ms`

Then later:

- `2026-03-17 05:07:57`: `Marked program fb63... for pause`
- `2026-03-17 05:12:05`: `Paused program fb63... from http://172.21.44.94:18000`
- `2026-03-17 05:12:05`: access log for the same program with `latency_ms=565391.3`
- `2026-03-17 05:13:36`: `Released and removed program: fb63...`
- `2026-03-17 05:13:36`: immediately after that, `Program fb63... has no valid backend`
- `2026-03-17 05:19:36`: another access log for the same program with `latency_ms=451141.3`

This sequence shows a real state inconsistency:

- the router removed the program from its active backend tracking
- but requests associated with the same `program_id` still existed long enough to produce later access-log completions

The key fact is not just that there was an error line. The key fact is that cleanup and late request completion overlapped on the same program identity.

## 7. Problem 5: Harbor Trial Cleanup Can Race With Late ThunderAgent Traffic

Harbor program release is issued here:

- `/home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/harbor_generator.py`

The exact behavior:

- each trial attempt gets a fresh `trial_session_id`
- that `trial_session_id` is attached to both `session_id` and `program_id`
- Harbor always calls `_best_effort_release_program(trial_session_id)` in the `finally` block

This means Harbor does not wait for some stronger external proof that all router/backend traffic for that program has drained. It best-effort releases the ThunderAgent program as soon as the local attempt exits its `try/finally` scope.

Why this matters for the observed failure:

- the router had already logged very long request latencies for some programs
- therefore some requests were still alive for many minutes
- Harbor release could happen while late traffic for the same `program_id` was still in flight or still finishing

This is the most concrete explanation found for why a released program could later participate in `has no valid backend` behavior.

## 8. Problem 6: Samples That Finished Still Failed Harbor Postprocessing

The main log contained:

- `3` occurrences of:
  - `did not return assistant logprobs/token ids despite collect_rollout_details=True`

These correspond exactly to the `3` `result.json` files with no final exception.

The three affected samples were:

- `code_contests-0032__F4HVJgr`
- `code_contests-0094__dasxCXT`
- `code_contests-0153__tr9GNBz`

The key point:

- these trials were not dropped because Harbor trial execution itself threw
- they were dropped later because the output lacked assistant token ids and logprobs needed for policy training

Therefore:

- even the "clean" trials did not become trainable samples

## 9. Problem 7: The First Training Batch Was Completely Masked

The main log contained:

- `1` occurrence of `All outputs are loss masked`

The precise sequence in the main log was:

```text
2026-03-17 05:20:12.836 Finished: 'wait_for_generation_buffer', time cost: 1415.15s
2026-03-17 05:20:12.839 WARNING ... All outputs are loss masked
2026-03-17 05:20:24.102 Started: 'sync_weights'
```

This matters because even before the sync hang, the first collected mini-batch had already collapsed to a fully masked training input.

In practical terms:

- setup failures contributed zero trainable data
- timeouted trials contributed zero trainable data
- the three postprocess-failed trials contributed zero trainable data

So the trainer reached the first update boundary with no usable outputs.

## 10. Problem 8: `sync_weights` Hung And Never Returned

This run reached:

- `Started: 'sync_weights'`

It never reached:

- `Finished: 'sync_weights'`

After that point:

- `trial_progress.tsv` stopped advancing
- rollout metrics showed the rollout servers idle
- the head-side router stopped answering `/router_state`

Confirmed timeout count in the main log:

- `49` occurrences of:
  - `Failed to scrape http://172.21.44.54:8080/router_state: TimeoutError`

A direct curl against the endpoint also timed out during investigation.

This means:

- the router process itself became non-responsive at the observability endpoint
- the freeze was not just a stale monitor script

## 11. Problem 9: ThunderAgent Pause Logic Does Not Converge Cleanly In This Failure Mode

The confirmed code path during the hang was:

1. trainer entered `sync_weights`
2. generation pause started
3. ThunderAgent router began `_pause_until_safe()`
4. the router repeatedly logged `Scheduler marked REASONING program ... for pause`
5. no later `sync_weights` completion was logged

The router implementation details that matter are:

- `REASONING` programs are not immediately paused
- they are only marked with `marked_for_pause=True`
- they are actually paused later only when the next request path triggers `_clear_mark_and_pause()`

Relevant code:

- `_mark_program_for_pause()`
- `_clear_mark_and_pause()`
- `_pause_until_safe()`

The convergence problem in this implementation is:

- `_pause_until_safe()` loops while `backend.remaining_capacity() < 0`
- but `remaining_capacity()` does not subtract `future_paused_tokens`
- marking a reasoning program therefore does not directly satisfy the loop's exit condition

So in the exact failure mode seen here:

- backend requests were already suffering `ReadTimeout` and `ReadError`
- many reasoning programs were only marked, not actually paused
- the "future release" implied by those marks was not counted by `remaining_capacity()`
- the router stayed in the pause loop without reaching a stable safe state

This is the specific scheduler bug that explains the hang at `sync_weights`.

## 12. Problem 10: This Was Not The Earlier `Too Many Open Files` Failure

This run did not reproduce the old `nofile` crash path.

The repo state used for this run already had:

- head nofile raising in `launch_qwen3_32b_ray_cluster.sh`
- rollout nofile raising in `start_qwen3_32b_external_rollout_servers.sh`

The observed failure signatures in this run were:

- `ReadTimeout`
- `ReadError`
- `has no valid backend`
- `sync_weights` hang

Not:

- `OSError: [Errno 24] Too many open files`

This distinction matters because the hang root cause is different from the earlier head-side FD exhaustion issue.

## 13. Comparison Against The Earlier CodeContests Run That Reached Step Completion

Earlier CodeContests run used for comparison:

- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-3step-4srv-rerun4-20260315_2207/launcher_train_driver.log`

Key differences observed directly from that earlier log:

- it used `terminus-2`
- this run used `mini-swe-agent`
- the earlier run had `3` `Started: 'step'`
- the earlier run had `3` `Finished: 'step'`
- the earlier run had `0` `AgentTimeoutError`
- the earlier run had `0` `has no valid backend`
- the earlier run had `0` `httpx.ReadTimeout`
- the earlier run had `0` `httpcore.ReadTimeout`

The earlier run still had some issues:

- `9` `did not return assistant logprobs/token ids despite collect_rollout_details=True`
- `2` `All outputs are loss masked`

But it did not exhibit the router/backend state corruption pattern that dominated this `mini-swe-agent` run.

The observed practical difference is:

- the earlier `terminus-2` CodeContests run advanced through complete steps
- this `mini-swe-agent max_turns=50` CodeContests run generated long-lived, unstable per-program traffic and then hung at the first sync boundary

## 14. Consolidated Causal Chain

All confirmed problems fit together as one chain:

1. Some samples died immediately in setup because agent setup could not resolve `github.com`.
2. Many surviving samples entered long multi-turn execution and experienced backend read timeouts and read errors through ThunderAgent.
3. During that same execution window, some programs lost valid backend association and emitted `has no valid backend`.
4. Some Harbor cleanup release calls occurred while long-latency traffic for the same program identities was still active.
5. The resulting trial population was almost entirely bad:
   - `11` setup-failed final runtime errors
   - `253` full trial timeouts
   - `3` postprocess-failed no-exception trials
6. The trainer reached the first update boundary with `All outputs are loss masked`.
7. During `sync_weights`, ThunderAgent entered a non-converging pause path and the router stopped responding.
8. The run then hung instead of advancing to the next step.

## 15. Files Worth Inspecting First

If reviewing the failure from scratch, the highest-signal files are:

- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/launcher_train_driver.log`
- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/thunderagent.log`
- `/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/monitoring/trial_progress.tsv`
- `/home/hkang/zthunder_agent/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/trials_run/code_contests-0336__S33XRq7/agent/setup/stdout.txt`
- `/home/hkang/zthunder_agent/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/trials_run/code_contests-0336__S33XRq7/result.json`
- `/home/hkang/zthunder_agent/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/trials_run/code_contests-0032__F4HVJgr/result.json`
- `/home/hkang/zthunder_agent/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/trials_run/code_contests-0094__dasxCXT/result.json`
- `/home/hkang/zthunder_agent/codecontest-qwen3-32b-6node-rootless-full-5step-4srv-miniswe50-retry4-rerunfixfd-20260317_044633/trials_run/code_contests-0153__tr9GNBz/result.json`

