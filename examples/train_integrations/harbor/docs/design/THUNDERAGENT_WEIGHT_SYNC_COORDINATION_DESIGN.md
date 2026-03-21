# ThunderAgent Weight Sync Coordination Design

Date: 2026-03-19

## Problem

When SkyRL's fully async trainer finishes a training step, it pauses all vLLM backends, syncs updated model weights, then resumes. This is done via:

```python
# fully_async_trainer.py:415-419
await self.inference_engine_client.pause_generation()      # /pause -> vLLM backends
await self.async_sync_policy_weights_to_inference_engines() # weight sync
await self.inference_engine_client.resume_generation()      # /resume -> vLLM backends
```

`RemoteInferenceClient` has two URL sets:
- `proxy_url` = ThunderAgent router (data plane: `/inference/v1/generate`)
- `server_urls` = vLLM backend URLs (control plane: `/pause`, `/resume`, `/sleep`, `/wake_up`)

Control plane calls go **directly to vLLM backends, bypassing ThunderAgent entirely**. TA has zero visibility into the weight sync cycle.

## What Goes Wrong

### 1. Mid-flight requests get aborted

Programs in REASONING status have active requests to vLLM. With `PauseMode.ABORT`, vLLM aborts these in-flight requests. The caller in `ThunderAgentRouter.inference_generate` (line 218-233) gets an exception, which is currently only logged — no retry. The program's state becomes inconsistent: TA thinks the request was forwarded, but it was aborted.

### 2. TA scheduler runs blind during weight sync

TA's scheduler loop (5s interval) continues running. It fetches metrics from paused backends, sees reduced/zero capacity, and may start pausing programs unnecessarily. After vLLM resumes, TA may have paused a large fraction of programs based on a transient state.

### 3. New requests routed to paused backends

Programs in ACTING status that finish their tool execution during weight sync will make their next request to TA. TA routes them to a paused vLLM backend. The request either blocks indefinitely or errors.

### 4. No state reconciliation after resume

After weight sync, vLLM's KV cache may be invalidated (depending on weight sync implementation). Programs whose KV cache entries were evicted need to be re-prefilled, but TA doesn't know this happened.

## Current Data Flow

```
Trainer                     RemoteInferenceClient           vLLM Backends
  |                              |                              |
  |--- pause_generation() ----->|                              |
  |                              |--- POST /pause ------------->| (direct, bypasses TA)
  |                              |                              |
  |--- sync_weights() -------->|                              |
  |                              |--- weight transfer --------->|
  |                              |                              |
  |--- resume_generation() --->|                              |
  |                              |--- POST /resume ------------>| (direct, bypasses TA)

Meanwhile:
ThunderAgent Router (unaware)
  |--- scheduler tick: fetch metrics from paused backends -> confused
  |--- route new requests to paused backends -> fail/block
  |--- programs stuck in REASONING on aborted requests -> orphaned
```

## Proposed Design: Weight Sync Awareness in TA

### Option A: Trainer notifies TA (recommended)

Add a weight sync lifecycle to the ThunderAgent router. The trainer brackets the weight sync with explicit notifications to TA.

**New TA endpoints:**

```
POST /weight_sync/begin   -> TA enters "weight sync mode"
POST /weight_sync/end     -> TA exits "weight sync mode"
```

**TA behavior in weight sync mode:**

1. **Stop scheduler**: Don't run `_greedy_resume` or `_pause_until_safe` ticks
2. **Hold new requests**: Queue incoming requests (don't route to backends). Programs transition to a new pseudo-state `WAITING_FOR_SYNC` instead of being sent to vLLM
3. **Don't retry aborted requests**: Programs whose in-flight requests were aborted get marked as needing re-submission after sync completes
4. **On sync end**:
   - Resume scheduler
   - Flush held requests to now-resumed backends
   - Re-submit aborted programs (they need to re-prefill KV cache anyway since weights changed)

**Trainer-side change:**

```python
# fully_async_trainer.py
await self.inference_engine_client.notify_weight_sync_begin()  # POST to TA proxy_url
await self.inference_engine_client.pause_generation()
await self.async_sync_policy_weights_to_inference_engines()
await self.inference_engine_client.resume_generation()
await self.inference_engine_client.notify_weight_sync_end()    # POST to TA proxy_url
```

### Option B: TA intercepts /pause and /resume

Route control plane through TA as well (not just data plane). TA intercepts `/pause` and `/resume`, forwards them to backends, and enters sync mode automatically.

**Pros**: No trainer-side changes
**Cons**: Requires changing `RemoteInferenceClient` to route control plane through `proxy_url`, which breaks the separation of proxy_url (single) vs server_urls (fan-out). Also, `/pause` might be called for reasons other than weight sync.

### Option C: TA polls backend health

TA's scheduler already fetches backend metrics each tick. It could detect backends transitioning to paused state and infer weight sync is happening.

**Pros**: Zero protocol changes
**Cons**: Reactive instead of proactive (5s lag). Can't distinguish weight sync from other pause reasons. Fragile.

## Recommendation: Option A

Option A is cleanest — explicit coordination, no ambiguity about intent, and TA can take the right actions immediately.

### Implementation Plan

**ThunderAgent side:**

1. Add `weight_sync_active: bool` flag to `MultiBackendRouter`
2. Add `_weight_sync_held_requests: asyncio.Queue` for queued requests
3. In `update_program_before_request`: if `weight_sync_active`, await on a `weight_sync_event` instead of routing
4. In scheduler loop: skip ticks while `weight_sync_active`
5. Register `/weight_sync/begin` and `/weight_sync/end` endpoints via `register_routes` or in `ThunderAgentRouter._build_app`
6. On `/weight_sync/end`: set event to release held requests, resume scheduler

**SkyRL side:**

1. Add `notify_weight_sync_begin()` and `notify_weight_sync_end()` to `RemoteInferenceClient`
   - These send to `proxy_url` (TA), not `server_urls` (backends)
2. Bracket the pause/sync/resume in `fully_async_trainer.py` with begin/end calls
3. Handle the case where TA is not used (no `proxy_url` or proxy doesn't support the endpoint): no-op fallback

**ThunderAgentRouter side (integration layer):**

Add the `/weight_sync/begin` and `/weight_sync/end` routes in `_build_app`, delegating to `MultiBackendRouter.begin_weight_sync()` / `end_weight_sync()`.

### Open Questions

- Should TA also reset its token accounting after weight sync? Weights changed, so the "total_tokens" on each program no longer reflects KV cache reality on the backend.
- Should programs that were REASONING when sync started be forced to re-prefill from scratch? (Likely yes, since KV cache computed with old weights is stale after weight update.)
- How does this interact with the prompt-group-aware pause/resume design? During weight sync, all groups are held equally — no group-level logic needed in sync mode.
- For external TA setups (both `external_proxy_url` and `external_server_urls` provided), the trainer needs to know if the external proxy supports weight sync endpoints. A capability probe (`GET /weight_sync/supported`) could handle this.
