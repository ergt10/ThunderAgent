# ThunderAgent Integration for SkyRL

## Overview

This example integrates [ThunderAgent](../../ThunderAgent/) as an optional inference gate inside SkyRL's training pipeline. When enabled, ThunderAgent replaces SkyRL's simple `InferenceRouter` with a program-aware scheduling proxy that:

- **Tracks per-program state** (REASONING vs ACTING) for agentic workloads
- **Manages GPU capacity** via BFD bin-packing across multiple backends
- **Provides pause/resume scheduling** to prevent KV-cache thrashing
- **Intercepts `/v1/chat/completions` and `/inference/v1/generate`** for program scheduling while passing all other data plane endpoints (`/tokenize`, etc.) through as a catch-all proxy

## Design Approach

This integration follows SkyRL's [development guide](https://docs.skyrl.ai/docs/getting-started/development) extension pattern, specifically the precedent set by `examples/train/flash_rl/` which overrides `BasePPOExp.get_inference_client()` for custom inference backends.

**No core SkyRL files are modified.** The integration is entirely self-contained in `examples/train/thunder_agent/`.

### Architecture

```
Agent Client
    │
    ▼
┌─────────────────────────────────────┐
│  ThunderAgentRouter (FastAPI)       │
│  ├─ POST /v1/chat/completions      │──▶ ThunderAgent MultiBackendRouter
│  │    (program scheduling)          │      (capacity tracking, pause/resume)
│  ├─ POST /inference/v1/generate     │──▶ ThunderAgent program tracking
│  │    (SkyRL rollout traffic)       │      (backend assignment, token accounting)
│  ├─ GET  /health                    │──▶ Combined health + program stats
│  ├─ GET  /programs                  │──▶ ThunderAgent program state
│  ├─ POST /programs/release          │──▶ Release a program
│  ├─ GET  /servers                   │──▶ List backend URLs
│  └─ /{path:path}  (catch-all)       │──▶ Round-robin / X-Session-ID proxy
└──────────────┬──────────────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
 vLLM #1    vLLM #2    vLLM #N
```

### Extension Pattern

| Component | Pattern Source | What We Override/Extend |
|-----------|--------------|------------------------|
| `ThunderAgentExp(BasePPOExp)` | `flash_rl/main_dapo_flashrl.py` | `get_inference_client()` and `_get_new_inference_client()` |
| `ThunderAgentGeneratorConfig` | `mini_swe_agent/mini_swe_generator.py` | `InferenceEngineConfig` with ThunderAgent fields |
| `ThunderAgentConfig` | `mini_swe_agent/main_mini_swe.py` | `make_config(generator_cls=...)` |
| `ThunderAgentRouter` | `skyrl/.../inference_servers/router.py` | Same interface (`start() -> url`, `shutdown()`) |

## Files

| File | Purpose |
|------|---------|
| `config.py` | `ThunderAgentInferenceEngineConfig` (extends `InferenceEngineConfig` with 7 ThunderAgent fields), `ThunderAgentGeneratorConfig`, `ThunderAgentConfig` |
| `thunder_agent_router.py` | `ThunderAgentRouter` — FastAPI + uvicorn + background thread, composes ThunderAgent's `MultiBackendRouter` for `/v1/chat/completions` and `/inference/v1/generate` with catch-all proxy |
| `main_thunder_agent.py` | `ThunderAgentExp(BasePPOExp)` — overrides `_get_new_inference_client()` to use `ThunderAgentRouter`, plus Ray entrypoint |
| `pyproject.toml` (modified) | Added `thunderagent` optional dependency and `ThunderAgent` uv source |

## Configuration

ThunderAgent-specific config fields are available under `generator.inference_engine.*`:

| Field | Default | Description |
|-------|---------|-------------|
| `thunder_agent_mode` | `"tr"` | Router mode: `"default"` (pure proxy) or `"tr"` (capacity scheduling) |
| `thunder_agent_acting_token_weight` | `1.0` | Weight for acting tokens in capacity calculation |
| `thunder_agent_scheduler_interval` | `5.0` | Seconds between scheduler checks |
| `thunder_agent_use_acting_token_decay` | `false` | Use exponential decay for acting tokens in resume logic |
| `thunder_agent_profile_enabled` | `false` | Enable per-program profiling |
| `thunder_agent_metrics_enabled` | `false` | Enable backend metrics monitoring |
| `thunder_agent_metrics_interval` | `5.0` | Seconds between metrics fetches |

## Usage

### Prerequisites

```bash
# Install ThunderAgent
cd ThunderAgent && pip install -e .

# Or via uv extras
uv sync --extra thunderagent
```

### Running

Requires the new HTTP inference layer (`_SKYRL_USE_NEW_INFERENCE=1`).

**With external vLLM servers:**
```bash
_SKYRL_USE_NEW_INFERENCE=1 uv run --extra fsdp --extra thunderagent \
  -m examples.train.thunder_agent.main_thunder_agent \
  generator.inference_engine.external_server_urls="['http://localhost:8000']" \
  generator.inference_engine.thunder_agent_mode=tr \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct"
```

**With internally built servers (colocated):**
```bash
_SKYRL_USE_NEW_INFERENCE=1 uv run --extra fsdp --extra thunderagent \
  -m examples.train.thunder_agent.main_thunder_agent \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  generator.inference_engine.thunder_agent_mode=tr
```

**Pure proxy mode (no capacity scheduling):**
```bash
_SKYRL_USE_NEW_INFERENCE=1 uv run --extra fsdp --extra thunderagent \
  -m examples.train.thunder_agent.main_thunder_agent \
  generator.inference_engine.external_server_urls="['http://localhost:8000']" \
  generator.inference_engine.thunder_agent_mode=default \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct"
```

## Tests

### Tests Passed (CPU, no GPU required)

All 11 tests in `tests/backends/skyrl_train/inference_servers/test_thunder_agent_router.py` pass:

| Test | What It Validates |
|------|------------------|
| `test_chat_completions_proxied` | `/v1/chat/completions` routes through ThunderAgent, reaches backend, returns correct usage stats |
| `test_catch_all_proxy` | Non-scheduled endpoints (`/tokenize`) proxy directly to backends via catch-all |
| `test_round_robin` | Catch-all distributes requests across both mock servers |
| `test_session_affinity` | `X-Session-ID` header routes consistently to same backend |
| `test_inference_generate_tracked` | `/inference/v1/generate` routes through ThunderAgent program tracking, creates program entry |
| `test_session_id_as_program_id` | `X-Session-ID` header is used as `program_id` fallback when no explicit `program_id` in body |
| `test_programs_endpoint` | `/programs` returns ThunderAgent program state after a chat completion |
| `test_health` | `/health` returns healthy status with program stats and backend list |
| `test_servers` | `/servers` returns all backend URLs |
| `test_chat_completions_error_clears_reasoning` | After a completed request, program transitions to ACTING (not stuck in REASONING) |
| `test_start_shutdown_lifecycle` | Router starts, serves health checks, and shuts down cleanly |

**Regression check:** Existing `test_router.py` (3 tests for `InferenceRouter`) also passes with no changes.

```bash
# Run ThunderAgent router tests
PYTHONPATH="./ThunderAgent:$PYTHONPATH" python3 -m pytest -v -s \
  tests/backends/skyrl_train/inference_servers/test_thunder_agent_router.py

# Run existing router tests (regression check)
PYTHONPATH="./ThunderAgent:$PYTHONPATH" python3 -m pytest -v -s \
  tests/backends/skyrl_train/inference_servers/test_router.py
```

### Tests Still Needed for Formal PR

The following tests require GPU hardware and/or full SkyRL dependencies that aren't available in the current dev environment:

#### 1. Config Integration Tests (CPU)
```bash
# Verify ThunderAgentConfig parses CLI overrides correctly
uv run --isolated --extra skyrl-train --extra dev pytest tests/train/ tests/backends/skyrl_train/ \
  --ignore=tests/backends/skyrl_train/gpu
```
Validates that the custom `ThunderAgentInferenceEngineConfig` fields work with `make_config()` + `from_cli_overrides()`.

#### 2. End-to-End Smoke Test with Mock vLLM (CPU/GPU)
```bash
# Start a mock OpenAI-compatible server, then:
_SKYRL_USE_NEW_INFERENCE=1 uv run --extra fsdp --extra thunderagent \
  -m examples.train.thunder_agent.main_thunder_agent \
  generator.inference_engine.external_server_urls="['http://localhost:8000']" \
  generator.inference_engine.thunder_agent_mode=tr \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct"
```
Validates the full training loop starts with ThunderAgentRouter as the inference proxy.

#### 3. GPU CI Tests (8xH100/A100)
```bash
# FSDP backend with ThunderAgent
_SKYRL_USE_NEW_INFERENCE=1 uv run --isolated --extra dev --extra fsdp --extra thunderagent \
  pytest -s tests/backends/skyrl_train/gpu/gpu_ci -m "not (integrations or megatron)"
```
Validates that ThunderAgent doesn't interfere with the GPU training pipeline.

#### 4. Capacity Scheduling Test (multi-backend, GPU)
Run with `thunder_agent_mode=tr` and multiple vLLM backends to validate:
- Programs are tracked (REASONING → ACTING transitions)
- Pause/resume works under capacity pressure
- BFD bin-packing distributes programs correctly across backends

#### 5. Lint & Format
```bash
bash format.sh
```
Already verified locally (ruff + black pass). Gitleaks requires `go` which isn't installed in this environment — CI will catch this.

## Key Design Decisions

1. **Extension pattern over core modification** — Following the dev guide, all integration code lives in `examples/train/thunder_agent/` using `BasePPOExp` hooks. No changes to `config.py`, `legacy.py`, `ppo_base_config.yaml`, or `main_base.py`.

2. **ThunderAgentRouter has same interface as InferenceRouter** — `start() -> str` and `shutdown() -> None`. This makes it a drop-in replacement inside `_get_new_inference_client()`.

3. **Lifespan context manager over `on_event`** — Uses FastAPI's `lifespan` parameter instead of deprecated `@app.on_event("startup")`/`@app.on_event("shutdown")` decorators to call `ta_router.start()`/`stop()` within the uvicorn event loop.

4. **Guard import with `THUNDERAGENT_AVAILABLE`** — ThunderAgent is optional. If not installed, `ThunderAgentRouter.__init__()` raises a clear `ImportError`.

5. **Config via `make_config(generator_cls=...)`** — ThunderAgent fields are added via `ThunderAgentInferenceEngineConfig(InferenceEngineConfig)` → `ThunderAgentGeneratorConfig(GeneratorConfig)` → `make_config()`, keeping them accessible as CLI overrides without touching core config.
