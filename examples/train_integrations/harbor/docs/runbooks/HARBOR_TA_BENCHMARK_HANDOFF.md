# Harbor TA Benchmark Handoff

This document is the minimum orientation handoff for the current Harbor benchmark.

Goal:
- continue the current `R2EGYM + Harbor + fully-async + ThunderAgent` benchmark path
- using branch `ergt10/skyrl-ta-runtime-full`

If this document disagrees with the run spec or wrapper, use the run spec.

## 1. Branch And Scope

Use branch:
- `ergt10/skyrl-ta-runtime-full`

This branch already contains:
- vendored ThunderAgent
- SkyRL <-> ThunderAgent integration core
- Harbor integration code
- Harbor launchers, docs, and runtime assets
- repo-local Harbor runtime patch assets

Do not use an old runbook or a dirty workspace as the execution source of truth.

## 2. Read These Files First

Read in this order:

1. agent handoff rules:
- `docs/agent-handoff/README.md`

2. current benchmark execution contract:
- `docs/agent-handoff/workstreams/harbor-ta-benchmark.run.yaml`

3. current execution wrapper:
- `examples/train_integrations/harbor/run_harbor_benchmark.sh`

4. launcher called by the wrapper:
- `examples/train_integrations/harbor/run_harbor_fully_async.sh`

5. rollout launcher called by the wrapper:
- `examples/train_integrations/harbor/start_harbor_rollout_servers.sh`

6. Harbor runtime patcher:
- `examples/train_integrations/harbor/ops/apply_harbor_runtime_patches.py`

The current benchmark contract lives in the run spec and wrapper above.

## 3. Current Benchmark Shape

The current canonical path uses:
- 1 merged head/rollout node
- 4 trainer nodes

The current rollout layout is:
- 4 rollout servers
- ports: `18000,18001,18002,18003`
- tensor parallel size: `2`

The current benchmark default is:
- `DOCKER_MODE=rootful`
- stage order `cleanup-stage all -> prepare -> head -> ray -> rollout -> status -> driver`
- literal commands copied from `harbor-ta-benchmark.run.yaml`

Do not copy topology, allocation, or variant values from this document when the
run spec already provides them.

## 4. Required External Assets

These are not provided by the repo:

- Qwen3-32B model directory
- Harbor-formatted R2EGYM datasets
- usable Docker access on the merged/head node
- task Docker images if the selected cluster does not already have them cached

The actual values in use come from `shared_env`, `allocation`, and the selected
`variants.*.env` block in the run spec.

## 5. Environment Build

For the benchmark contract, prefer the existing runtime environment recorded in
the run spec:
- `PYTHON_BIN`
- `RAY_BIN`

If a new machine does not already have that environment, rebuild from:
- `pyproject.toml`
- `uv.lock`

Do not introduce a different ad-hoc environment and still call it the same benchmark.

## 6. Harbor Patch Handling

The wrapper is expected to handle runtime patch application through its normal
path. Manual patch application is only needed when validating or repairing the
runtime assets directly.

Manual patch commands:

```bash
python examples/train_integrations/harbor/ops/apply_harbor_runtime_patches.py --backup
```

This patcher installs the repo-local Harbor runtime assets used by the current
benchmark path, including the patched verifier, docker templates, and
mini-swe-agent runtime behavior.

To verify:

```bash
python examples/train_integrations/harbor/ops/apply_harbor_runtime_patches.py --check
```

## 7. Variables That Must Be Overridden On A New Cluster

Do not trust old wrapper defaults or historical cluster names. Use the run spec
as the source of truth for these:

- `MERGED_JOB_ID`
- `MERGED_NODE`
- `ROLLOUT_JOB_ID`
- `ROLLOUT_NODE`
- `TRAINER_NODE_SPECS`
- `WRAPPER`
- `TRAIN_DATA`
- `EVAL_DATA`
- `MODEL_PATH`
- `RUN_ARTIFACT_ROOT`
- `DOCKER_DATA_ROOT`
- `HARBOR_SHARED_UV_CACHE_HOST_DIR`
- `HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME`
- `HARBOR_SHARED_UV_PYTHON_HOST_DIR`

Also take run-shape values from the selected variant, not from memory:
- `RUN_NAME_OVERRIDE`
- `FULL_EPOCHS`
- `MAX_TRAIN_TASKS`
- `MAX_EVAL_TASKS`
- `EVAL_INTERVAL_STEPS`
- `HARBOR_AGENT_MAX_TURNS`
- `TRAINER_RESUME_MODE`
- `TRAINER_RESUME_PATH`

## 8. Fixed Runtime Assumptions To Preserve

Keep these only if the run spec still says so:

- `DOCKER_MODE=rootful`
- `DOCKER_INOTIFY_MAX_USER_INSTANCES=8192`
- `AGENT_TIMEOUT_SEC=9000`
- `MINI_SWE_MODEL_TIMEOUT_SEC=1200`
- `ROLLOUT_ENGINES=4`
- `ROLLOUT_TP_SIZE=2`

Do not treat `FULL_EPOCHS`, `MAX_TRAIN_TASKS`, `MAX_EVAL_TASKS`, or
`HARBOR_AGENT_MAX_TURNS` as fixed here. Those are variant-specific.

## 9. Required Execution Order

1. Use the benchmark environment recorded in the run spec, or rebuild it from the repo if needed
2. Export the benchmark env exactly as recorded in
   `docs/agent-handoff/workstreams/harbor-ta-benchmark.run.yaml`
3. Launch only through the canonical wrapper stages

For the current cross-job benchmark wrapper, keep the canonical stage order
`cleanup-stage all -> prepare -> head -> ray -> rollout -> status -> driver`.
That `driver` action is the durable path: it starts a detached local `srun`
client internally and then waits on
`examples/train_integrations/harbor/ops/wait_harbor_driver_until_terminal.py`.
The current benchmark default is system Docker on the merged/head node
(`DOCKER_MODE=rootful`), not rootless Docker.
Do not use `driver-detach` for the benchmark path.

## 10. What Usually Goes Wrong

- reading this handoff as if it were the execution contract
- reading old runbooks instead of the current run spec and wrapper
- synthesizing launch commands instead of copying the run spec and wrapper path
- reusing stale node names, data paths, or checkpoint paths
- assuming a variant value from memory instead of reading the selected `variants.*.env` block
- treating wrapper internals as permission to invent new outer launch commands

## 11. Minimal Launch Template

Use the run spec bootstrap commands and selected variant env as the starting
point. The minimal shell shape is:

```bash
cd <repo-root>
# copy bootstrap.commands from harbor-ta-benchmark.run.yaml
# apply the selected variants.*.env block
# then run only:
bash "$WRAPPER" cleanup-stage all
bash "$WRAPPER" prepare
bash "$WRAPPER" head
bash "$WRAPPER" ray
bash "$WRAPPER" rollout
bash "$WRAPPER" status
bash "$WRAPPER" driver
```

## 12. Success Criteria

Success is variant-specific and comes from the run spec.

For the benchmark as a whole:
- baseline and treatment must use the same wrapper path
- the intended semantic delta is whether ThunderAgent is disabled via `SKYRL_DISABLE_THUNDERAGENT`
- stage order and launch commands must remain literal copies from the run spec
