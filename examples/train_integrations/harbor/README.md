## Harbor Integration

Current source of truth for the Harbor benchmark path is:

1. `docs/agent-handoff/README.md`
2. `docs/agent-handoff/workstreams/harbor-ta-benchmark.run.yaml`
3. `examples/train_integrations/harbor/run_harbor_benchmark.sh`

The current benchmark contract is the cross-job R2EGYM path. Do not use legacy
one-off runbooks or old replay wrappers as execution source of truth.

Key files that are still part of the active path:

- `dataset.py`
- `harbor_generator.py`
- `run_harbor_fully_async.sh`
- `start_harbor_rollout_servers.sh`
- `ops/apply_harbor_runtime_patches.py`
- `validation/run_harbor_docker_concurrency_smoke.sh`
- `ops/wait_harbor_driver_until_terminal.py`
