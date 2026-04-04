## Skills

Available skills:
- `skill-creator`: Guide for creating or updating skills. File: `/home/hkang/zthunder_yagent/.codex/skills/.system/skill-creator/SKILL.md`
- `skill-installer`: Install Codex skills from a curated list or a GitHub repo path. File: `/home/hkang/zthunder_yagent/.codex/skills/.system/skill-installer/SKILL.md`

Rules:
- Use a skill when the user names it or the task clearly matches its description.
- Open the listed `SKILL.md` and read only what you need.
- Resolve relative paths from the skill directory first.
- Prefer bundled `scripts/`, `assets/`, and templates over reimplementing them.
- If multiple skills apply, use the minimal set and say which ones you are using.
- If a named skill is missing or unreadable, say so briefly and continue with the best fallback.

## Canonical Harbor RL Start Path

Strict path means literal-command-only execution.

Allowed command sources for a Harbor RL full run:
- `docs/agent-handoff/workstreams/harbor-ta-benchmark.run.yaml` -> `bootstrap.commands`
- `docs/agent-handoff/workstreams/harbor-ta-benchmark.run.yaml` -> `variants.baseline.env` or `variants.thunderagent.env`
- `docs/agent-handoff/workstreams/harbor-ta-benchmark.run.yaml` -> `stages[*].run`
- `docs/agent-handoff/workstreams/harbor-ta-benchmark.run.yaml` -> `recovery.*`
- the wrapper script itself for what each stage command does internally

Required execution rule:
- Every command used to start, monitor-gate, or recover the train must be copied verbatim from the sources above.
- Do not synthesize new launch commands, wrapper actions, exports, or shortcuts at runtime.
- If the needed command is not already present in the run spec or wrapper, stop and update the document/script first. Do not improvise in the shell.

The only allowed full-run stage order is the literal sequence already recorded in `stages[*].run`:
1. `bash "$WRAPPER" cleanup-stage all`
2. `bash "$WRAPPER" prepare`
3. `bash "$WRAPPER" head`
4. `bash "$WRAPPER" ray`
5. `bash "$WRAPPER" rollout`
6. `bash "$WRAPPER" status`
7. `bash "$WRAPPER" driver`

The required base launch env is the literal set already recorded in `bootstrap.commands` and `shared_env`, including:
- `PYTHON_BIN=/data/zy/models/hkang/run_worktrees/skyrl-r2e-clean-20260321/.venv/bin/python`
- `RAY_BIN=/data/zy/models/hkang/run_worktrees/skyrl-r2e-clean-20260321/.venv/bin/ray`
- `RUN_ARTIFACT_ROOT=/scratch/triton_cache/$USER/harbor_run_artifacts`
- `CKPT_INTERVAL=-1`
- `HF_SAVE_INTERVAL=-1`
- `SKYRL_INFERENCE_ROUTER_PORT=18080`
- `DOCKER_MODE=rootful` for the current benchmark contract unless the run spec is explicitly changed again

Run-shape-specific values such as `RUN_NAME_OVERRIDE`, `FULL_EPOCHS`, `RUN_PREFLIGHT_CHECKS`, `AGENT_RUNTIME_PREFLIGHT`, and `RUN_HARBOR_DOCKER_CONCURRENCY_SMOKE_PREPARE` must come from the selected `variants.*.env` entry in the run spec. Do not override them ad hoc in the shell.

Forbidden for benchmark starts:
- any command not present in the run spec or wrapper
- `bash "$WRAPPER" all`
- `bash "$WRAPPER" driver-detach`
- changing launch order
- changing `RUN_TS`, `RUN_SHORT_ID`, or `RUN_NAME_OVERRIDE` mid-run

Recovery must stay stage-scoped and use only the literal `recovery` commands from the run spec.

## Harbor RL Runtime Checks

Use these non-skill checks for the full RL train flow in `docs/agent-handoff/workstreams/harbor-ta-benchmark.run.yaml`.

Shared paths:
- `LOG_DIR=/home/hkang/zthunder_yagent/tmp_logs/$RUN_NAME_OVERRIDE`
- `MONITOR_DIR=$LOG_DIR/monitoring`

Stage checks:
1. `prepare`
- Use wrapper exit status only.

2. `head`
- Use wrapper exit status only.

3. `ray`
- Use `bash "$WRAPPER" status` first.
- Then inspect `$LOG_DIR/launcher_ray.log`.

4. `rollout`
- Use `bash "$WRAPPER" status` first.
- Then inspect `$LOG_DIR/launcher_rollout.log` and `$LOG_DIR/rollout/rollout_*.log`.
- Use `$LOG_DIR/rollout/monitoring/vllm_metrics.tsv` only as supporting evidence.

5. `driver`
- The canonical `bash "$WRAPPER" driver` action self-detaches locally and waits on `examples/train_integrations/harbor/ops/wait_harbor_driver_until_terminal.py`.
- Use `bash "$WRAPPER" status` first.
- Then inspect `$LOG_DIR/launcher_train_driver.log`.
- Use `$MONITOR_DIR/trial_progress.tsv` as the progress truth for completed trials.
- If needed, inspect rollout logs and trainer-side GPU state manually.

Forbidden:
- do not use retired runtime probe skills for this workstream
