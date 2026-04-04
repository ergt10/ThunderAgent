# Agent Handoff Rules

Read order:

1. `project.latest.yaml`
2. the relevant `workstreams/*.run.yaml`
3. this file

File ownership:

- `project.latest.yaml`: objective, phase/status, constraints, top-level workstreams, top 3 next actions, evidence
- `workstreams/*.run.yaml`: one workstream's execution contract

Update `project.latest.yaml` only when these change:

- objective
- success criteria
- phase/status
- top-level workstreams
- invariants / forbidden actions
- next actions
- source of truth

Update `workstreams/*.run.yaml` only when these change:

- allocation
- required env
- baseline/treatment definition
- stage order
- monitoring contract
- post-run analysis contract
- recovery contract
- comparison checklist

Do not edit handoff files for a run failure unless the contract changed.

Hard limits:

- `project.latest.yaml`: `summary` is 3-5 lines
- `project.latest.yaml`: `next_actions` max 3
- `project.latest.yaml`: high-impact claims need evidence
- `project.latest.yaml`: facts must be `fact`, `inference`, or `stale`
- `workstreams/*.run.yaml`: commands must be copyable
- `workstreams/*.run.yaml`: baseline/treatment delta must be explicit
- `workstreams/*.run.yaml`: monitoring outputs must be named files
- `workstreams/*.run.yaml`: recovery must be stage-scoped

Forbidden here:

- raw logs
- shell transcripts
- run history
- debugging diary
- temporary notes
- stale workarounds

Put those in runbooks, notes, or analysis output dirs.

Add a new `workstreams/<name>.run.yaml` only if all are true:

- the workstream persists across multiple runs
- it has its own execution contract
- a new agent cannot safely continue without reading it

Validation after edit:

```bash
cd /home/hkang/zthunder_yagent/SkyRL
python3 - <<'PY'
import yaml
paths = [
    "docs/agent-handoff/project.latest.yaml",
    "docs/agent-handoff/workstreams/harbor-ta-benchmark.run.yaml",
]
for path in paths:
    with open(path) as f:
        yaml.safe_load(f)
    print("YAML_OK", path)
PY
```

If references changed:

```bash
cd /home/hkang/zthunder_yagent/SkyRL
rg -n "agent-handoff|project.latest.yaml|run.yaml" docs/agent-handoff
```
