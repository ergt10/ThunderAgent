# Agent Handoff Rules

Read order:

1. the relevant `workstreams/*.run.yaml`
2. this file

File ownership:

- `workstreams/*.run.yaml`: one workstream's execution contract

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
rg -n "agent-handoff|run.yaml" docs/agent-handoff
```
