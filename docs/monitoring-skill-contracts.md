# Monitoring Skill Contracts

Strict input and output contracts for:
- `driver-phase-watch`
- `ray-cluster-watch`
- `trainer-gpu-watch`

The goal is to force each skill to return one authoritative conclusion instead of raw log summaries.

## Shared Contract

### Shared Input

All three skills must accept this common envelope:

```json
{
  "observation_mode": "offline | live",
  "format": "json | text",
  "run_name": "string",
  "log_dir": "/abs/path/to/tmp_logs/<run>",
  "time_budget_sec": 30
}
```

### Shared Output

`json` is authoritative. `text` is only a projection of the same result.

```json
{
  "skill": "driver-phase-watch | ray-cluster-watch | trainer-gpu-watch",
  "run_name": "string",
  "observation_mode": "offline | live",
  "authoritative": true,
  "generated_at_epoch": 1774182000,
  "overall_status": "skill-specific enum",
  "diagnosis": "non_empty_snake_case",
  "recommended_action": "skill-specific enum",
  "evidence": "one short decisive sentence",
  "probe_errors": ["string"],
  "payload": {}
}
```

### Shared Rules

- Unknown values must be explicit `null`, not omitted.
- `probe_errors` must always be present. Use `[]` when empty.
- `evidence` must be one short decisive statement, not pasted logs.
- `text` output must be a lossless summary of the `json` result.
- `authoritative=true` is allowed only when all required live probes succeed.
- `authoritative=false` must imply that missing probes or degraded observation are visible in output.

## driver-phase-watch

### Required Input

```json
{
  "observation_mode": "live",
  "format": "json | text",
  "run_name": "string",
  "log_dir": "/abs/path/to/tmp_logs/<run>",
  "head": {
    "hostname": "research-secure-06",
    "job_id": 28860
  },
  "ray": {
    "address": "172.27.23.41:6381"
  },
  "router": {
    "expected_port": 18080
  },
  "owned_process_matchers": [
    "run_name string",
    "main_harbor_thunder_agent_fully_async_head_pinned",
    "run_codecontest_qwen3_32b_6node_rootless_fully_async"
  ],
  "trial_progress_path": "/abs/path/to/monitoring/trial_progress.tsv"
}
```

### Input Rules

- In `live` mode, `head.hostname`, `head.job_id`, `ray.address`, `owned_process_matchers`, and `trial_progress_path` are required.
- If the run uses the new inference path, `router.expected_port` is required.
- `owned_process_matchers` must be non-empty, otherwise `authoritative=true` is forbidden.

### Output

```json
{
  "skill": "driver-phase-watch",
  "run_name": "string",
  "observation_mode": "live",
  "authoritative": true,
  "generated_at_epoch": 1774182000,
  "overall_status": "progressing | blocked | failed | completed | unknown",
  "diagnosis": "snake_case",
  "recommended_action": "wait | inspect_log | restart_owned_driver | restart_owned_ray_worker | restart_owned_rollout_engine | repair_env_then_restart | escalate_cluster_issue | stop_run",
  "evidence": "string",
  "probe_errors": [],
  "payload": {
    "phase": "not_started | launcher_started | preflight | placement_group_wait | inference_client_init | router_starting | router_ready | trainer_init | generation_active | training_active | completed | failed | unknown",
    "blocking_reason": "string | null",
    "driver_process": {
      "launcher_pid": 308649,
      "entrypoint_pid": 309087,
      "alive": true,
      "elapsed_sec": 30
    },
    "router": {
      "expected_port": 18080,
      "listening": false,
      "health": false,
      "pid": null
    },
    "ray_gate": {
      "alive_nodes": 5,
      "schedulable_trainer_nodes": 4,
      "required_trainer_nodes": 4,
      "placement_ready": true
    },
    "trial_progress": {
      "last_timestamp": "2026-03-22T05:01:56-0700",
      "trial_dirs": 0,
      "result_json_count": 0,
      "completed_trials_count": 0
    },
    "log_markers": {
      "last_phase_marker": "Initializing new inference client",
      "last_error_marker": null,
      "last_success_marker": null
    }
  }
}
```

### Output Rules

- `phase` must resolve to exactly one value.
- If `overall_status` is `blocked` or `failed`, `blocking_reason` must be non-null.
- If `driver_process.alive=false` and there is no completion marker, `overall_status=progressing` is forbidden.
- `router.health=false` must not be masked by historical ready markers.
- `trial_progress` staying at zero is evidence only. It is not by itself a failure verdict.

## ray-cluster-watch

### Required Input

```json
{
  "observation_mode": "live",
  "format": "json | text",
  "run_name": "string",
  "head": {
    "hostname": "research-secure-06",
    "job_id": 28860
  },
  "ray": {
    "address": "172.27.23.41:6381"
  },
  "expected_topology": {
    "head_node": "research-secure-06",
    "trainer_nodes": [
      {"hostname": "research-secure-18", "job_id": 28860},
      {"hostname": "research-secure-21", "job_id": 28860},
      {"hostname": "research-secure-30", "job_id": 28882},
      {"hostname": "research-secure-01", "job_id": 28837}
    ],
    "required_cpu_per_trainer": 8,
    "required_gpu_per_trainer": 8
  },
  "placement_requirement": {
    "bundles": [
      {"CPU": 8, "GPU": 8},
      {"CPU": 8, "GPU": 8},
      {"CPU": 8, "GPU": 8},
      {"CPU": 8, "GPU": 8}
    ],
    "distinct_hosts_required": true
  }
}
```

### Input Rules

- In `live` mode, `head`, `ray.address`, `expected_topology`, and `placement_requirement` are required.
- `expected_topology.trainer_nodes` must be non-empty.
- `placement_requirement.bundles` must match the expected trainer count.

### Output

```json
{
  "skill": "ray-cluster-watch",
  "run_name": "string",
  "observation_mode": "live",
  "authoritative": true,
  "generated_at_epoch": 1774182000,
  "overall_status": "ready | degraded | blocked | failed | unknown",
  "diagnosis": "snake_case",
  "recommended_action": "wait | inspect_log | restart_owned_ray_worker | restart_owned_ray_head | escalate_cluster_issue",
  "evidence": "string",
  "probe_errors": [],
  "payload": {
    "cluster_state": "head_missing | workers_missing | resources_missing | placement_pending | placement_ready | mismatched_topology | unknown",
    "head_reachable": true,
    "ray_connected": true,
    "cluster_resources": {
      "CPU": 712.0,
      "GPU": 32.0
    },
    "available_resources": {
      "CPU": 680.0,
      "GPU": 32.0
    },
    "nodes": [
      {
        "hostname": "research-secure-18",
        "node_ip": "172.27.31.235",
        "alive": true,
        "cpu_total": 176.0,
        "gpu_total": 8.0,
        "cpu_ok": true,
        "gpu_ok": true
      }
    ],
    "missing_nodes": [],
    "unexpected_nodes": [],
    "placement_group": {
      "allocatable": true,
      "required_bundles": [
        {"CPU": 8, "GPU": 8}
      ],
      "bundle_probe_results": [
        {
          "bundle_index": 0,
          "hostname": "research-secure-01",
          "node_ip": "172.27.18.27"
        }
      ],
      "distinct_bundle_hosts": [
        "172.27.18.27",
        "172.27.21.55",
        "172.27.23.81",
        "172.27.31.235"
      ]
    },
    "blocking_reason": null
  }
}
```

### Output Rules

- `overall_status=ready` is allowed only when:
  - `ray_connected=true`
  - all expected trainer nodes are alive
  - each expected trainer node meets CPU and GPU minimums
  - `placement_group.allocatable=true`
- `missing_nodes` and `unexpected_nodes` must always be arrays.
- `placement_group.bundle_probe_results` must have one item per required bundle.
- If a bundle probe fails, the item must still exist with `hostname=null` and `node_ip=null`.
- Wrapper PID files must not be treated as authoritative Ray truth.

## trainer-gpu-watch

### Required Input

```json
{
  "observation_mode": "live",
  "format": "json | text",
  "run_name": "string",
  "trainer_nodes": [
    {
      "hostname": "research-secure-18",
      "job_id": 28860,
      "expected_gpu_count": 8
    }
  ],
  "owned_process_matchers": [
    "run_name string",
    "FSDPPolicyWorker",
    "FSDPRefWorker",
    "main_harbor_thunder_agent_fully_async_head_pinned"
  ],
  "gpu_processes_tsv": "/abs/path/to/monitoring/gpu_processes.tsv"
}
```

### Input Rules

- `trainer_nodes` must be non-empty.
- `owned_process_matchers` must be non-empty, otherwise `authoritative=true` is forbidden.
- In `live` mode, every trainer node must support both `nvidia-smi` and `ps` probing.
- `gpu_processes_tsv` is optional and auxiliary only. It must not override live classification.

### Output

```json
{
  "skill": "trainer-gpu-watch",
  "run_name": "string",
  "observation_mode": "live",
  "authoritative": true,
  "generated_at_epoch": 1774182000,
  "overall_status": "clean | owned_active | foreign_contaminated | mixed | query_failed | unknown",
  "diagnosis": "snake_case",
  "recommended_action": "wait | inspect_log | clean_owned_processes | restart_owned_trainer | escalate_cluster_issue | stop_run",
  "evidence": "string",
  "probe_errors": [],
  "payload": {
    "nodes": [
      {
        "hostname": "research-secure-30",
        "job_id": 28882,
        "node_state": "clean | owned_active | foreign_contaminated | mixed | query_failed",
        "owned_gpu_process_count": 0,
        "foreign_gpu_process_count": 8,
        "query_error": null,
        "observed_gpu_processes": [
          {
            "pid": 93782,
            "process_name": "sglang::scheduler",
            "used_gpu_memory_mb": 69940,
            "owner": "foreign",
            "match_reason": "did_not_match_owned_process_matchers"
          }
        ]
      }
    ],
    "owned_nodes": [],
    "foreign_nodes": ["research-secure-30"],
    "clean_nodes": ["research-secure-18", "research-secure-21", "research-secure-01"],
    "blocking_reason": "foreign GPU consumers present on trainer nodes"
  }
}
```

### Output Rules

- `owner` must be exactly one of `owned | foreign | unknown`.
- `node_state=clean` only when there are no GPU compute processes on that node.
- `node_state=owned_active` only when all GPU compute processes are owned.
- `node_state=foreign_contaminated` only when all GPU compute processes are foreign.
- `node_state=mixed` only when both owned and foreign GPU compute processes exist.
- `overall_status=clean` only when every node is `clean`.
- The skill must never recommend killing foreign processes.
- Any foreign contamination must resolve to `escalate_cluster_issue`, not local cleanup.

## Acceptance Criteria

These three skills together must answer three distinct questions:
- `driver-phase-watch`: where is the driver blocked right now
- `ray-cluster-watch`: can the current Ray cluster actually host this run
- `trainer-gpu-watch`: who is currently using the trainer GPUs
