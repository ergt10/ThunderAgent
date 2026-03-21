# Full Run Timeline Analysis: r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-replay-20260319_035035

## Window

- launcher_start: 2026-03-19 03:57:07.090000-07:00
- launcher_end: 2026-03-19 04:54:29.248000-07:00
- cancel_time: 2026-03-19 05:05:11-07:00
- agent_timeout_count: 0
- thunder_wait_timeout_count: 0
- none_type_trajectory_id_count: 0

## Phase Table

| phase | step | start | end | duration_s | notes |
| --- | --- | --- | --- | ---: | --- |
| bring_up_and_preflight | 0 | 2026-03-19 03:57:07.090000-07:00 | 2026-03-19 04:10:00.669000-07:00 | 773.58 | monitor + ray/preflight + trainer bootstrap |
| wait_for_generation_buffer | 1 | 2026-03-19 04:10:00.669000-07:00 | 2026-03-19 04:40:33.445000-07:00 | 1832.78 | step=1 |
| run_training | 1 | 2026-03-19 04:40:34.124000-07:00 | 2026-03-19 04:42:29.966000-07:00 | 115.84 | step=1 |
| step_finalize | 1 | 2026-03-19 04:42:29.966000-07:00 | 2026-03-19 04:43:24.759000-07:00 | 54.79 | step=1 |

## Step Summary

| step | wait_s | train_s | finalize_s | global_step_after |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1832.78 | 115.84 | 54.79 | 1 |

## Trainer Peaks

| trainer_gpu | peak_reserved_gib |
| --- | ---: |
| research-dev-coder-012:gpu6 | 54.89 |
| research-dev-coder-013:gpu6 | 54.89 |
| research-dev-coder-013:gpu4 | 54.89 |
| research-dev-coder-013:gpu1 | 54.89 |
| research-dev-coder-013:gpu0 | 54.42 |
| research-dev-coder-012:gpu0 | 54.14 |
| research-dev-coder-012:gpu2 | 54.14 |
| research-dev-coder-012:gpu1 | 54.14 |
| research-dev-coder-013:gpu7 | 54.14 |
| research-dev-coder-012:gpu7 | 54.14 |
| research-dev-coder-012:gpu5 | 54.14 |
| research-dev-coder-012:gpu3 | 54.14 |
| research-dev-coder-013:gpu2 | 54.14 |
| research-dev-coder-014:gpu2 | 54.14 |
| research-dev-coder-014:gpu1 | 54.14 |
| research-dev-coder-014:gpu0 | 54.14 |
| research-dev-coder-015:gpu4 | 54.14 |
| research-dev-coder-015:gpu1 | 54.14 |
| research-dev-coder-015:gpu2 | 54.14 |
| research-dev-coder-014:gpu3 | 54.14 |
| research-dev-coder-014:gpu5 | 54.14 |
| research-dev-coder-014:gpu6 | 54.14 |
| research-dev-coder-014:gpu7 | 54.14 |
| research-dev-coder-015:gpu0 | 54.14 |
| research-dev-coder-015:gpu6 | 54.14 |
| research-dev-coder-015:gpu5 | 54.14 |
| research-dev-coder-015:gpu7 | 54.14 |
| research-dev-coder-015:gpu3 | 54.14 |
| research-dev-coder-013:gpu5 | 53.89 |
| research-dev-coder-013:gpu3 | 53.89 |
| research-dev-coder-012:gpu4 | 53.89 |
| research-dev-coder-014:gpu4 | 53.89 |

| trainer_node | peak_reserved_gib |
| --- | ---: |
| research-dev-coder-013 | 435.15 |
| research-dev-coder-012 | 433.64 |
| research-dev-coder-015 | 433.16 |
| research-dev-coder-014 | 432.90 |

## Rollout Peaks

| rollout_gpu | peak_used_gib |
| --- | ---: |
| research-dev-coder-008:gpu0 | 67.10 |
| research-dev-coder-008:gpu1 | 67.10 |
| research-dev-coder-008:gpu2 | 67.10 |
| research-dev-coder-008:gpu3 | 67.10 |
| research-dev-coder-008:gpu4 | 67.10 |
| research-dev-coder-008:gpu5 | 67.10 |
| research-dev-coder-008:gpu6 | 67.10 |
| research-dev-coder-008:gpu7 | 67.10 |

| rollout_backend | peak_kv_usage_pct |
| --- | ---: |
| rollout_b | 99.99 |
| rollout_d | 99.99 |
| rollout_c | 99.89 |
| rollout_a | 99.89 |

| rollout_backend | peak_prefix_cache_hit_rate_pct |
| --- | ---: |
| rollout_c | 96.56 |
| rollout_b | 94.72 |
| rollout_a | 93.03 |
| rollout_d | 90.42 |

| rollout_backend | peak_num_requests_running |
| --- | ---: |
| rollout_d | 69 |
| rollout_b | 50 |
| rollout_a | 48 |
| rollout_c | 41 |

| rollout_backend | peak_num_requests_waiting |
| --- | ---: |
| rollout_d | 9 |
| rollout_b | 6 |
| rollout_a | 5 |
| rollout_c | 3 |
