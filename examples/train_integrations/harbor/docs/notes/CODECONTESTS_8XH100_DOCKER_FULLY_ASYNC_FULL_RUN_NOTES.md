# CodeContests 8xH100 Docker Fully-Async Full Run Notes

日期：2026-03-07

## 1. 目标

本次记录对应 Harbor 接入 `FullyAsyncRayPPOTrainer` 后的第一轮真实 `full run`。

目标不是一次跑完整个 `1 epoch`，而是先回答下面几个问题：

- Harbor 这条多轮 agent + Docker sandbox + verifier 的重负载路径，能不能接上 SkyRL 的 `fully_async` trainer
- 训练时是否真的发生了 rollout / train overlap，而不是只开了 `async_engine=true`
- 在 `8 x H100`、`4 rollout + 4 train/ref` 拓扑下，能否稳定跑过多于 `20` step
- checkpoint / export / eval / buffer trace 是否能持续落盘
- fully-async 在 Harbor 场景下的真实瓶颈是什么

结论：

- 这轮 `fully_async full run` 是成功的
- Harbor fully-async 路径已连续跑到 `global_step=71`
- 本轮不是异常退出，而是按人工要求在“看到 step70 后停止”
- 由于轮询粒度，实际最后一个 `dequeue_batch` 落在 `global_step=71`
- 没有出现训练侧 OOM，也没有出现 vLLM engine core crash
- 但 `buffer` 基本长期被 trainer 吃空，主要瓶颈仍然是 Harbor rollout 长尾，而不是训练吞吐

## 2. 使用环境

本次 full run 使用环境：

- 工作区：`/home/hkang/zthunder_yagent/SkyRL`
- `HOME`：`/home/hkang/zthunder_yagent`
- Python 环境：`/home/hkang/zthunder_yagent/SkyRL/.venv`
- Python 版本：`3.13.11`
- Torch 版本：`2.8.0+cu128`
- vLLM 版本：`0.10.2`
- 共享模型目录：`/data/zy/models`
- HF cache：`/data/zy/models/hub`
- 训练数据：
  - `/home/hkang/zthunder_yagent/data/harbor/CodeContests`
  - `/home/hkang/zthunder_yagent/data/harbor/OpenThoughts-TB-dev`
- 运行产物根目录：`/scratch/hkang/codecontest-8xh100-docker-fully-async-full-r1`
- 日志目录：`/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-fully-async-full-r1`
- monitoring 目录：`/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-fully-async-full-r1/monitoring`

本次 fully-async 路径依赖这几处代码：

- 入口：
  - [main_harbor_fully_async.py](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/entrypoints/main_harbor_fully_async.py)
- 启动脚本：
  - [run_codecontest_8xh100_docker_fully_async.sh](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/run_codecontest_8xh100_docker_fully_async.sh)
- fully-async trainer trace：
  - [fully_async_trainer.py](/home/hkang/zthunder_yagent/SkyRL/skyrl/train/fully_async_trainer.py)
- trace 汇总脚本：
  - [plot_fully_async_buffer_trace.py](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/plot_fully_async_buffer_trace.py)

## 3. 启动命令

实际启动命令：

```bash
cd /home/hkang/zthunder_yagent/SkyRL

RUN_NAME_OVERRIDE=codecontest-8xh100-docker-fully-async-full-r1 \
bash examples/train_integrations/harbor/run_codecontest_8xh100_docker_fully_async.sh full \
  trainer.resume_mode=none \
  trainer.ckpt_interval=20 \
  trainer.hf_save_interval=20
```

停止方式：

- 在看到 `step70` 后按要求停止
- 实际最终 `trace` 停在 `global_step=71`
- 原因是 watcher 按分钟轮询，不是逐 event 中断

## 4. 实际使用配置

### 4.1 拓扑

- 单机 `8 x H100`
- `trainer.placement.colocate_all=false`
- `trainer.placement.colocate_policy_ref=true`
- rollout 池：`GPU 0-3`
- train/ref 池：`GPU 4-7`
- `trainer.strategy=fsdp2`
- `trainer.critic.model.path=null`
- `trainer.algorithm.use_kl_loss=true`

### 4.2 fully-async 参数

- `trainer.fully_async.max_staleness_steps=1`
- `trainer.fully_async.num_parallel_generation_workers=4`
- `generator.inference_engine.async_engine=true`
- `generator.batched=false`
- `trainer.train_batch_size=4`
- `trainer.policy_mini_batch_size=4`
- `trainer.micro_forward_batch_size_per_gpu=1`
- `trainer.micro_train_batch_size_per_gpu=1`
- `generator.n_samples_per_prompt=2`
- `generator.eval_n_samples_per_prompt=1`
- `trainer.algorithm.max_seq_len=6144`

### 4.3 Harbor / rollout 参数

- `harbor_trial_config.environment.type=docker`
- `harbor_trial_config.environment.override_cpus=2`
- `harbor_trial_config.environment.override_memory_mb=4096`
- `harbor_trial_config.environment.override_storage_mb=4096`
- `harbor_trial_config.agent.override_timeout_sec=900`
- `harbor_trial_config.agent.kwargs.max_turns=10`
- `harbor_trial_config.agent.kwargs.enable_summarize=false`
- `harbor_trial_config.agent.kwargs.store_all_messages=true`
- `harbor_trial_config.agent.kwargs.temperature=0.3`
- `harbor_trial_config.agent.kwargs.llm_call_kwargs.extra_body.chat_template_kwargs.enable_thinking=false`
- `harbor_trial_config.agent.kwargs.llm_call_kwargs.extra_body.include_reasoning=false`
- `harbor_trial_config.agent.kwargs.model_info.max_input_tokens=32768`
- `harbor_trial_config.agent.kwargs.model_info.max_output_tokens=32768`

### 4.4 vLLM / rate limit 参数

- `generator.inference_engine.num_engines=4`
- `generator.inference_engine.tensor_parallel_size=1`
- `generator.inference_engine.run_engines_locally=true`
- `generator.inference_engine.backend=vllm`
- `generator.inference_engine.gpu_memory_utilization=0.8`
- `generator.inference_engine.weight_sync_backend=nccl`
- `generator.inference_engine.enforce_eager=true`
- `generator.inference_engine.enable_http_endpoint=true`
- `generator.inference_engine.http_endpoint_host=127.0.0.1`
- `generator.inference_engine.http_endpoint_port=8000`
- `generator.inference_engine.engine_init_kwargs.max_model_len=32768`
- `generator.rate_limit.enabled=true`
- `generator.rate_limit.trajectories_per_second=2`
- `generator.rate_limit.max_concurrency=8`

### 4.5 日志与保存

- `trainer.logger=['tensorboard','console']`
- `trainer.collect_memory_metrics=true`
- `trainer.collect_memory_metrics_interval=1`
- `trainer.ckpt_interval=20`
- `trainer.hf_save_interval=20`

## 5. 验收结果

| 验收项 | 结果 | 说明 |
| --- | --- | --- |
| Harbor fully-async 真实跑通 | 达成 | 连续训练到 `global_step=71` |
| rollout / train overlap 真实发生 | 达成 | 有持续 `enqueue_group` / `dequeue_batch` / staleness trace |
| 训练侧 OOM | 未出现 | 跑到 `71` step 没有训练 OOM |
| vLLM engine core crash | 未出现 | 有 overlong request，但没有打崩 engine |
| checkpoint 正常 | 达成 | `global_step_20/40/60` 已写出 |
| export 正常 | 达成 | `global_step_20/40/60` 已写出 |
| eval 导出 | 达成 | `exports/dumped_evals` 存在 |
| 人工停止前保持活跃 | 达成 | 主进程和 GPU 直到停止前都仍活跃 |

## 6. 最终统计

最终 trace 汇总：

- `max_global_step_in_trace = 71`
- `num_dequeue_batches = 71`
- `max_staleness_seen = 5`
- `num_groups_over_budget = 39`
- `last_dequeue.global_step = 71`
- `last_dequeue.staleness_values = [3, 1, 0, 0]`
- `last_dequeue.total_samples = 8`
- `last_dequeue.total_response_tokens = 19909`

产物计数：

- `trials_run/result.json = 592`

checkpoint：

- [global_step_20](/scratch/hkang/codecontest-8xh100-docker-fully-async-full-r1/ckpts/global_step_20)
- [global_step_40](/scratch/hkang/codecontest-8xh100-docker-fully-async-full-r1/ckpts/global_step_40)
- [global_step_60](/scratch/hkang/codecontest-8xh100-docker-fully-async-full-r1/ckpts/global_step_60)

exports：

- [global_step_20](/scratch/hkang/codecontest-8xh100-docker-fully-async-full-r1/exports/global_step_20)
- [global_step_40](/scratch/hkang/codecontest-8xh100-docker-fully-async-full-r1/exports/global_step_40)
- [global_step_60](/scratch/hkang/codecontest-8xh100-docker-fully-async-full-r1/exports/global_step_60)
- [dumped_evals](/scratch/hkang/codecontest-8xh100-docker-fully-async-full-r1/exports/dumped_evals)

## 7. 关键观察

### 7.1 fully-async 是真的，不是伪 async

这轮最核心的结论是：Harbor 这条路径已经不是“只开了 async_engine 的同步 trainer”。

证据：

- trace 里持续存在：
  - `buffer_wait_start`
  - `worker_slot_acquired`
  - `enqueue_group`
  - `dequeue_batch`
- `dequeue_batch` 带有明确的：
  - `worker_ids`
  - `uids`
  - `staleness_values`
  - `buffer_qsize_after`
  - `total_response_tokens`
- 训练和 rollout 的 GPU 使用表现也是交替重叠的：
  - rollout 卡长期常驻在 `~67 GiB`
  - 训练卡在 consume batch 时周期性拉到 `30-74 GiB`

### 7.2 Harbor fully-async 的主瓶颈不是训练，而是长尾 rollout

这轮里最重要的系统特征不是 OOM，而是：

- buffer 经常没有积压
- trainer 几乎总是在“刚等够 4 个 groups 就立刻取走”
- `buffer_qsize_after` 很多时候回到 `0`
- rollout worker 时长差异大，经常有 1-2 个 worker 把整个 batch 拖慢

这说明：

- fully-async 在 Harbor 场景下当然能 overlap
- 但它没有把系统变成“rollout 永远提前很多”
- 当前更像是“轻度流水线 + 极强长尾约束”

### 7.3 staleness manager 在 Harbor 上不是严格 per-group 上限

当前配置：

- `trainer.fully_async.max_staleness_steps=1`

但这轮真实观测到：

- `max_staleness_seen = 5`
- `num_groups_over_budget = 39`

这说明：

- staleness manager 确实在控制整体 ahead 程度
- 但在 Harbor 这种长尾任务上，不应把它理解成“每个 group 永远不超过 1 step stale”

### 7.4 overlong request 仍在出现，但稳定性比之前好

这轮至少出现过多次 overlong：

- `32854 > 32768`
- `32850 > 32768`
- `32854 > 32768` 再次出现

但和之前同步 full run 的失败路径不同，这轮里：

- 这些 overlong 请求没有把 engine core 打崩
- 主流程在报错后仍能继续推进到后续 step

这说明当前 Harbor / vLLM 路径虽然没有从根上消灭 overlong，但鲁棒性已经明显好于之前的 `vllm==0.16` 路线。

## 8. 典型长尾案例

`step 54` 是一段很有代表性的长尾区间：

- `buffer_wait_start` 之后很长时间没有凑够一批
- trace 中几乎只有 `worker 0` 连续交活
- 其他 worker 长时间没有形成新的 `enqueue_group`
- `buffer_qsize_after` 长时间只有 `1`

这段说明：

- Harbor fully-async 的真实瓶颈并不是 trainer 消费慢
- 而是某些 task / sample 的 rollout 时间极不均匀
- 在这种情况下，就算 trainer 是 fully async，buffer 也攒不深

## 9. 可复用经验

### 9.1 当前这组参数是 Harbor fully-async 的第一版可接受起跑线

这轮表明下面这组参数是能稳定跑起来的：

- `train_batch_size=4`
- `policy_mini_batch_size=4`
- `n_samples_per_prompt=2`
- `max_seq_len=6144`
- `max_staleness_steps=1`
- `num_parallel_generation_workers=4`
- `max_turns=10`
- `temperature=0.3`
- `enable_thinking=false`
- `include_reasoning=false`
- `enforce_eager=true`
- `flash_attn=false`
- `use_sample_packing=false`

### 9.2 fully-async 在 Harbor 上的第一优先级不是继续加 worker，而是先控制长尾

从这轮结果看，直接把 `num_parallel_generation_workers` 加大，不一定先带来吞吐收益。

更值得优先做的是：

- 控制 Harbor agent 的上下文增长
- 限制极长 task 的拖尾时间
- 让 overlong 更早被前置拦截
- 让 buffer 更稳定地积到 `>1 batch`

### 9.3 `/scratch` 路径设置是正确的

本轮 checkpoint/export 全在 `/scratch`，没有再把 `/home` 写满。

这点必须保留，不要回退。

## 10. 关键产物路径

主运行目录：

- [codecontest-8xh100-docker-fully-async-full-r1](/scratch/hkang/codecontest-8xh100-docker-fully-async-full-r1)

主日志：

- [infra-260307_194242.log](/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-fully-async-full-r1/infra-260307_194242.log)

原始 trace：

- [async_trace.jsonl](/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-fully-async-full-r1/monitoring/async_trace.jsonl)

最终汇总图：

- [async_buffer_summary_final.png](/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-fully-async-full-r1/monitoring/async_buffer_summary_final.png)
- [async_buffer_summary_final.svg](/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-fully-async-full-r1/monitoring/async_buffer_summary_final.svg)

最终聚合数据：

- [buffer_qsize.csv](/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-fully-async-full-r1/monitoring/async_trace_aggregates_final/buffer_qsize.csv)
- [rollout_worker_volume.csv](/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-fully-async-full-r1/monitoring/async_trace_aggregates_final/rollout_worker_volume.csv)
- [trainer_consumption.csv](/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-fully-async-full-r1/monitoring/async_trace_aggregates_final/trainer_consumption.csv)

## 11. 复现最终图

```bash
cd /home/hkang/zthunder_yagent/SkyRL

python examples/train_integrations/harbor/plot_fully_async_buffer_trace.py \
  --trace /home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-fully-async-full-r1/monitoring/async_trace.jsonl \
  --output /home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-fully-async-full-r1/monitoring/async_buffer_summary_final.png \
  --csv-dir /home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-fully-async-full-r1/monitoring/async_trace_aggregates_final
```

## 12. 当前结论

对 Harbor 来说，这轮 fully-async full run 已经回答了最关键的问题：

- `FullyAsyncRayPPOTrainer` 可以接上 Harbor
- 可以稳定跑过多步训练，不是一次性 demo
- 真正的系统瓶颈仍然是 Harbor rollout 长尾，而不是训练侧算力
- 下一阶段如果要继续优化，重点不该只是“再加 async”，而是要让 `buffer` 真正积起来
