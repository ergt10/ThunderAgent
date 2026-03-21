# CodeContests 8xH100 Docker Fully-Async 20-Step Test Notes

日期：2026-03-08

## 1. 目标

这份记录对应 Harbor fully-async 路径的一次定长 `20 step` 训练验收。

这次测试要回答的问题不是“能不能长跑到 70+ step”，而是：

- 在 `examples/train_integrations/harbor` 当前修复后的代码上，能不能稳定复现一次完整的 `20 step` fully-async 训练
- `global_step_20` 的 checkpoint / export 能不能真实落盘
- 在 Harbor 多轮 agent + Docker sandbox + verifier 这条重路径上，当前最真实的失败模式是什么
- 前一轮 full run / smoke / r5 诊断里暴露出来的经验教训，哪些必须体现在复现参数里

结论：

- `20 step` 验收已完成
- 推荐以 `r6` 作为复现实例，而不是 `r5`
- `global_step_20` 的 checkpoint 和 HF export 都已落盘
- `r6` 里出现过的 `Harbor rollout details 偶发不完整`，在当前本地 Harbor 根修后的 `r7` 同配置 fresh rerun 里未复现
- 当前仍明确存在的问题是：
  - `agent_timeout`
  - staleness 偶发超过 `max_staleness_steps=1`
  - task-level protocol error（`Invalid JSON: Invalid \\escape`）

## 2. 使用环境

本次测试环境：

- 工作区：`/home/hkang/zthunder_yagent/SkyRL`
- `HOME`：`/home/hkang/zthunder_yagent`
- Python 环境：`/home/hkang/zthunder_yagent/SkyRL/.venv`
- Python 版本：`3.13.11`
- Torch 版本：`2.8.0+cu128`
- vLLM 版本：`0.10.2`
- 共享模型目录：`/data/zy/models`
- HF cache：`/data/zy/models/hub`
- Harbor 数据目录：
  - `/home/hkang/zthunder_yagent/data/harbor/CodeContests`
  - `/home/hkang/zthunder_yagent/data/harbor/OpenThoughts-TB-dev`
- 运行时 scratch：
  - `/scratch/hkang/skyrl_runtime`
- 运行产物根目录：
  - `/scratch/hkang/codecontest-8xh100-docker-fully-async-20step-r6`
- 日志目录：
  - `/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-fully-async-20step-r6`

本次使用的关键代码路径：

- 入口：
  - [main_harbor_fully_async.py](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/entrypoints/main_harbor_fully_async.py)
- 启动脚本：
  - [run_codecontest_8xh100_docker_fully_async.sh](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/run_codecontest_8xh100_docker_fully_async.sh)
- Harbor 生成器：
  - [harbor_generator.py](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/harbor_generator.py)
- Harbor 数据集：
  - [dataset.py](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/dataset.py)
- fully-async trainer：
  - [fully_async_trainer.py](/home/hkang/zthunder_yagent/SkyRL/skyrl/train/fully_async_trainer.py)

## 3. 这轮测试依赖的前置修复

在做这次 20-step 验收前，Harbor 集成已经有下面这些修复：

1. Harbor 生成器会把 SkyRL 的 `sampling_params` 传进 Harbor trial config。
2. 当 `logprobs` 被请求时，会打开 `collect_rollout_details`，并把 `rollout_logprobs` 真正回传给训练器。
3. Harbor rollout details 的 token 对齐不再依赖重新 tokenize assistant 文本，而是优先使用 Harbor 返回的 `completion_token_ids`。
4. Harbor dataset 的 UID 已改成稳定路径 UID，resume 不再依赖不稳定的目录索引。
5. 本地 Harbor runtime 已在 `OutputLengthExceededError` / overflow-retry 分支里把截断 turn 的 `completion_token_ids / logprobs` 一并保留下来，不再只把截断 assistant 文本塞回消息历史。

这意味着本次测试不是在“原始 broken 集成”上跑的，而是在修复后的基线上做验收。

## 4. 为什么不用 r5 做验收样本

在最终的 `r6` 之前，我先做过一轮更接近默认脚本参数的 `r5`：

- `train_batch_size=2`
- `policy_mini_batch_size=2`
- `max_train_tasks=40`
- `num_parallel_generation_workers=2`
- 但仍保留：
  - `max_turns=10`
  - `agent.override_timeout_sec=900`
  - `max_generate_length=1024`

`r5` 的结果是：

- 虽然跑到了 `global_step=13`
- 但出现了明显不适合作为 20-step 验收样本的问题：
  - `Context length exceeded and summarization is OFF`
  - 长尾 task 把单个 step 拖到 `275s`
  - Harbor rollout details 缺失仍偶发出现
  - staleness 多次高于预算

其中最关键的经验教训是：

- Harbor fully-async 的第一优先级不是“继续加 worker”
- 而是先控制 agent 上下文增长，避免长尾把 generation buffer 卡死

## 5. 最终采用的 r6 复现命令

最终推荐的 `20 step` 复现命令如下：

```bash
cd /home/hkang/zthunder_yagent/SkyRL

RUN_NAME_OVERRIDE=codecontest-8xh100-docker-fully-async-20step-r6 \
RUN_ARTIFACT_ROOT=/scratch/$USER \
SCRATCH_ROOT=/scratch/$USER/skyrl_runtime \
bash examples/train_integrations/harbor/run_codecontest_8xh100_docker_fully_async.sh smoke \
  max_train_tasks=40 \
  max_eval_tasks=null \
  trainer.resume_mode=none \
  trainer.eval_interval=0 \
  trainer.ckpt_interval=20 \
  trainer.hf_save_interval=20 \
  trainer.logger="['tensorboard','console']" \
  trainer.collect_memory_metrics=false \
  harbor_trial_config.agent.override_timeout_sec=240 \
  harbor_trial_config.agent.kwargs.max_turns=6 \
  generator.sampling_params.max_generate_length=768 \
  generator.eval_sampling_params.max_generate_length=768
```

为什么基于 `smoke` stage 而不是 `pilot`：

- `smoke` 默认就是：
  - `train_batch_size=2`
  - `policy_mini_batch_size=2`
  - `num_parallel_generation_workers=2`
  - `rate_limit.max_concurrency=4`
- 对 `20 step` 验收更轻，更容易把问题聚焦在 Harbor 路径本身，而不是过重拓扑带来的长尾

## 6. r6 的实际生效配置

这轮 `r6` 日志中确认生效的关键参数：

- `max_train_tasks=40`
- `train_batch_size=2`
- `policy_mini_batch_size=2`
- `num_parallel_generation_workers=2`
- `generator.n_samples_per_prompt=2`
- `generator.sampling_params.temperature=0.3`
- `generator.sampling_params.logprobs=1`
- `generator.sampling_params.max_generate_length=768`
- `harbor_trial_config.agent.override_timeout_sec=240`
- `harbor_trial_config.agent.kwargs.max_turns=6`
- `harbor_trial_config.agent.kwargs.enable_summarize=false`
- `trainer.fully_async.max_staleness_steps=1`
- `trainer.algorithm.max_seq_len=6144`

训练器日志明确打印了：

- `Total steps: 20`
- `Number of steps per epoch: 20`
- `Total training steps: 20`

## 7. 最终结果

### 7.1 训练是否完成

完成。

关键里程碑：

- `global_step=20` 已到达
- `step 20` 的训练指标已写出
- `global_step_20` 的 checkpoint 已成功保存
- `global_step_20` 的 HF export 已成功保存

另外还观察到一个实现细节：

- 训练器在 `step 20` 之后还会做一次“final checkpoint / final hf save”
- 因此最终目录里还出现了：
  - `ckpts/global_step_21`
  - `exports/global_step_21`

但本次验收目标是 `global_step_20`，所以验收基准仍然以 `global_step_20` 为准。

### 7.2 关键产物

主运行目录：

- [codecontest-8xh100-docker-fully-async-20step-r6](/scratch/hkang/codecontest-8xh100-docker-fully-async-20step-r6)

checkpoint：

- [global_step_20](/scratch/hkang/codecontest-8xh100-docker-fully-async-20step-r6/ckpts/global_step_20)
- [global_step_21](/scratch/hkang/codecontest-8xh100-docker-fully-async-20step-r6/ckpts/global_step_21)

HF export：

- [global_step_20](/scratch/hkang/codecontest-8xh100-docker-fully-async-20step-r6/exports/global_step_20)
- [global_step_21](/scratch/hkang/codecontest-8xh100-docker-fully-async-20step-r6/exports/global_step_21)

主日志：

- [infra-260308_004822.log](/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-fully-async-20step-r6/infra-260308_004822.log)
- [worker-...1681283.err](/scratch/hkang/skyrl_runtime/ray_tmp/ray/session_latest/logs/worker-7fe1368c9b83f4c6d42f5007a0dc4ae962dd27fce93393e5db058de1-01000000-1681283.err)

TensorBoard：

- [events.out.tfevents...](/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-fully-async-20step-r6/tensorboard/events.out.tfevents.1772959752.research-secure-10.cloud.together.ai.1681283.0)

### 7.3 统计

本轮一些简单统计：

- `trials_run/result.json = 80`
- `latest_ckpt_global_step = 21`

`step 20` 指标里最关键的几项：

- `trainer/global_step = 20`
- `reward/avg_pass_at_2 = 0.5000`
- `reward/avg_raw_reward = 0.5000`
- `policy/rollout_train_logprobs_abs_diff_mean = 1.0997`
- `async/staleness_max = 2`
- `async/staleness_violation_count = 1`

## 8. 这次 20-step 里真实出现的问题

虽然 `20 step` 跑完了，但下面这些问题仍然是真实存在的：

### 8.1 Staleness 仍会超预算

尽管配置是：

- `trainer.fully_async.max_staleness_steps=1`

实际日志里仍多次出现：

- `cur_staleness=2`
- `cur_staleness=3`

这和 full run 笔记里的结论一致：

- Harbor fully-async 上的 staleness manager 不应该被理解为“每个 group 永远不超过 1 step stale”

### 8.2 r6 中 Harbor rollout details 偶发不完整

本轮至少多次出现：

- `Missing completion token ids for assistant message #3. Provided 2 completion token id lists.`
- `Missing completion token ids for assistant message #7. Provided 6 completion token id lists.`

这些问题会导致：

- 单个 trajectory 被降级成 masked error
- 某些 prompt group 被整组 mask

当前好消息是：

- 这些问题不会再把整个 `TaskGroup` 打崩

但坏消息是：

- 这说明 Harbor runtime 返回的 rollout details 仍然不是严格完备的

这条结论只对应 `r6` 的历史结果，不再代表当前本地 Harbor runtime 的最新状态。最新复核见第 `12` 节。

### 8.3 Harbor agent timeout 仍偶发存在

本轮至少出现过：

- `code_contests-0020`
- `code_contests-0025`
- `code_contests-0045`

的 `agent_timeout`。

这些 timeout 会把：

- 当前 repetition loss-mask 掉
- 某些 step 的 `wait_for_generation_buffer` 明显拉长

最后一步 `step 20` 就被拖到了：

- `timing/step = 200.3471`
- `timing/wait_for_generation_buffer = 187.3785`

### 8.4 task-level protocol error 仍会出现

日志里还出现过多次：

- `ERROR: Invalid JSON: Invalid \\escape`

这不是 SkyRL 基础设施错误，而是 Harbor 任务协议 / agent 输出格式问题。当前它不会直接把训练器打崩，但会拖低 rollout 质量。

### 8.5 全量 loss-masked batch 仍可能出现

日志里明确出现过：

- `All outputs are loss masked, which may lead to NaN loss`

本次 run 没有因此真的炸掉，但这仍然是一个需要持续关注的风险信号。

## 9. 这轮最重要的经验教训

### 9.1 对 Harbor fully-async 做 20-step 验收，优先收紧上下文增长

和 `r5` 相比，`r6` 能稳定跑完 `20 step`，最关键的不是训练器参数变化，而是下面三项前置约束：

- `max_turns: 10 -> 6`
- `agent timeout: 900 -> 240`
- `max_generate_length: 1024 -> 768`

这三项的组合，本质上是在做同一件事：

- 把 Harbor agent 的长尾前移拦截，而不是让它吃满 timeout

### 9.2 20-step 能跑完，不代表路径已经“干净”

这轮要明确区分两件事：

- “训练是否能跑到 `step 20`”：
  - 能
- “Harbor fully-async 路径是否已经没有关键噪声”：
  - 不能这么说

当前的真实结论应该是：

- 这条路径已经足够做 `20 step` 验收
- `r6` 里的 rollout details incompleteness 已经需要和当前状态分开看待；在 `r7` 同配置 fresh rerun 里它没有复现
- 但还不适合把“agent_timeout / occasional stale violation / task-level protocol error”当成已经解决

### 9.3 复现实验时应优先用 r6 这组参数

如果目的只是：

- 验证当前 Harbor fully-async 是否还能跑通 20-step
- 验证 checkpoint / export 是否仍能正常落盘

那么应该优先复现 `r6` 这组参数，而不是直接回到更激进的 `pilot/full` 默认值。

## 10. 建议的后续工作

如果下一步要继续把 Harbor fully-async 做稳，优先级建议是：

1. 优先分析 `agent_timeout`，尤其是 output-length overflow retry 和单 task 长尾对 generation buffer 的拖累
2. 把 staleness violation 视为 Harbor 长尾现象来分析，而不是只从 trainer 逻辑角度看
3. 继续处理 task-level protocol error，例如 `Invalid JSON: Invalid \\escape`
4. 把这组更保守的 `r6/r7` 参数沉淀成脚本里的专用 `20-step validation` stage，并保留 rollout-details 回归检查

## 11. 收尾与清理

在本次记录结束前，我做了下面的清理：

- 训练主进程已停止
- 残留的 `VLLM::EngineCore` 已手动清理
- `8` 张 GPU 已全部释放

如果以后手动复现并需要清理残留，可用类似命令确认：

```bash
pgrep -af 'main_harbor_fully_async|ray::skyrl_entrypoint|AsyncVLLMInferenceEngine|VLLM::EngineCore' || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

## 12. r7 rollout-details 复核补充

为了确认 `r6` 里出现的 `Harbor rollout details 偶发不完整` 是否已经被当前本地 Harbor 根修解决，我又按同一组 `20 step` 参数做了一次 fresh rerun：

```bash
cd /home/hkang/zthunder_yagent/SkyRL

RUN_NAME_OVERRIDE=codecontest-8xh100-docker-fully-async-20step-r7-rollout-details-recheck \
RUN_ARTIFACT_ROOT=/scratch/$USER \
SCRATCH_ROOT=/scratch/$USER/skyrl_runtime \
bash examples/train_integrations/harbor/run_codecontest_8xh100_docker_fully_async.sh smoke \
  max_train_tasks=40 \
  max_eval_tasks=null \
  trainer.resume_mode=none \
  trainer.eval_interval=0 \
  trainer.ckpt_interval=20 \
  trainer.hf_save_interval=20 \
  trainer.logger="['tensorboard','console']" \
  trainer.collect_memory_metrics=false \
  harbor_trial_config.agent.override_timeout_sec=240 \
  harbor_trial_config.agent.kwargs.max_turns=6 \
  generator.sampling_params.max_generate_length=768 \
  generator.eval_sampling_params.max_generate_length=768
```

这轮 `r7` 的关键产物：

- 运行目录：
  - [codecontest-8xh100-docker-fully-async-20step-r7-rollout-details-recheck](/scratch/hkang/codecontest-8xh100-docker-fully-async-20step-r7-rollout-details-recheck)
- checkpoint：
  - [global_step_20](/scratch/hkang/codecontest-8xh100-docker-fully-async-20step-r7-rollout-details-recheck/ckpts/global_step_20)
  - [global_step_21](/scratch/hkang/codecontest-8xh100-docker-fully-async-20step-r7-rollout-details-recheck/ckpts/global_step_21)
- HF export：
  - [global_step_20](/scratch/hkang/codecontest-8xh100-docker-fully-async-20step-r7-rollout-details-recheck/exports/global_step_20)
  - [global_step_21](/scratch/hkang/codecontest-8xh100-docker-fully-async-20step-r7-rollout-details-recheck/exports/global_step_21)
- 主日志：
  - [infra-260308_054807.log](/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-fully-async-20step-r7-rollout-details-recheck/infra-260308_054807.log)

`r7` 的最终结果：

- `20 step` 跑完，`trainer/global_step = 20`
- `global_step_20` checkpoint 和 HF export 都成功落盘
- 最终也照常生成了 `global_step_21` 的 final checkpoint / final export

最关键的 rollout-details 复核结果：

- `trials_run/result.json = 80`
- `AgentTimeoutError = 8`
- 其他 `exception_type = 0`
- `assistant messages > 0` 但 `completion_token_ids == 0` 的 trial 数：`0`
- `assistant message count != completion_token_ids list count` 的 trial 数：`0`
- 全量搜索日志：
  - `Missing completion token ids`：`0`
  - `failed during Harbor postprocessing`：`0`

也就是说：

- `r6` 里那类 “assistant 文本在，但 `completion_token_ids` 缺项” 的历史问题
- 在当前本地 Harbor 根修后的 `r7` fresh rerun 中没有复现

这轮 `r7` 里仍然真实存在的问题是：

- `agent_timeout` 仍然存在，总计 `8` 个 timeout trial
- staleness 仍会偶发超过预算，例如 `step 20` 里 `async/staleness_max = 2`
- task-level protocol error 仍存在，日志里还能看到 `Invalid JSON: Invalid \\escape`

所以更新后的判断应该是：

- `Harbor rollout details 偶发不完整` 是 `r6` 的真实历史 bug
- 但按当前本地 Harbor runtime 的代码和这轮 `r7` fresh rerun 结果看，它已经不应再被列为“当前仍未解决的问题”
- 后续真正该继续盯的，是 `agent_timeout`、staleness 和 protocol-format 错误
