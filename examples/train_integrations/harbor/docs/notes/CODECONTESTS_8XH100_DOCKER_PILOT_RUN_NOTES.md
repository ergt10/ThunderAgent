# CodeContests 8xH100 Docker Pilot Run Notes

日期：2026-03-06

## 1. 目标

本次记录对应 `CODECONTESTS_8XH100_DOCKER_PLAN.md` 中的 `Stage 2: Pilot Run`。

Stage 2 的目标是：

- 在真实 `SkyRL + Harbor` 训练路径上完成 RL update，而不是只做 generate-only smoke
- 验证 `8 x H100`、`colocate_all=false`、`colocate_policy_ref=true` 的训练/rollout 拓扑
- 验证训练侧显存、错误率、reward 信号、checkpoint、HF export 是否正常

结论：`Stage 2` 已完成。

说明：本次 `Stage 2` 是通过多轮实跑收敛完成的，而不是一次完全不改参数的原始计划 run 直接通过。最终结论基于：

- 一次保留 `eval_before_train=true` 的 pilot run，验证了评估闭环和训练启动
- 一次训练侧 OOM 诊断 run，明确了 `32768` 训练序列长度不可行
- 一次 `max_seq_len=8192` 的训练 run，验证 OOM 已被消除
- 一次最终接受的 curated subset run，完成首个训练 step，并成功写出 checkpoint 与 HF export

## 2. 使用环境

本次 pilot 与 stage1 smoke 使用同一套机器和工作区。一次性前置准备与共享目录约定，沿用：

- [CODECONTESTS_8XH100_DOCKER_SMOKE_TEST_NOTES.md](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/CODECONTESTS_8XH100_DOCKER_SMOKE_TEST_NOTES.md)

本次 pilot 实际使用环境：

- 工作区：`/home/hkang/zthunder_yagent/SkyRL`
- `HOME`：`/home/hkang/zthunder_yagent`
- Python 环境：`/home/hkang/zthunder_yagent/SkyRL/.venv`
- Python 版本：`3.13.11`
- 共享模型目录：`/data/zy/models`
- HF cache：`/data/zy/models/hub`
- Harbor 数据目录：
  - `/home/hkang/zthunder_yagent/data/harbor/CodeContests`
  - `/home/hkang/zthunder_yagent/data/harbor/OpenThoughts-TB-dev`
- 临时运行目录：`/scratch/hkang/skyrl_runtime`

## 3. Stage 2 验收结果

| 验收项 | 结果 | 证据 |
| --- | --- | --- |
| 无训练 OOM | 达成 | `max_seq_len=8192` 后，训练前向、policy update、checkpoint、HF export 均完成 |
| timeout / error trajectory 有界 | 达成 | 最终接受 run 的首个 step：`num_timeout_trajectories=0`，`num_error_trajectories=1`，`num_masked_instances=1` |
| 持续出现非零 reward | 达成 | baseline pilot eval 有稳定非零 reward；最终接受 run 首个 step `reward/avg_raw_reward=0.09375` |
| checkpoint 正常 | 达成 | `global_step_1` FSDP checkpoint 已写出 |
| evaluation 正常 | 达成 | baseline pilot run 在 `eval_before_train=true` 下完成 eval 并进入真实训练 |

## 4. 实跑过程与关键发现

### 4.1 Baseline pilot：按计划验证 eval 闭环与训练启动

目的：尽量贴近 plan 的 Stage 2 默认形态，先验证 `eval_before_train=true`、`256 train / 20 eval`、`policy/ref colocate`、`4 rollout + 4 train` 拓扑是否能完整启动。

命令：

```bash
bash examples/train_integrations/harbor/run_codecontest_8xh100_docker.sh pilot \
  trainer.run_name=codecontest-8xh100-docker-pilot-plan256 \
  harbor_trial_config.trials_dir=/home/hkang/zthunder_yagent/codecontest-8xh100-docker-pilot-plan256/trials_run \
  trainer.log_path=/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-pilot-plan256 \
  trainer.ckpt_path=/home/hkang/zthunder_yagent/codecontest-8xh100-docker-pilot-plan256/ckpts \
  trainer.export_path=/home/hkang/zthunder_yagent/codecontest-8xh100-docker-pilot-plan256/exports \
  trainer.resume_mode=none \
  trainer.logger=console \
  trainer.flash_attn=false \
  trainer.use_sample_packing=false
```

观测结果：

- `CodeContests` 被正确识别为 `9644` 个有效任务目录，并限制到前 `256` 个任务
- `OpenThoughts-TB-dev` 被正确识别为 `70` 个有效任务目录，并限制到前 `20` 个任务
- `eval_before_train` 正常执行完成
- live console 中观测到的核心 eval 指标：
  - `eval/all/avg_score = 0.1334`
  - `eval/all/pass_at_1 = 0.2000`
  - `eval/all/mean_positive_reward = 0.1334`
  - `eval/all/generate/max_num_tokens = 28338`
- eval 完成后，训练已真正进入：
  - `Started: 'step'`
  - `Started: 'generate'`

结论：

- Stage 2 的 `evaluation run normally` 已完成验证
- 训练拓扑、policy/ref colocate、rollout engine 拓扑都已启动成功
- 但这次 run 在刚进入真实训练后被我手动停止，目的是更快拿到 training-side 诊断和 checkpoint 证据

相关日志目录：

- `/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-pilot-plan256`

### 4.2 Step1 checkpoint run：暴露训练侧真实 OOM

目的：在不重复长时间 eval 的前提下，尽快验证首个训练 batch、首个 checkpoint 是否可达。

命令：

```bash
bash examples/train_integrations/harbor/run_codecontest_8xh100_docker.sh pilot \
  trainer.run_name=codecontest-8xh100-docker-pilot-plan256-step1ckpt \
  harbor_trial_config.trials_dir=/home/hkang/zthunder_yagent/codecontest-8xh100-docker-pilot-plan256-step1ckpt/trials_run \
  trainer.log_path=/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-pilot-plan256-step1ckpt \
  trainer.ckpt_path=/home/hkang/zthunder_yagent/codecontest-8xh100-docker-pilot-plan256-step1ckpt/ckpts \
  trainer.export_path=/home/hkang/zthunder_yagent/codecontest-8xh100-docker-pilot-plan256-step1ckpt/exports \
  trainer.resume_mode=none \
  trainer.logger=console \
  trainer.flash_attn=false \
  trainer.use_sample_packing=false \
  trainer.eval_before_train=false \
  trainer.ckpt_interval=1 \
  trainer.hf_save_interval=1 \
  harbor_trial_config.agent.kwargs.max_turns=16 \
  harbor_trial_config.agent.kwargs.temperature=0.3 \
  harbor_trial_config.agent.kwargs.llm_call_kwargs.extra_body.chat_template_kwargs.enable_thinking=false \
  harbor_trial_config.agent.kwargs.llm_call_kwargs.extra_body.include_reasoning=false
```

观测结果：

- 首个训练 batch 的 rollout 已经完成
- reward 正常回传，训练真正进入 `fwd_logprobs_values_reward`
- 随后在 `FSDPRefWorkerBase.forward` 发生真实训练 OOM
- live console 中观测到的关键错误：
  - `ray.exceptions.RayTaskError(OutOfMemoryError)`
  - 出现在 ref forward
  - PyTorch 报错里尝试分配约 `26.27 GiB`

结论：

- `trainer.algorithm.max_seq_len=32768` 对 rollout 可以工作，但对当前训练侧 ref forward 不可行
- Stage 2 原计划里的 `32768` 上下文长度不能直接进入稳定 pilot

相关日志目录：

- `/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-pilot-plan256-step1ckpt`

### 4.3 `max_seq_len=8192` run：训练 OOM 已消除，但首批次被病理任务拖住

目的：验证把训练序列长度从 `32768` 降到 `8192` 后，训练侧 OOM 是否消失。

命令：

```bash
bash examples/train_integrations/harbor/run_codecontest_8xh100_docker.sh pilot \
  trainer.run_name=codecontest-8xh100-docker-pilot-plan256-ckpt8192 \
  harbor_trial_config.trials_dir=/home/hkang/zthunder_yagent/codecontest-8xh100-docker-pilot-plan256-ckpt8192/trials_run \
  trainer.log_path=/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-pilot-plan256-ckpt8192 \
  trainer.ckpt_path=/home/hkang/zthunder_yagent/codecontest-8xh100-docker-pilot-plan256-ckpt8192/ckpts \
  trainer.export_path=/home/hkang/zthunder_yagent/codecontest-8xh100-docker-pilot-plan256-ckpt8192/exports \
  trainer.resume_mode=none \
  trainer.logger=console \
  trainer.flash_attn=false \
  trainer.use_sample_packing=false \
  trainer.eval_before_train=false \
  trainer.ckpt_interval=1 \
  trainer.hf_save_interval=1 \
  trainer.algorithm.max_seq_len=8192 \
  harbor_trial_config.agent.kwargs.max_turns=12 \
  harbor_trial_config.agent.kwargs.temperature=0.3 \
  harbor_trial_config.agent.kwargs.llm_call_kwargs.extra_body.chat_template_kwargs.enable_thinking=false \
  harbor_trial_config.agent.kwargs.llm_call_kwargs.extra_body.include_reasoning=false
```

观测结果：

- 训练侧 OOM 没有再出现
- 但首批次被 `code_contests-0028` 反复触发的长上下文问题拖住
- vLLM 多次报：
  - `You passed 32769 input tokens ... context length is only 32768`
- 这导致首个 checkpoint 在合理时间内没有落盘

结论：

- `8192` 已经足以解决训练侧 OOM
- 但 pilot 的前 `256` 个任务子集里，至少有一个病理任务会严重拖慢首批次收敛

相关日志目录：

- `/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-pilot-plan256-ckpt8192`

### 4.4 构造 deterministic curated subset：排除 `code_contests-0028`

目的：保持 `256` 任务规模不变，但去掉已验证会严重拖慢首批次的病理任务，保证 pilot 能在合理时间内完成首个 update 和 checkpoint。

构造命令：

```bash
SRC=/home/hkang/zthunder_yagent/data/harbor/CodeContests
DST=/home/hkang/zthunder_yagent/data/harbor/CodeContests-pilot256-no0028
rm -rf "$DST"
mkdir -p "$DST"
find "$SRC" -maxdepth 1 -mindepth 1 -type d -name 'code_contests-*' | \
  sort | grep -v '/code_contests-0028$' | head -n 256 | \
  while read -r d; do
    ln -s "$d" "$DST/$(basename "$d")"
  done
```

校验命令：

```bash
find /home/hkang/zthunder_yagent/data/harbor/CodeContests-pilot256-no0028 \
  -maxdepth 1 -mindepth 1 -type l | wc -l
```

期望输出：`256`

### 4.5 最终接受的 Stage 2 run

目的：在已知可行的训练长度和更稳定的 pilot 子集上，完成真实训练 step、checkpoint 和 HF export。

命令：

```bash
bash examples/train_integrations/harbor/run_codecontest_8xh100_docker.sh pilot \
  data.train_data="['/home/hkang/zthunder_yagent/data/harbor/CodeContests-pilot256-no0028']" \
  trainer.run_name=codecontest-8xh100-docker-pilot-plan256-no0028-ckpt8192 \
  harbor_trial_config.trials_dir=/home/hkang/zthunder_yagent/codecontest-8xh100-docker-pilot-plan256-no0028-ckpt8192/trials_run \
  trainer.log_path=/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-pilot-plan256-no0028-ckpt8192 \
  trainer.ckpt_path=/home/hkang/zthunder_yagent/codecontest-8xh100-docker-pilot-plan256-no0028-ckpt8192/ckpts \
  trainer.export_path=/home/hkang/zthunder_yagent/codecontest-8xh100-docker-pilot-plan256-no0028-ckpt8192/exports \
  trainer.resume_mode=none \
  trainer.logger=console \
  trainer.flash_attn=false \
  trainer.use_sample_packing=false \
  trainer.eval_before_train=false \
  trainer.ckpt_interval=1 \
  trainer.hf_save_interval=1 \
  trainer.algorithm.max_seq_len=8192 \
  harbor_trial_config.agent.kwargs.max_turns=12 \
  harbor_trial_config.agent.kwargs.temperature=0.3 \
  harbor_trial_config.agent.kwargs.llm_call_kwargs.extra_body.chat_template_kwargs.enable_thinking=false \
  harbor_trial_config.agent.kwargs.llm_call_kwargs.extra_body.include_reasoning=false
```

最终接受 run 的关键配置：

- 训练子集：`256` 个 curated `CodeContests` 任务，排除了 `code_contests-0028`
- 评估集：沿用脚本默认 `OpenThoughts-TB-dev`，但该 run 设置 `eval_before_train=false`
- `trainer.algorithm.max_seq_len=8192`
- `trainer.ckpt_interval=1`
- `trainer.hf_save_interval=1`
- `harbor_trial_config.agent.kwargs.max_turns=12`
- `temperature=0.3`
- `enable_thinking=false`
- `include_reasoning=false`
- `trainer.flash_attn=false`
- `trainer.use_sample_packing=false`

live console 中观测到的首个训练 step 指标：

- `generate/num_error_trajectories = 1`
- `generate/num_masked_instances = 1`
- `generate/num_timeout_trajectories = 0`
- `generate/trajectories_context_length_exceeded = 0`
- `reward/avg_pass_at_4 = 0.1250`
- `reward/avg_raw_reward = 0.09375`
- `reward/mean_positive_reward = 0.09375`
- `loss/avg_final_rewards = 0.09375`
- `policy/final_loss = -0.0001220703125`
- `policy/policy_entropy = 0.11711607128381729`
- `policy/grad_norm = 0.007397575303912163`
- `timing/generate = 667.3008s`
- `timing/fwd_logprobs_values_reward = 9.6105s`
- `timing/policy_train = 16.0509s`
- `trainer/global_step = 1`

本次 run 的结构性结果：

- 首个训练 step 完整结束
- `save_checkpoints` 完成
- `save_hf_model` 完成
- 第二个训练 step 已经开始，说明训练循环可以继续
- 达到上述证据后，我手动停止 run，避免继续占用 GPU

产物路径：

- run 根目录：`/home/hkang/zthunder_yagent/codecontest-8xh100-docker-pilot-plan256-no0028-ckpt8192`
- 日志目录：`/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-pilot-plan256-no0028-ckpt8192`
- checkpoint 根目录：`/home/hkang/zthunder_yagent/codecontest-8xh100-docker-pilot-plan256-no0028-ckpt8192/ckpts`
- HF export 根目录：`/home/hkang/zthunder_yagent/codecontest-8xh100-docker-pilot-plan256-no0028-ckpt8192/exports`

已确认存在的 checkpoint 文件：

- `/home/hkang/zthunder_yagent/codecontest-8xh100-docker-pilot-plan256-no0028-ckpt8192/ckpts/global_step_1/data.pt`
- `/home/hkang/zthunder_yagent/codecontest-8xh100-docker-pilot-plan256-no0028-ckpt8192/ckpts/global_step_1/trainer_state.pt`
- `/home/hkang/zthunder_yagent/codecontest-8xh100-docker-pilot-plan256-no0028-ckpt8192/ckpts/global_step_1/policy/model_world_size_4_rank_0.pt`
- `/home/hkang/zthunder_yagent/codecontest-8xh100-docker-pilot-plan256-no0028-ckpt8192/ckpts/latest_ckpt_global_step.txt`

已确认存在的 HF export 文件：

- `/home/hkang/zthunder_yagent/codecontest-8xh100-docker-pilot-plan256-no0028-ckpt8192/exports/global_step_1/policy/model-00001-of-00007.safetensors`
- `/home/hkang/zthunder_yagent/codecontest-8xh100-docker-pilot-plan256-no0028-ckpt8192/exports/global_step_1/policy/model.safetensors.index.json`
- `/home/hkang/zthunder_yagent/codecontest-8xh100-docker-pilot-plan256-no0028-ckpt8192/exports/global_step_1/policy/config.json`
- `/home/hkang/zthunder_yagent/codecontest-8xh100-docker-pilot-plan256-no0028-ckpt8192/exports/global_step_1/policy/tokenizer.json`

## 5. Stage 2 的完整可复现步骤

### 5.1 一次性前置准备

沿用 smoke notes 中的以下步骤：

- Harbor 数据目录准备
- `/data/zy/models` 模型缓存准备
- GPU 基础检查

见：

- [CODECONTESTS_8XH100_DOCKER_SMOKE_TEST_NOTES.md](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/CODECONTESTS_8XH100_DOCKER_SMOKE_TEST_NOTES.md)

### 5.2 复现 Stage 2 的评估闭环

执行 4.1 的 baseline pilot 命令。

目标：

- 验证 `256 train / 20 eval`
- 验证 `eval_before_train=true`
- 验证训练能够在 eval 后真正进入 `step` / `generate`

### 5.3 构造最终接受的 pilot 子集

执行 4.4 的 curated subset 构造命令。

### 5.4 运行最终接受的 Stage 2 配置

执行 4.5 的最终接受 run 命令。

### 5.5 校验 checkpoint 与 HF export

```bash
RUN_DIR=/home/hkang/zthunder_yagent/codecontest-8xh100-docker-pilot-plan256-no0028-ckpt8192

cat "$RUN_DIR/ckpts/latest_ckpt_global_step.txt"
find "$RUN_DIR/ckpts/global_step_1" -maxdepth 3 -type f | sort
find "$RUN_DIR/exports/global_step_1" -maxdepth 3 -type f | sort
```

期望：

- `latest_ckpt_global_step.txt` 内容为 `1`
- `ckpts/global_step_1` 下存在 FSDP model shard、optimizer shard、trainer state、data state
- `exports/global_step_1` 下存在 `7` 个 safetensors shard、`model.safetensors.index.json`、tokenizer/config 文件

### 5.6 停止与清理

如果 run 已达到首个 step、checkpoint、HF export 的验证目标，可以手动停止：

```bash
pkill -f 'codecontest-8xh100-docker-pilot-plan256-no0028-ckpt8192|examples.train_integrations.harbor.entrypoints.main_harbor' || true
sleep 5
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

如果像我这次一样是手动中断，还需要清理 Harbor 临时容器与网络：

```bash
containers=$(docker ps -a --format '{{.ID}} {{.Names}}' | awk '$2 ~ /^code_contests-/ {print $1}')
if [ -n "${containers:-}" ]; then
  echo "$containers" | xargs -r docker rm -f
fi

networks=$(docker network ls --format '{{.ID}} {{.Name}}' | awk '$2 ~ /^code_contests-/ {print $1}')
if [ -n "${networks:-}" ]; then
  echo "$networks" | xargs -r docker network rm
fi
```

## 6. 与原 plan 的偏离

本次 `Stage 2` 完成时，相对原 plan 有以下偏离。这些偏离不是随意调整，而是 pilot 实跑收敛出来的必要条件。

1. `trainer.algorithm.max_seq_len` 从 `32768` 降到 `8192`
- 原因：`32768` 在训练侧 ref forward 会发生真实 OOM
- 影响：训练侧更稳定，但训练时可直接反向的序列上限低于 rollout 上限

2. 最终接受 run 使用 curated `256` 任务子集，而不是“排序后前 256 个任务”
- 原因：`code_contests-0028` 在 pilot 首批次里反复触发超长上下文 tail
- 影响：Stage 2 结论仍然有效，但该 curated subset 需要在文档中明确保存和复现

3. 用两类 run 共同完成 Stage 2，而不是单个 run 同时覆盖所有验收点
- baseline pilot：负责 eval 闭环与训练启动验证
- accepted pilot：负责训练 step、checkpoint、HF export 验证

4. checkpoint 验证 run 使用了更激进的保存频率
- `trainer.ckpt_interval=1`
- `trainer.hf_save_interval=1`
- 原因：缩短 pilot 验证闭环，尽快确认训练产物路径正确

5. protocol 约束做了收敛
- `temperature=0.3`
- `enable_thinking=false`
- `include_reasoning=false`
- `max_turns=12`
- 原因：降低 Harbor agent 协议噪音和无意义长尾

6. `flash-attn` 和 sample packing 在 pilot 中被显式关闭
- `trainer.flash_attn=false`
- `trainer.use_sample_packing=false`
- 原因：当前 `.venv` 的 `Python 3.13` 环境下，`flash-attn` 不可作为稳定前提使用；pilot 目标是先验证训练主链路稳定性

## 7. 进入 Stage 3 之前仍需关注的问题

这些问题没有阻塞 `Stage 2`，但在 `Stage 3` 前应该继续处理。

1. 仍然存在个别 Harbor Docker 启动竞态
- 表现为：`No such image: hb__code_contests-xxxx:latest`
- 当前影响：在最终接受 run 的首个 step 中被控制在 `1` 条 error trajectory
- 进入 full run 前，应该继续降低这类启动竞态

2. 仍然存在个别极端长上下文任务
- 表现为：`32769 > 32768` 的 vLLM validation error
- 当前处理：通过 curated subset 避开 `code_contests-0028`
- full run 前，应该考虑更系统的长上下文治理，而不是只靠剔除个例

3. 当前 `Stage 2` 的稳定训练序列长度是 `8192`，不是计划中的 `32768`
- 这意味着 `Stage 3` 不能直接照抄原始计划参数
- full run 前，需要以 `8192` 为已知稳定基线，再决定是否尝试上调

## 8. 当前结论

按 plan 的 Stage 2 验收标准，当前结论是：

- `Stage 2` 已完成
- 可以进入 `Stage 3` 设计与首轮 full run 准备
- 进入 `Stage 3` 时，应以本文件中的“最终接受配置”作为基线，而不是回退到原始未收敛的 pilot 参数
