# CodeContests 8xH100 Docker Full Run Notes

日期：2026-03-07

## 1. 目标

本次记录对应 `CODECONTESTS_8XH100_DOCKER_PLAN.md` 中的 `Stage 3: Full Run`。

这里的“完成”不是指已经跑完整个 `1 epoch`，而是指：

- 在全量 `CodeContests` 上启动真实 `full` 训练
- 保持 `SkyRL + Harbor + Docker + vLLM + FSDP2` 整条链路稳定
- 打通统一监控
- 连续完成至少 `20` 个 training step
- 验证 checkpoint / HF export / reward / loss / memory 指标都能落盘

结论：

- `Stage 3` 的第一版可接受 full-run 基线已经完成
- 已连续完成 `20` 个 training step
- 但还没有完成 `1 epoch`
- 当前长跑的主要阻塞从“训练逻辑不稳定”转成了“`/home` 配额不足，不适合继续把 checkpoint/export 落在 home 下”

## 2. 使用环境

本次 full run 与前面的 smoke / pilot 使用同一台机器与工作区，一次性前置准备见：

- [CODECONTESTS_8XH100_DOCKER_SMOKE_TEST_NOTES.md](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/CODECONTESTS_8XH100_DOCKER_SMOKE_TEST_NOTES.md)
- [CODECONTESTS_8XH100_DOCKER_PILOT_RUN_NOTES.md](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/CODECONTESTS_8XH100_DOCKER_PILOT_RUN_NOTES.md)

本次 full run 实际使用环境：

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
- 运行时临时目录：`/scratch/hkang/skyrl_runtime`

本次 full run 依赖两个关键兼容性修正：

1. Harbor agent 在发请求前做输入长度预检查，超窗直接抛 `ContextLengthExceededError`，不让超长请求进入 vLLM
   - `.venv/lib/python3.13/site-packages/harbor/agents/terminus_2/terminus_2.py`
2. `SkyRL` 的 vLLM engine 对 `vllm==0.10.2` 的 import 兼容补丁
   - `skyrl/backends/skyrl_train/inference_engines/vllm/vllm_engine.py`

如果后续重建 `.venv` 或切换 Harbor 安装源，这两处需要重新确认，不要默认它们还在。

## 3. Stage 3 验收结果

| 验收项 | 结果 | 说明 |
| --- | --- | --- |
| full dataset 上真实训练启动 | 达成 | 使用全量 `CodeContests` 训练，全量 `OpenThoughts-TB-dev` 评估 |
| 统一监控可用 | 达成 | `TensorBoard + console + 外部 GPU/vLLM monitor` 已打通 |
| 连续训练 `>=20` step | 达成 | 已完成到 `global_step=20` |
| checkpoint 正常 | 达成 | `global_step_5/10/15/20` 均写出过 |
| HF export 正常 | 达成 | `global_step_5/10/15/20` 均写出过 |
| 训练不再被首步 OOM 卡死 | 达成 | 最终接受 run 没有重现 step2 backward OOM |
| 完整 `1 epoch` | 未达成 | 本次记录只完成 `20 step` 基线，长跑后续需把 checkpoint/export 移到 `/scratch` |

## 4. 实跑过程与关键发现

### 4.1 旧路径问题：vLLM 0.16 + overlong prompt 不稳定

在真正进入稳定 full run 之前，Stage 3 先暴露了两个系统性问题：

- `vllm==0.16.0` 这条路径对当前 Harbor 多轮 agent 工作负载不稳定
- Harbor 多轮消息在极端情况下会把输入推到 `32769 > 32768`
- 旧路径上，这类 overlong request 不只是返回 validation error，还可能把 engine core 直接打死

结论：

- 不能依赖“让 vLLM 自己拒绝超长请求”来兜底
- 必须在 Harbor 发请求前就做输入长度预检查
- 当前接受的 Stage 3 基线以 `vllm==0.10.2` 为前提

### 4.2 aggressive full run：step 2 backward OOM

在切到 `vllm==0.10.2` 后，先跑过一轮更激进的 full 配置，目的是验证“只换 vLLM 是否已经够了”。

这轮 aggressive baseline 的核心参数是：

- `generator.n_samples_per_prompt=4`
- `trainer.train_batch_size=8`
- `trainer.policy_mini_batch_size=8`
- `trainer.algorithm.max_seq_len=8192`
- `harbor_trial_config.agent.kwargs.max_turns=12`

结果：

- `step 1` 可以完成
- `step 2` 在 `policy backward` OOM

这说明：

- 只解决 vLLM 稳定性还不够
- 在当前 `8 x H100`、`policy/ref colocate`、`4 rollout + 4 train` 拓扑下，这组 full 配置对训练侧仍然过重

### 4.3 最终接受的 full run：20 step 基线

为拿到一个可持续运行的 Stage 3 基线，最终把 full 默认值收缩成：

- `generator.n_samples_per_prompt=2`
- `generator.eval_n_samples_per_prompt=1`
- `trainer.train_batch_size=4`
- `trainer.policy_mini_batch_size=4`
- `trainer.micro_forward_batch_size_per_gpu=1`
- `trainer.micro_train_batch_size_per_gpu=1`
- `trainer.algorithm.max_seq_len=6144`
- `trainer.flash_attn=false`
- `trainer.use_sample_packing=false`
- `generator.rate_limit.trajectories_per_second=2`
- `generator.rate_limit.max_concurrency=16`
- `harbor_trial_config.agent.kwargs.max_turns=10`
- `harbor_trial_config.agent.kwargs.temperature=0.3`
- `enable_thinking=false`
- `include_reasoning=false`
- `trainer.eval_before_train=false`
- `trainer.eval_interval=50`
- `trainer.logger=['tensorboard','console']`
- `trainer.collect_memory_metrics=true`

最终接受 run：

- `RUN_NAME=codecontest-8xh100-docker-full-v0102-step20-r3`
- 已连续完成 `20` 个 training step
- `global_step_5/10/15/20` 的 checkpoint 与 HF export 均写出过

## 5. 当前接受配置

本节对应当前 `examples/train_integrations/harbor/run_codecontest_8xh100_docker.sh` 中 `full` 的可接受基线。

### 5.1 拓扑

- 单机 `8 x H100`
- `trainer.placement.colocate_all=false`
- `trainer.placement.colocate_policy_ref=true`
- 训练池：`4 GPU`
- rollout 池：`4 GPU`
- ref model 启用方式：`trainer.algorithm.use_kl_loss=true`
- critic 关闭：`trainer.critic.model.path=null`

### 5.2 训练参数

- `trainer.strategy=fsdp2`
- `trainer.algorithm.advantage_estimator=grpo`
- `trainer.algorithm.loss_reduction=seq_mean_token_sum_norm`
- `trainer.algorithm.grpo_norm_by_std=false`
- `trainer.algorithm.use_kl_loss=true`
- `trainer.algorithm.kl_loss_coef=0.001`
- `trainer.algorithm.max_seq_len=6144`
- `trainer.epochs=1`
- `trainer.train_batch_size=4`
- `trainer.policy_mini_batch_size=4`
- `trainer.micro_forward_batch_size_per_gpu=1`
- `trainer.micro_train_batch_size_per_gpu=1`
- `trainer.policy.optimizer_config.lr=1e-6`
- `trainer.flash_attn=false`
- `trainer.use_sample_packing=false`
- `trainer.ckpt_interval=5`
- `trainer.hf_save_interval=5`

### 5.3 rollout / Harbor 参数

- `generator.n_samples_per_prompt=2`
- `generator.eval_n_samples_per_prompt=1`
- `generator.apply_overlong_filtering=true`
- `generator.inference_engine.num_engines=4`
- `generator.inference_engine.tensor_parallel_size=1`
- `generator.inference_engine.run_engines_locally=true`
- `generator.inference_engine.backend=vllm`
- `generator.inference_engine.async_engine=true`
- `generator.inference_engine.gpu_memory_utilization=0.8`
- `generator.inference_engine.enable_http_endpoint=true`
- `generator.inference_engine.http_endpoint_host=127.0.0.1`
- `generator.inference_engine.http_endpoint_port=8000`
- `generator.inference_engine.engine_init_kwargs.max_model_len=32768`
- `generator.rate_limit.enabled=true`
- `generator.rate_limit.trajectories_per_second=2`
- `generator.rate_limit.max_concurrency=16`
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

### 5.4 监控参数

- `trainer.logger=['tensorboard','console']`
- `trainer.collect_memory_metrics=true`
- `trainer.collect_memory_metrics_interval=1`
- 自动启动 `monitor_stage3_resources.sh`
- 统一写入 `$HOME/tmp_logs/$RUN_NAME/tensorboard`

## 6. 复现步骤

### 6.1 一次性前置准备

1. 准备 Harbor 数据：

```bash
cd /home/hkang/zthunder_yagent/SkyRL

./.venv/bin/python examples/train_integrations/harbor/prepare_harbor_dataset.py \
  --dataset open-thoughts/CodeContests

./.venv/bin/python examples/train_integrations/harbor/prepare_harbor_dataset.py \
  --dataset open-thoughts/OpenThoughts-TB-dev
```

2. 确认模型缓存已经统一在 `/data/zy/models`：

```bash
ls /data/zy/models/hub/models--Qwen--Qwen3-8B/snapshots
```

3. 确认运行环境版本：

```bash
cd /home/hkang/zthunder_yagent/SkyRL
./.venv/bin/python - <<'PY'
import sys
import torch
import vllm
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("vllm", vllm.__version__)
PY
```

期望输出：

- `python 3.13.11`
- `torch 2.8.0+cu128`
- `vllm 0.10.2`

4. 确认 Harbor 预检查补丁仍在：

```bash
cd /home/hkang/zthunder_yagent/SkyRL
rg -n "projected_tokens = self._count_total_tokens_with_prompt|ContextLengthExceededError" \
  .venv/lib/python3.13/site-packages/harbor/agents/terminus_2/terminus_2.py
```

5. 确认 SkyRL 的 vLLM 兼容补丁仍在：

```bash
cd /home/hkang/zthunder_yagent/SkyRL
rg -n "serving_chat|serving_completion|serving_models" \
  skyrl/backends/skyrl_train/inference_engines/vllm/vllm_engine.py
```

### 6.2 历史接受 run 的原始启动命令

这是当时实际用于拿到 `20 step` 基线的命令：

```bash
cd /home/hkang/zthunder_yagent/SkyRL
RUN_NAME_OVERRIDE=codecontest-8xh100-docker-full-v0102-step20-r3 \
  bash examples/train_integrations/harbor/run_codecontest_8xh100_docker.sh full
```

说明：

- 这条命令对应的是当时的脚本行为
- 当时 `trials / ckpts / exports / monitoring / logs` 都默认写到 `$HOME`
- 它能复现历史结果
- 但不适合继续做长跑，因为会再次逼近 `/home` 配额

### 6.3 推荐的复现命令

当前脚本已经改成：

- `full` 默认把 `trials / ckpts / exports / monitoring` 放到 `/scratch/$USER/$RUN_NAME`
- `logs / tensorboard` 继续放到 `$HOME/tmp_logs/$RUN_NAME`

因此当前推荐的复现命令已经可以直接用：

```bash
cd /home/hkang/zthunder_yagent/SkyRL

RUN_NAME_OVERRIDE=codecontest-8xh100-docker-full-rerun \
SCRATCH_ROOT=/scratch/$USER/skyrl_runtime \
bash examples/train_integrations/harbor/run_codecontest_8xh100_docker.sh full
```

说明：

- 日志仍然默认留在 `$HOME/tmp_logs/$RUN_NAME`
- `trials / ckpts / exports / monitoring` 默认放到 `/scratch/$USER/$RUN_NAME`
- 这样可以保留统一的 TensorBoard 路径，又避免继续打满 `/home`
- 如果想手动覆盖 full 的 artifact 根目录，可以额外设置：

```bash
RUN_ARTIFACT_ROOT=/scratch/$USER
```

### 6.4 打开统一可视化

训练启动后，另开一个终端：

```bash
cd /home/hkang/zthunder_yagent/SkyRL
./.venv/bin/tensorboard \
  --logdir "$HOME/tmp_logs/codecontest-8xh100-docker-full-rerun/tensorboard" \
  --host 0.0.0.0 \
  --port 6006
```

访问地址：

- `http://<机器IP>:6006`

### 6.5 运行时观测

看训练主日志：

```bash
tail -f /home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-full-rerun/infra-*.log
```

看 GPU：

```bash
watch -n 5 nvidia-smi
```

看 checkpoint 进度：

```bash
find /scratch/$USER/codecontest-8xh100-docker-full-rerun/ckpts \
  -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' | sort
```

看 TensorBoard 事件里最新的核心指标：

```bash
cd /home/hkang/zthunder_yagent/SkyRL
./.venv/bin/python - <<'PY'
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

tb_dir = Path("/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-full-rerun/tensorboard")
files = sorted(tb_dir.glob("events.out.tfevents.*"))
ea = event_accumulator.EventAccumulator(str(files[-1]))
ea.Reload()
for tag in ["policy/final_loss", "policy/policy_kl", "reward/avg_pass_at_2"]:
    vals = ea.Scalars(tag)
    if vals:
        print(tag, vals[-1].step, vals[-1].value)
PY
```

## 7. 遇到的坑

### 7.1 不能依赖 vLLM 自己兜底超长输入

现象：

- Harbor 多轮 agent 在极端情况下会把输入推到 `32769 > 32768`
- 旧路径上，这类请求不仅是 `VLLMValidationError`
- 还会把 engine core 直接打死

处理：

- 在 Harbor 发请求前做 tokenizer 级输入长度预检查
- 超窗直接在 Harbor 侧抛 `ContextLengthExceededError`

结论：

- 这不是“可选优化”，而是当前 Stage 3 的稳定性前提

### 7.2 aggressive full 参数在 step 2 训练侧 OOM

现象：

- `n_samples=4 + train_batch_size=8 + mini_batch_size=8 + max_seq_len=8192`
- `step 1` 能过
- `step 2` backward OOM

处理：

- 把 full 基线收缩为：
  - `n_samples=2`
  - `train_batch_size=4`
  - `policy_mini_batch_size=4`
  - `max_seq_len=6144`
  - `max_turns=10`

结论：

- 当前 Stage 3 能稳定到 `20 step` 的基线，就是现在脚本里的保守 full 默认值

### 7.3 checkpoint/export 不应默认继续写在 `/home`

现象：

- 历史接受 run 在 `global_step_20` 完成 checkpoint/export 后，因 `/home` 配额耗尽而退出
- `TensorBoard` 标量只稳定保留到 `step 19`
- `step 20` 的 checkpoint/export 已经完成，但最后一次 event flush 没完整写下去

本次清理后当前空间状态：

- `/home`：`9.5T used / 526G avail / 95%`
- `/scratch`：`1.4T used / 5.5T avail / 20%`

结论：

- 长跑时，`trainer.ckpt_path` 和 `trainer.export_path` 必须改到 `/scratch`
- 日志留在 `/home` 即可

### 7.4 milestone 不能只看 TensorBoard

现象：

- 因为最后一次 writer flush 失败，`TensorBoard` 里只剩到 `step 19`
- 但 `step 20` 的 checkpoint/export 已经真实写出过

处理：

- milestone 判断同时看：
  - 训练日志
  - checkpoint 目录
  - export 目录
  - TensorBoard scalar

结论：

- 只盯一个来源容易误判 step 是否完成

## 8. 可复用经验

1. full run 不要直接沿用 plan 初稿的 aggressive 参数

当前这台机器上，先以当前保守基线起跑更合理：

- `n_samples=2`
- `train_batch_size=4`
- `policy_mini_batch_size=4`
- `max_seq_len=6144`
- `max_turns=10`

2. policy/ref colocate 可以保留，但不要在 full 阶段同时把 batch 和样本数拉太大

当前稳定性的核心矛盾在训练侧显存，不在 rollout GPU 不够。

3. Qwen3 做 Harbor coding agent 时，协议约束必须收紧

建议继续保留：

- `temperature=0.3`
- `enable_thinking=false`
- `include_reasoning=false`

否则 JSON 协议噪音会明显增加。

4. 日志、checkpoint、export 应该分层放置

推荐：

- 日志：`/home/.../tmp_logs`
- checkpoint：`/scratch/.../ckpts`
- export：`/scratch/.../exports`
- trials：`/scratch/.../trials_run`

5. 长跑前先确认两个本地补丁还在

- Harbor preflight length check
- SkyRL 对 `vllm 0.10.2` 的兼容 patch

这是当前 full run 能稳定推进的前提，不要假设“只要脚本一样就一定能复现”。

## 9. 当前保留的产物

本次历史接受 run 的 checkpoint/export 为释放 `/home` 配额，已在验证完成后清理。

当前保留：

- run 目录：
  - `/home/hkang/zthunder_yagent/codecontest-8xh100-docker-full-v0102-step20-r3`
- 日志目录：
  - `/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-full-v0102-step20-r3`
- 主日志：
  - `/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-full-v0102-step20-r3/infra-260307_064212.log`
- 曲线图：
  - `/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-full-v0102-step20-r3/policy_final_loss_curve.svg`
  - `/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-full-v0102-step20-r3/reward_avg_pass_at_2_curve.svg`
  - `/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-full-v0102-step20-r3/policy_kl_curve.svg`
- 对应 CSV：
  - `/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-full-v0102-step20-r3/policy_final_loss_curve.csv`
  - `/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-full-v0102-step20-r3/reward_avg_pass_at_2_curve.csv`
  - `/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-full-v0102-step20-r3/policy_kl_curve.csv`

run 目录当前剩余体积：

- run 根目录：约 `43M`
- 日志目录：约 `1.6M`

说明：

- `ckpts/` 和 `exports/` 当前已清空
- 这是事后清理动作，不影响上面对 `global_step_20` 已完成的结论
