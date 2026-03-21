# CodeContests 8xH100 Docker Smoke Test Notes

日期：2026-03-06

## 1. 目标

本次 smoke 的目标是验证 plan 中定义的基础设施链路，而不是验证训练效果：

- `SkyRL -> Harbor -> Docker sandbox -> verifier reward -> SkyRL`
- `8 x H100` 单机资源下，`colocate_all=false`、`colocate_policy_ref=true` 的拓扑可启动
- 本地 `vLLM` HTTP endpoint 可服务 Harbor agent
- `CodeContests` Harbor 任务能真实执行并回传 reward

结论：目标已达成。

## 2. 本次测试使用的实际环境

- 工作区：`/home/hkang/zthunder_yagent/SkyRL`
- `HOME`：`/home/hkang/zthunder_yagent`
- Python 环境：`/home/hkang/zthunder_yagent/SkyRL/.venv`
- Python 版本：`3.13.11`
- 共享模型目录：`/data/zy/models`
- HF cache：`/data/zy/models/hub`
- Xet cache：`/data/zy/models/xet`
- Harbor 数据目录：
  - `/home/hkang/zthunder_yagent/data/harbor/CodeContests`
  - `/home/hkang/zthunder_yagent/data/harbor/OpenThoughts-TB-dev`
- Ray tmp：`/scratch/hkang/skyrl_runtime/ray_tmp`
- 运行时缓存：
  - `UV_CACHE_DIR=/scratch/hkang/skyrl_runtime/uv-codex`
  - `TORCHINDUCTOR_CACHE_DIR=/scratch/hkang/skyrl_runtime/torchinductor`
  - `TRITON_CACHE_DIR=/scratch/hkang/skyrl_runtime/triton`

这些路径现在已经由 [run_codecontest_8xh100_docker.sh](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/run_codecontest_8xh100_docker.sh) 统一设置。

## 3. 代码前提

本次 smoke 基于当前仓库状态，尤其依赖这两个修复：

1. `main_harbor_generate.py` 不再硬编码只跑前 `10` 个 prompt
- 见 [main_harbor_generate.py](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/entrypoints/main_harbor_generate.py#L70)

2. `run_codecontest_8xh100_docker.sh` 已修复 stage 参数透传，并统一模型/cache/tmp 目录
- 见 [run_codecontest_8xh100_docker.sh](/home/hkang/zthunder_yagent/SkyRL/examples/train_integrations/harbor/run_codecontest_8xh100_docker.sh#L17)

如果回退这些修复，下面的复现步骤不成立。

## 4. 一次性准备步骤

### 4.1 准备 Harbor 数据目录

在仓库根目录执行：

```bash
/home/hkang/zthunder_yagent/SkyRL/.venv/bin/python \
  examples/train_integrations/harbor/prepare_harbor_dataset.py \
  --dataset open-thoughts/CodeContests

/home/hkang/zthunder_yagent/SkyRL/.venv/bin/python \
  examples/train_integrations/harbor/prepare_harbor_dataset.py \
  --dataset open-thoughts/OpenThoughts-TB-dev
```

期望产物：

- `/home/hkang/zthunder_yagent/data/harbor/CodeContests`
- `/home/hkang/zthunder_yagent/data/harbor/OpenThoughts-TB-dev`

### 4.2 准备模型缓存到 `/data/zy/models`

如果 `Qwen/Qwen3-8B` 还没缓存到共享模型目录，执行：

```bash
HF_HOME=/data/zy/models \
HUGGINGFACE_HUB_CACHE=/data/zy/models/hub \
HF_HUB_CACHE=/data/zy/models/hub \
TRANSFORMERS_CACHE=/data/zy/models/hub \
HF_XET_CACHE=/data/zy/models/xet \
/home/hkang/zthunder_yagent/SkyRL/.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3-8B",
    repo_type="model",
    cache_dir="/data/zy/models/hub",
)
PY
```

### 4.3 基础检查

```bash
nvidia-smi -L
/home/hkang/zthunder_yagent/SkyRL/.venv/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
```

期望：

- 能看到 `8` 张 H100
- `torch.cuda.is_available()` 为 `True`
- `torch.cuda.device_count()` 为 `8`

## 5. 本次 smoke 的精确配置

本次实际跑通的 plan smoke 配置如下：

- 入口：`examples.train_integrations.harbor.entrypoints.main_harbor_generate`
- 模型：`Qwen/Qwen3-8B`
- `trainer.strategy=fsdp2`
- `trainer.placement.colocate_all=false`
- `trainer.placement.colocate_policy_ref=true`
- `trainer.placement.policy_num_gpus_per_node=4`
- `trainer.placement.ref_num_gpus_per_node=4`
- `generator.inference_engine.num_engines=4`
- `generator.inference_engine.tensor_parallel_size=1`
- `generator.inference_engine.run_engines_locally=true`
- `generator.inference_engine.backend=vllm`
- `generator.inference_engine.gpu_memory_utilization=0.8`
- `generator.inference_engine.enable_http_endpoint=true`
- `generator.inference_engine.engine_init_kwargs.max_model_len=32768`
- `trainer.algorithm.max_seq_len=32768`
- `generator.rate_limit.trajectories_per_second=1`
- `generator.rate_limit.max_concurrency=8`
- `max_train_tasks=32`
- `max_eval_tasks=10`
- `generator.n_samples_per_prompt=1`
- `generator.eval_n_samples_per_prompt=1`
- `harbor_trial_config.environment.type=docker`
- `harbor_trial_config.environment.override_cpus=2`
- `harbor_trial_config.environment.override_memory_mb=4096`
- `harbor_trial_config.environment.override_storage_mb=4096`
- `harbor_trial_config.agent.override_timeout_sec=900`
- `harbor_trial_config.agent.kwargs.max_turns=24`
- `harbor_trial_config.agent.kwargs.enable_summarize=false`
- `harbor_trial_config.agent.kwargs.store_all_messages=true`
- `harbor_trial_config.agent.kwargs.temperature=1.0`（默认 smoke，未额外压到 0）

重要说明：

- `main_harbor_generate` 会加载 `val_data`，所以 `10` 个 `OpenThoughts-TB-dev` 任务路径会被验证。
- 但 generate-only 入口实际生成 trajectory 时只使用 `train_dataset`，也就是本次实际 rollout 的是 `32` 个 `CodeContests` 任务。
- 这不影响 smoke 目标，因为 smoke 关注的是 rollout 基础设施闭环，不是完整 eval 回路。

## 6. 完整可复现命令

在仓库根目录执行下面这条命令：

```bash
bash examples/train_integrations/harbor/run_codecontest_8xh100_docker.sh smoke \
  trainer.run_name=codecontest-8xh100-docker-smoke-plan32 \
  harbor_trial_config.trials_dir=/home/hkang/zthunder_yagent/codecontest-8xh100-docker-smoke-plan32/trials_run \
  trainer.log_path=/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-smoke-plan32 \
  trainer.ckpt_path=/home/hkang/zthunder_yagent/codecontest-8xh100-docker-smoke-plan32/ckpts \
  trainer.export_path=/home/hkang/zthunder_yagent/codecontest-8xh100-docker-smoke-plan32/exports \
  trainer.resume_mode=disable
```

这条命令的目的：

- 保持脚本里的 plan smoke 默认参数不变
- 用单独的 `run_name / trials_dir / log_path / ckpt_path / export_path`，避免和之前的试验产物混在一起
- `resume_mode=disable`，避免意外复用旧状态

## 7. 运行中如何观察

### 7.1 观察主进程和 rollout actor

```bash
pgrep -af 'main_harbor_generate|ray::skyrl_entrypoint|AsyncVLLMInferenceEngine|VLLM::EngineCore'
```

### 7.2 观察 GPU 占用

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

期望：

- `GPU 0-3` 被 rollout engine 占用
- 每张大约 `15-16 GiB` 显存用于加载 `Qwen3-8B`

### 7.3 观察基础设施日志

```bash
latest=$(ls -1t /home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-smoke-plan32/infra-*.log | head -n 1)
echo "$latest"
tail -f "$latest"
```

关键成功信号：

- `Found 32 valid task directories`
- `Found 70 valid task directories`
- `HarborTaskDataset limiting to 10 tasks`
- `InferenceEngineClient initialized with 4 engines`
- `Generating Trajectories: 0/32`
- Harbor task 的 `/chat/completions` 请求开始出现

### 7.4 观察 trial 产物是否落盘

```bash
find /home/hkang/zthunder_yagent/codecontest-8xh100-docker-smoke-plan32/trials_run \
  -maxdepth 2 -name result.json | wc -l
```

### 7.5 查看当前已完成 trial 的 reward 和异常

```bash
find /home/hkang/zthunder_yagent/codecontest-8xh100-docker-smoke-plan32/trials_run \
  -maxdepth 2 -name result.json -print0 | \
  xargs -0 jq -r '[.task_name, (.verifier_result.rewards.reward // null), .exception_info.exception_type] | @tsv'
```

## 8. Smoke 通过的判定方法

这次实际采用的判定标准：

1. `32` 个训练任务子集被正确加载
2. `4` 个本地 rollout engine 成功启动并加载模型
3. Harbor generator 进入真实生成阶段，而不是初始化阶段就失败
4. Docker sandbox 真正执行任务
5. `result.json` 实际写出
6. `verifier_result.rewards.reward` 非空，说明 reward 已真实回传
7. 没有出现基础设施级别的系统性失败

## 9. 本次实际观测结果

本次 run 在满足 smoke 目标后，我手动停止，避免继续占用 GPU。停止前观测到：

- 产物目录：`/home/hkang/zthunder_yagent/codecontest-8xh100-docker-smoke-plan32/trials_run`
- 日志目录：`/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-smoke-plan32`
- 已完成 trial：`20`
- 有 reward 返回：`20/20`
- 正 reward：`1/20`
- 零 reward：`19/20`
- 基础设施失败：`0/20`

这说明：

- smoke 目标已经达到
- 当前的主要瓶颈不再是 infra，而是 agent 对 coding 任务的协议遵循和求解质量

## 10. 停止和清理命令

如果只是为了验证 smoke，不想继续占用 GPU，可以手动停止：

```bash
pkill -f 'codecontest-8xh100-docker-smoke-plan32|examples.train_integrations.harbor.entrypoints.main_harbor_generate' || true
sleep 5
pgrep -af 'main_harbor_generate|ray::skyrl_entrypoint|AsyncVLLMInferenceEngine|VLLM::EngineCore' || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

期望：

- 不再有相关进程
- `8` 张 GPU 都回到接近空闲状态

## 11. 额外验证过的 protocol 调试参数

下面这组参数已经验证可以正确注入运行时配置：

```bash
harbor_trial_config.agent.kwargs.temperature=0.0
harbor_trial_config.agent.kwargs.llm_call_kwargs.extra_body.chat_template_kwargs.enable_thinking=false
harbor_trial_config.agent.kwargs.llm_call_kwargs.extra_body.include_reasoning=false
```

我实际使用过的命令是：

```bash
bash examples/train_integrations/harbor/run_codecontest_8xh100_docker.sh smoke \
  trainer.run_name=codecontest-8xh100-docker-smoke-plan32-protocol \
  harbor_trial_config.trials_dir=/home/hkang/zthunder_yagent/codecontest-8xh100-docker-smoke-plan32-protocol/trials_run \
  trainer.log_path=/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-smoke-plan32-protocol \
  trainer.ckpt_path=/home/hkang/zthunder_yagent/codecontest-8xh100-docker-smoke-plan32-protocol/ckpts \
  trainer.export_path=/home/hkang/zthunder_yagent/codecontest-8xh100-docker-smoke-plan32-protocol/exports \
  trainer.resume_mode=disable \
  harbor_trial_config.agent.kwargs.temperature=0.0 \
  harbor_trial_config.agent.kwargs.llm_call_kwargs.extra_body.chat_template_kwargs.enable_thinking=false \
  harbor_trial_config.agent.kwargs.llm_call_kwargs.extra_body.include_reasoning=false
```

用途：

- 这组参数是 protocol/debug 优化，不是 smoke 成功的必要条件
- 主要用于减少 `<think>` 和 JSON 解析噪音
- 不建议直接把 `temperature=0.0` 当作 RL rollout 的长期默认值

## 12. 当前剩余问题

1. 正 reward 稀疏
- 默认 smoke 已经出现非零 reward
- 但比例不高，说明 coding agent 的稳定性还不够

2. 长上下文风险仍在
- Harbor 多轮交互仍可能把 prompt 顶到上限
- 这是后续 pilot 需要继续盯的项
