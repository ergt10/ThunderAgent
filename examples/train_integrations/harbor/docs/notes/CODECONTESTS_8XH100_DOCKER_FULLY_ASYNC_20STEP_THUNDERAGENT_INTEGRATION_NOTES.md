# CodeContests 8xH100 Docker Fully-Async 20-Step ThunderAgent Integration Notes

日期：2026-03-08

## 1. 目标

这份文档记录的是把 ThunderAgent 接到现有 `SkyRL + Harbor` fully-async 20-step 验收配置上的一次 bring-up 过程。

要回答的问题有三个：

- ThunderAgent 是否被正确导入并真正接入 Harbor 训练链路，而不只是“代码能 import”
- 在不改 Harbor 任务配置主体的前提下，能否完成一次 `20 step` fully-async 训练
- 这次 bring-up 过程中哪些 patch 是必要的，分别是在修什么问题

## 2. 结论

结论很直接：

- ThunderAgent 已经正确接入 `SkyRL + Harbor`
- 这次验证不只是 import 成功，而是完整跑完了 `20 step`
- 成功 run 没有再出现 `Missing completion token ids`
- 这次额外暴露并修掉的核心问题，不在 Harbor，而在新 HTTP inference 路径下 `VLLMServerActor` 的 control-plane 接口不完整

成功 run：

- 运行目录：`/scratch/hkang/codecontest-8xh100-docker-fully-async-20step-r8g-thunderagent-default`
- 日志目录：`/home/hkang/zthunder_yagent/tmp_logs/codecontest-8xh100-docker-fully-async-20step-r8g-thunderagent-default`

简单统计：

- `num_trials = 80`
- `num_results = 80`
- `num_exceptions = 6`
- `num_timeouts = 6`
- `num_missing_completion = 0`
- `num_assistant_completion_mismatch = 0`

最终产物：

- `ckpts/global_step_20`
- `ckpts/global_step_21`
- `exports/global_step_21/policy`
- `ckpts/latest_ckpt_global_step.txt = 21`

## 3. 复现步骤

### 3.1 环境前提

- 工作区：`/home/hkang/zthunder_yagent/SkyRL`
- Python：`/home/hkang/zthunder_yagent/SkyRL/.venv/bin/python`
- vLLM：`0.10.2`
- Harbor 数据目录：
  - `/home/hkang/zthunder_yagent/data/harbor/CodeContests`
  - `/home/hkang/zthunder_yagent/data/harbor/OpenThoughts-TB-dev`
- 模型目录：`/data/zy/models`

### 3.2 复现命令

下面这条命令是基于现有 Harbor `r6` 的 20-step 验收参数，只额外打开 ThunderAgent 和新 inference layer：

```bash
cd /home/hkang/zthunder_yagent/SkyRL

_SKYRL_USE_NEW_INFERENCE=1 \
RUN_NAME_OVERRIDE=codecontest-8xh100-docker-fully-async-20step-r8g-thunderagent-default \
RUN_ARTIFACT_ROOT=/scratch/$USER \
SCRATCH_ROOT=/scratch/$USER/skyrl_runtime \
ENTRYPOINT_OVERRIDE=examples.train_integrations.harbor.entrypoints.main_harbor_thunder_agent_fully_async \
PYTHONPATH_PREPEND=/home/hkang/zthunder_yagent/SkyRL/ThunderAgent \
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

### 3.3 成功判定

最小成功标准：

- 训练进度到 `20/20`
- `trainer/global_step = 20`
- 生成 `ckpts/global_step_20`
- 生成 `exports/global_step_21/policy`
- 没有 `Missing completion token ids`
- 没有 assistant / `completion_token_ids` 数量失配

## 4. 这次 bring-up 的核心经验教训

### 4.1 ThunderAgent 不是单纯“加了个 router”

这次最重要的认知更新是：

- ThunderAgent 不是只在原 Harbor 路径前面加一个代理
- 它要求切到 SkyRL 的新 HTTP inference layer
- 一旦切到新路径，很多旧 Harbor run 没覆盖到的问题都会被暴露出来

### 4.2 为什么以前 Harbor 能跑，现在却要补 `/pause` 和 `/resume`

这件事最容易让人误解，结论是：

- `fully-async trainer` 的确一直都需要 `pause_generation()` / `resume_generation()`
- 但以前没接 ThunderAgent 时，Harbor 默认走的是 legacy inference path
- legacy path 用的是 `InferenceEngineClient`，它在 Python/Ray actor 层自己实现 pause/resume
- 这条旧路径不依赖后端 HTTP `/pause`、`/resume`

而这次接入 ThunderAgent 时：

- 必须设置 `_SKYRL_USE_NEW_INFERENCE=1`
- 于是路径切到 `RemoteInferenceClient + router + server_urls`
- 新路径里数据面走 router
- 控制面则直接 fan-out 到后端 `server_urls`
- 所以真正被调用的是后端 server 的 HTTP `/pause`、`/resume`

也就是说：

- 不是 ThunderAgent 平白引入了 pause/resume 需求
- 而是 ThunderAgent 迫使实验切到了“必须通过 HTTP control-plane 调后端”的那条新实现
- 于是 `VLLMServerActor` 原本没补齐的接口第一次被真实打到了

### 4.3 ThunderAgent router 只管 data plane，不管 control plane

这个点也很关键：

- `/v1/chat/completions`、`/inference/v1/generate` 这些数据面请求会经过 ThunderAgent router
- `/pause`、`/resume`、`/sleep`、`/wake_up` 这些控制面请求不会经过 ThunderAgent router
- 它们是 `RemoteInferenceClient` 直接对 backend `server_urls` 发的

所以第一次看到 `POST /pause -> 404` 时，不应该去怀疑 ThunderAgent proxy 转发逻辑，应该先看 backend server 本体有没有提供这个端点。

### 4.4 这次 bring-up 真正修的是“新 inference stack”

从抽象上看，这次不少 patch 虽然是为了把 ThunderAgent 跑起来，但本质上修的是新 HTTP inference 路径，而不是 Harbor 或 ThunderAgent 专属逻辑。

最典型的就是：

- vLLM 0.10.2 API 兼容
- `VLLMServerActor` control-plane 端点
- pause 时阻止新的 generation 请求进入

这些补丁在“启用新 inference layer”的任何路径上都可能有价值，不只是在 ThunderAgent 模式下。

## 5. 必要 patch 说明

### 5.1 新增 Harbor + ThunderAgent 组合入口

文件：

- `examples/train_integrations/harbor/entrypoints/main_harbor_thunder_agent_fully_async.py`

理由：

- 现有仓库里原本只有 Harbor fully-async 入口
- 也有单独的 ThunderAgent example 入口
- 但没有一个现成入口同时把 Harbor dataset/generator/config 和 ThunderAgent inference client 组合起来

这个新入口做的事是：

- 沿用 Harbor 的 dataset 和 generator
- 沿用 `FullyAsyncRayPPOTrainer`
- 继承 ThunderAgent 的 inference client 构造方式

没有这层组合入口，就只能手工改现有 Harbor 入口，复用性和可读性都很差。

### 5.2 让 Harbor 运行脚本支持切换入口和补 PYTHONPATH

文件：

- `examples/train_integrations/harbor/run_codecontest_8xh100_docker_fully_async.sh`

新增点：

- `ENTRYPOINT_OVERRIDE`
- `PYTHONPATH_PREPEND`

理由：

- 原脚本写死了 Harbor 默认入口
- 也默认只把仓库根目录加到 `PYTHONPATH`
- 这不足以支持“同一份 Harbor 验收参数，只切换入口到 ThunderAgent 版本”
- 也不足以支持当前这种本地嵌入的 `ThunderAgent/` 目录导入

这个 patch 的好处是：

- 复用原 Harbor 脚本，不必复制出第二份几乎相同的运行脚本
- 让“带 ThunderAgent 的 Harbor 验收”和“原 Harbor 验收”只在少数环境变量上有差异

### 5.3 补 vLLM 0.10.2 兼容层

文件：

- `skyrl/backends/skyrl_train/inference_servers/utils.py`
- `skyrl/backends/skyrl_train/inference_engines/vllm/vllm_server.py`
- `skyrl/backends/skyrl_train/inference_servers/vllm_server_actor.py`

触发问题：

- `FlexibleArgumentParser` 导入路径变化
- `set_ulimit` 导入路径变化
- `init_app_state(...)` 的函数签名变化

理由：

- 这套新 inference 代码不是严格按当前 `.venv` 里的 vLLM `0.10.2` API 写的
- 不补兼容层的话，server 在正式起起来之前就会因为 import 或签名不匹配直接失败

这类 patch 是纯 bring-up 性质：

- 不改训练语义
- 只是在兼容当前 vLLM 版本

### 5.4 让 ThunderAgent router 不再写死占用 `8080`

文件：

- `examples/train/thunder_agent/thunder_agent_router.py`

触发问题：

- 启动时直接因为 `0.0.0.0:8080` 被占用而失败

修法：

- 启动前调用 `get_open_port(...)`
- 如果默认端口不可用，就自动换到可用端口

理由：

- 训练环境里 `8080` 很容易被别的实验或服务占掉
- 这不是逻辑错误，但会让 bring-up 过程不稳定

这是一个典型的工程性 patch：

- 不改变 ThunderAgent 的路由逻辑
- 只减少无意义的端口冲突失败

### 5.5 给 `VLLMServerActor` 补齐 control-plane 接口

文件：

- `skyrl/backends/skyrl_train/inference_servers/vllm_server_actor.py`

新增能力：

- `POST /pause`
- `POST /resume`
- `POST /sleep`
- `POST /wake_up`
- pause 时阻止新的 generation 请求进入
- `pause(mode=abort)` 时中断 in-flight 请求并清 prefix cache

这是这次最关键的 patch。

触发问题：

- 训练第一步之后，fully-async trainer 会执行：
  - `pause_generation()`
  - 同步权重
  - `resume_generation()`
- 在新 HTTP inference 路径里，这些调用会直接打到 backend `server_urls`
- 原始 `VLLMServerActor` 没有 `/pause`、`/resume`
- 所以会在 step 1 之后直接出现 `POST /pause -> 404`

理由：

- 不补这个 patch，ThunderAgent + Harbor 的 fully-async 训练根本跑不过第一步
- 这不是 ThunderAgent router 的问题，而是 backend server 没实现 SkyRL 新 control-plane 协议

### 5.6 为什么 `/sleep` 和 `/wake_up` 也一起补了

虽然这次把训练跑通主要卡在 `/pause`、`/resume`，但 `/sleep`、`/wake_up` 和这套 control-plane 协议属于同一组能力。

一起补的理由是：

- 避免新 inference stack 后续在别的训练模式下再次暴露“控制面半残”的问题
- 保持 `RemoteInferenceClient` 和 backend server 的接口集合基本一致

## 6. 验证过程中的实际失败顺序

这次 bring-up 不是一把过的，大致经历了下面这几个阶段：

1. 切到 Harbor + ThunderAgent 入口后，先暴露出 `ThunderAgent` 导入路径问题。
2. 补 `PYTHONPATH_PREPEND` 后，开始暴露 vLLM 0.10.2 的 import / API 兼容问题。
3. 补完兼容层后，ThunderAgent router 又因为默认 `8080` 被占用而启动失败。
4. 端口问题解决后，训练真正开始推进，但在第一步训练后的权重同步阶段命中 `POST /pause -> 404`。
5. 给 `VLLMServerActor` 补齐 control-plane 端点后，训练终于可以完整推进到 `20 step`。

这个顺序说明：

- 之前“没跑通”不是一个单点 bug
- 而是一组新路径 bring-up 问题叠在一起

## 7. 这次 run 证明了什么

这次 run 实际证明的是：

- ThunderAgent 已经能被 Harbor fully-async 训练真正使用
- 新入口不是“纸面集成”
- ThunderAgent router 和 Harbor generator 至少在 `20 step` 验收规模上可以共存
- 本地 Harbor 的 rollout-details 根修在 ThunderAgent 模式下仍然成立

这次 run 没有证明的东西：

- ThunderAgent scheduler 本身一定完全正确
- 更长训练或更大规模并发下不会出现新问题
- 所有 control-plane 清理逻辑都已经完美

## 8. 剩余观察项

这次成功 run 之后，仍有一些非 blocker 现象值得后续跟踪：

- 仍有 `AgentTimeoutError`，数量是 `6`
- 退出时还有 `aiohttp` unclosed session warning
- 退出时有 `destroy_process_group() was not called` warning

这些问题没有阻止 `20 step` 验收通过，但如果后面要把这条链路做成更稳定的长期训练入口，仍然值得继续收尾。

## 9. 最短总结

一句话总结这次经验：

- Harbor 原来的 20-step 验收能跑，不代表新 HTTP inference 路径已经成熟
- ThunderAgent 的真正价值之一，是把这条新路径里没补齐的地方都逼出来了
- 这次最重要的 patch，不是 Harbor task 逻辑，而是给 `VLLMServerActor` 补齐 SkyRL 需要的 HTTP control-plane 能力
