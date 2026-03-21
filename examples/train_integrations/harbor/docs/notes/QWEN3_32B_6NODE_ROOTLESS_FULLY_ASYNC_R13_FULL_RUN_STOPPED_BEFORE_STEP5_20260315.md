# QWEN3-32B 6 节点 Rootless Fully Async R13 Full Run 在 Step 5 前停摆的短版结论

日期：`2026-03-15`

作业号：`1139`

运行名：`codecontest-qwen3-32b-6node-rootless-full-r13-64x4x64-stale2-conc256-20260315_0051`

## 1. 最短结论

这轮不是“trainer 一直等把系统等崩了”，而是：

1. 旧的 rollout program 一直跑到最后，最后一个正常 `result.json` 在 `01:28:01` 落盘。
2. ThunderAgent 也一直活到那个时间点，最后一条 `Released and removed program` 也是 `01:28:01`。
3. 同一秒，Harbor 又创建了 4 个新 trial，但这 4 个 trial 的 `trial.log` 从一开始就是 `0` 字节，说明它们没有真正启动起来。
4. 从这一刻开始，再也没有新的 `result.json`、`trajectory.json`、`exception.txt` 落盘。
5. 所以不是 ThunderAgent 先死，也不是 rollout 很早就完全没产出，而是“最后一批旧任务收尾后，新一批任务没接上”，trainer 才一直卡在等 generation buffer。

## 2. 已证实的事实

### 2.1 rollout 到最后有没有产出

有。

最后一个正常结果文件是：

- [code_contests-9387__5YDyemT/result.json](/home/hkang/zthunder_agent/harbor_runs/codecontest-qwen3-32b-6node-rootless-full-r13-64x4x64-stale2-conc256-20260315_0051/trials_run/code_contests-9387__5YDyemT/result.json)

它的 `finished_at` 是：

- `2026-03-15T08:28:01.093235Z`

这个文件里 `exception_info = null`，说明它是正常完成的，不是异常结果。

### 2.2 ThunderAgent 到最后有没有工作

有。

文件：

- [thunderagent.log](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-r13-64x4x64-stale2-conc256-20260315_0051/thunderagent.log)

最后阶段关键信息：

- 最后一条成功 `/v1/chat/completions ... status=200`：`01:27:41`
- 最后一条 `Released and removed program`：`01:28:01`

这说明：

- ThunderAgent 不是先崩的
- 它是把最后几个 in-flight program 正常收尾了
- 之后没有新 program 再进来

### 2.3 最后真正断在哪

`01:28:01` 同一秒，Harbor 新建了 4 个 trial：

- [code_contests-11244__UvSfoLi/trial.log](/home/hkang/zthunder_agent/harbor_runs/codecontest-qwen3-32b-6node-rootless-full-r13-64x4x64-stale2-conc256-20260315_0051/trials_run/code_contests-11244__UvSfoLi/trial.log)
- [code_contests-11244__hQu9esu/trial.log](/home/hkang/zthunder_agent/harbor_runs/codecontest-qwen3-32b-6node-rootless-full-r13-64x4x64-stale2-conc256-20260315_0051/trials_run/code_contests-11244__hQu9esu/trial.log)
- [code_contests-11244__neRUhDv/trial.log](/home/hkang/zthunder_agent/harbor_runs/codecontest-qwen3-32b-6node-rootless-full-r13-64x4x64-stale2-conc256-20260315_0051/trials_run/code_contests-11244__neRUhDv/trial.log)
- [code_contests-11244__vCPnAPt/trial.log](/home/hkang/zthunder_agent/harbor_runs/codecontest-qwen3-32b-6node-rootless-full-r13-64x4x64-stale2-conc256-20260315_0051/trials_run/code_contests-11244__vCPnAPt/trial.log)

这 4 个目录里：

- `config.json` 已经存在
- `trial.log` 全是 `0` 字节

这说明：

- Harbor 已经走到了“创建新 trial 目录和配置”这一步
- 但新 trial 没有真正进入执行
- 所以后面没有任何新的 rollout 请求继续喂给 ThunderAgent

### 2.4 之后系统为什么看起来还活着

因为它是停摆，不是崩溃。

监控文件一直写到 `01:37` 左右：

- [trainer gpu_summary.tsv](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-r13-64x4x64-stale2-conc256-20260315_0051/trainer_monitors/research-dev-coder-012/gpu_summary.tsv)
- [rollout gpu_summary.tsv](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-r13-64x4x64-stale2-conc256-20260315_0051/rollout/monitoring/gpu_summary.tsv)

但最后采样里：

- 显存还占着
- `util_gpu_pct = 0`

这说明 worker 进程还在，模型也还在显存里，但业务流已经不再前进。

## 3. 那些业务级 trial 失败到底有没有关系

先说清楚：前一版把下面这些东西直接叫成“系统不健康信号”，这个表述是错的。

这轮里确实有这些业务级 trial 失败：

- `OutputLengthExceededError`
- `ContextLengthExceededError`
- `status=400`
- `Server disconnected`

其中前两个在 Harbor 代码里就是明确建模的业务异常，不是基础设施异常：

- [harbor/llms/base.py](/home/hkang/zthunder_agent/SkyRL/.venv/lib/python3.13/site-packages/harbor/llms/base.py)
- [harbor/llms/lite_llm.py](/home/hkang/zthunder_agent/SkyRL/.venv/lib/python3.13/site-packages/harbor/llms/lite_llm.py)
- [harbor/agents/terminus_2/terminus_2.py](/home/hkang/zthunder_agent/SkyRL/.venv/lib/python3.13/site-packages/harbor/agents/terminus_2/terminus_2.py)

更具体地说：

- `OutputLengthExceededError` 是 `finish_reason == "length"` 时 Harbor 主动抛出的
- `ContextLengthExceededError` 是 Harbor 把 provider 的上下文超限错误翻译成自己的异常

所以这两个错误本身只能说明：

- 有一些 trial 在 agent 交互里失败了
- 它们是业务层 outcome 的一部分
- **不能单独拿它们当“系统级停摆根因”**

它们和这次停摆的关系，只能保守说成：

- 它们说明这轮并不是“所有 rollout 样本都成功”
- 但它们并不能解释为什么 `01:28:01` 之后新 trial 完全没接上

真正把系统拖进停摆的直接事件，不是“有几个业务级失败样本”，而是：

- 最后一批旧任务正常收尾后
- 新一批 Harbor trial 没真正启动起来
- 于是后面彻底没有新结果流进 trainer

## 4. 现在能下的最准结论

最准结论就是：

**这轮最后死法是“旧任务跑完了，新任务没接上”，不是“trainer 等着把系统等崩了”，也不是“ThunderAgent 自己先死了”。**

## 5. 还没证实的点

现有日志还不能 100% 证明这 4 个 `0` 字节新 trial 具体卡在：

- Harbor 容器启动
- tmux 初始化
- agent 进程启动前
- 或更早的别的环节

也就是说：

- 我已经能定位到“最后一棒没接上”
- 但还不能只凭现有日志把责任精确打到某一行代码

## 6. 清理结果

这轮训练、rollout、Ray、rootless Docker、监控相关 step 都已经停掉，只保留了 `1139.batch`。

清理验证文件：

- [cleanup_verification.txt](/home/hkang/zthunder_agent/tmp_logs/codecontest-qwen3-32b-6node-rootless-full-r13-64x4x64-stale2-conc256-20260315_0051/postmortem_step5/cleanup_verification.txt)

验证结果：

- `003/008/012-015` 上没有残留 GPU compute 进程
- 没有残留 `6381/8265/8080/18000/18001` 监听
- 没有残留 `raylet`、`gcs_server`、`vllm_server.py`、`dockerd-rootless`、`rootlesskit`
