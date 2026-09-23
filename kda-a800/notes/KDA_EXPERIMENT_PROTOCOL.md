# KDA 30 题实验协议

> 版本：`kda-experiment-policy-v1`；日期：2026-09-16。
> 机器可读配置：`/data1/workspace/weihongren/experiment_manifest.json`。

## 1. 搜索采样与最终评测

全量 workload 不应在每个候选上重复执行。30 题中部分 FlashInfer 题有
38–100 个 workload；若每次修改都跑全量，评测时间会压缩真正的候选探索预算。

采用两阶段口径：

1. **搜索反馈阶段**：每题固定 seed 0，默认每个候选评测 5 个分层 workload。
2. **最终认证阶段**：只对搜索结束时最好的正确候选执行一次全量 workload 评测。

采样不是每轮重新随机。所有候选必须复用同一个固定样本，才能直接比较。样本优先覆盖：

- shape/sequence/batch/head/page 等轴的最小值和最大值；
- 特殊边界、非整除和尾部形状；
- 剩余名额用确定性的内部样本补齐。

若一题不足 5 个 workload，则使用全部 workload。候选在反馈样本上通过，不等于最终有效；
最终论文或正式表格只能引用全量评测结果。

## 2. 每题预算

### 2.1 候选评测预算

- 搜索阶段每题最多 **100 次候选评测**。
- 一次候选评测指：冻结一版不可变的 kernel 源码，在固定反馈样本上完成一次正确性和性能评测。
- 默认反馈样本包含 5 个 workload；这 5 个 workload 合计仍只算 **1 次候选评测**，不是 5 次。
- 同一源码、同一配置因基础设施故障重试，不增加候选版本，但重试原因和次数需单独记录。
- 源码、编译参数、launch/config 或算法发生有意义变化，必须生成新候选 ID；其评测计为新的一次。
- 编译失败的冻结候选也计 1 次候选评测，状态记为 `invalid`，不得通过 Torch fallback 规避。
- 最终最佳候选的一次全量认证独立记账，不占 100 次候选评测预算；每题最多执行一次。

100 次预算理论上允许最多 100 个冻结 kernel 版本。实际不要求用满；达到晋级条件或继续探索价值不足时应提前停止。应保留预算给：

- 修复 correctness 的候选；
- 对关键边界的定向复测；
- 最终候选确认前的回归检查。

### 2.2 Token 预算

每题按 `input_tokens + output_tokens` 计费：

- **1,000,000 token：软阈值。** 达到后不再启动新候选；已经启动的原子阶段允许完成。
- **1,500,000 token：正常结算阈值。** 当前原子阶段完成后停止模型迭代。
- **1,650,000 token：绝对阈值。** 不再启动任何新阶段；用于限制异常探索和重试放大。
- provider 报告的 cached input 仍属于 input token，因此包含在总预算内；另外单列 cached/uncached 便于成本分析。
- KDA Opus 4.8 启动器必须经过本地 SSE 代理。代理把 Claude Code 的流式请求改为同一次
  上游非流式请求，再重建标准 Anthropic SSE，因此可从完整响应中保留准确的
  `input_tokens`、`output_tokens` 和 cache usage。
- 若未来某次响应仍未报告 output token，必须标记 `usage_incomplete=true`，不能把缺失值当作
  真实的 0；该题同时退化为 input-token 达到 1,000,000 即软停。

停止条件取最先达到者：晋级条件满足、100 次候选评测、1M 软停后完成收尾、
1.2M 硬停、或继续探索已无合理价值。

## 3. 候选版本链

每次有意义的实现变化都生成一个不可复用的候选 ID：

```text
baseline
c001
c002
...
```

禁止覆盖旧候选证据。`candidates.jsonl` 每行至少记录：

```json
{
  "id": "c003",
  "parent": "c001",
  "status": "valid",
  "decision": "promote",
  "source_sha256": "...",
  "change_summary": "...",
  "feedback_workload_ids": ["..."],
  "feedback_workload_count": 5,
  "evaluation_index": 3,
  "cumulative_candidate_evaluations": 3,
  "token_total_at_decision": 615000,
  "evidence_paths": ["..."],
  "created_at": "2026-09-16T00:00:00+08:00"
}
```

版本流程：

1. 保存 baseline 源码和 SHA-256；
2. 从某个已记录 parent 创建新候选；
3. 保存候选源码快照和 SHA-256；
4. 运行 Triton-only 静态检查；
5. 编译及 correctness；
6. 在固定反馈样本上评测；
7. 记录 promote/revise/reject/invalid；
8. 只有当前最佳有效候选可进入最终全量认证；
9. 最终失败时保留失败证据，并根据协议决定修复或回退到次优候选。

建议每个候选保存到：

```text
runs/candidates/<candidate-id>/
  source/
  source.sha256
  validation.json
  feedback.json
  metadata.json
```

## 4. 工具与 Skill 审计

记录工具使用是合理且必要的，但工具次数不能直接当作性能或智能指标。它主要用于：

- 复现实验过程；
- 发现无效循环和异常高成本；
- 确认模型是否读取了任务、评测器和 skill；
- 分析不同模型的工作方式。

按会话记录：

- `Read`、`Write`、`Edit`、`Bash` 等工具调用次数；
- Bash 命令类别，而不是只记录一个总次数；
- `KernelWiki` 和 `ncu-report-skill` 的 loaded/read/command/effective 四级状态；
- tool error、permission error、timeout 和非零退出；
- 首次正确候选前的工具次数、最终候选前的工具次数；
- token、wall time、候选评测次数和每次使用的反馈 workload 数。

Skill 计数定义：

- `loaded`：会话上下文暴露了该 skill；
- `read`：模型读取对应 `SKILL.md`；
- `command`：执行了 skill 目录下的脚本或命令；
- `effective`：产物中保存了该 skill 要求的报告/证据。

仅安装 skill，或在提示词中出现 skill 名字，不计为有效调用。

## 5. 每题最终产物

```text
TASK.md
CLAUDE.md
docs/draft.md
docs/plan.md
docs/final.md
candidates.jsonl
evaluations.jsonl
workload_results.jsonl
tool_usage.json
token_usage.json
runs/candidates/
outputs/
profile/
final_evaluation/
```

最终汇总必须明确区分：搜索反馈结果、最终全量结果、模型 token、工具调用、skill 使用、
候选版本链和停止原因。

Claude transcript 可用以下命令生成基础审计汇总：

```bash
python /data1/workspace/weihongren/kda-observability/summarize_claude_transcript.py \
  <transcript.jsonl> --output <task-workspace>/tool_usage.json
```

该脚本不保存 prompt、response 或 API key。它统计工具、skill 路径访问和 usage 字段，并在
中转站未可靠返回 output token 时设置 `usage_incomplete=true`。
