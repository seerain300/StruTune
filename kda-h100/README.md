# KDA (Kernel Design Agents) — H100 首轮实验归档

> 实验线：KDA agent 工作流（draft → plan → 候选迭代 → 证据账本 → 终局精测），
> 在 H100 80GB (sm_90) 上对 30 道 Triton kernel 优化题的完整首轮运行。
> 运行窗口：2026-09-21 20:44 → 2026-09-23（含夜间值守与后续续跑轮次）。

## 1. 实验协议

- **任务集**：30 题 = FlashInfer-Bench 10 + SOL-ExecBench L1 10 + SOL-ExecBench L2 10，
  共 738 workloads（`experiment_manifest.json`，sha256 冻结）。
- **Agent**：Claude Opus 4.8（effort=xhigh），每阶段独立会话（`--resume` 续接），
  工具白名单仅文件操作 + KernelWiki/ncu skill + 评测入口；prompt 见 `scripts/run_task_stage.py`。
- **时间预算**：每题 60 分钟墙钟（draft+plan+全部候选迭代合计），timer_driver 外部计时
  （agent 不知情），评测危险窗口宽限 ≤10 分钟。
- **评测协议**（`evaluators/`，与 task.json 的 evaluator_sha256 锁定）：
  - **feedback（搜索内粗测）**：全量 workload，warmup 2 / 10 iterations，SOL 侧参考延迟缓存；
    仅作候选间排序信号（±5–10% 噪声）。
  - **final（终局权威）**：全量 workload，warmup 3–10 / **100 iterations**，参考实现
    同进程成对实时计时（无缓存）；独占卡 + 外部进程监控，外来 GPU 进程即判废。
- **合规门控**：静态检查（必须 `@triton.jit`；禁 torch 计算算子/compile/CPU/NumPy 回退）
  + 终局方法形式盲区人工审计（`.sort(`/`bincount`/`.cumsum(` 等 tokenize 扫描）。
- **token 口径**：budget = 未缓存输入 + 缓存写 + 缓存读 + 输出（非计费口径）。
- **占卡系统**：评测/剖析窗口自动让位（持有器停 → 等显存释放 → 独占计时 → 空闲 30s 占回）。

## 2. 结果总表

见 `summary/results_table.md`（CSV 同目录）。要点：
**17/30 精测有效**（全部零 torch 回退）；其中 9 题超越 A800 旧机同口径终局
（最高 L1-070 400.79× vs 旧机 231.09×）。未解出 13 题均标注终态，无空白。

## 3. 口径与硬件

- 全部加速比绑定 **H100 80GB (sm_90)**；与 A800 旧机数字对比必须标注硬件差异。
- feedback（粗测/缓存参考）与 final（精测/成对实时）数字不可混用；本归档总表两列并列展示。
- 每题 1 小时预算 vs 旧机平均 3–12 小时——对比时注意时间投入差异。

## 4. 目录导航

```
summary/results_table.{md,csv}  结果总表（含口径说明）
summary/best_solutions/<题>.py  17 个最优候选源码（头部注明来源/两口径指标）
summary/best_timing_detail.csv  最优候选终局 per-workload 明细（ref_ms/sol_ms/speedup/axes）
tasks/<题>/formal-kda-h100-20260920/
  CLAUDE.md TASK.md              工作区合同（agent 行为约束）
  docs/{draft,plan}.md           agent 产出的分析与计划
  candidates.jsonl               证据账本（每评测一行，append-only）
  runs/candidates/cXXX/          每候选 feedback.json/final.json/feedback.log（原始评测明细）
  control_state.json             评测计数与租约元数据（含占卡让位记录）
  control_candidates/cXXX/       候选 sha256 锁定快照（不可变）
  control_transcripts/*.jsonl    agent 每阶段完整会话（模型原话/工具调用/usage）
  SEARCH_COMPLETE / *_REACHED    终态标记（内容含原因与最优候选）
batch_logs/                      campaign/timer/keeper 级日志与时间记账（timer.jsonl 逐窗审计）
scripts/                         全部控制脚本（timer_driver/evaluate_candidate/占卡状态机等）
evaluators/                      评测器（与 task.json sha256 锁定配套）
experiment_manifest.json         30 题冻结清单
campaign.json                    campaign 定义（题↔工作区映射）
```

查"某题模型说过什么"：`tasks/<题>/<批次>/control_transcripts/`（按阶段编号排序）。
查"某候选评测判了什么"：`tasks/<题>/<批次>/runs/candidates/<cXXX>/feedback.json`（per-workload 明细）。

## 5. 关键工程事项（复现必读）

1. **占卡让位时序**：持有器（fill_vram）释放 73GB 需数十秒；评测必须等显存真正释放
   （`evaluate_candidate.occupancy_lease` 轮询 <1GB），否则评测进程 CUDA OOM（16/16
   RUNTIME_ERROR 假象）或被外部进程监控误判废。2026-09-23 修复，历史有 11 例受影响窗口。
2. **复合命令权限拒绝**：agent 若在评测命令后附加 `; echo` 等复合语法，权限白名单整体
   拒绝；agent 会误判"评测被封"而放弃（gemm/gqa_ragged 首轮 30 窗口空转的根因）。
   prompt 已加"裸命令"要求。
3. **中转站特性**：SSE 截断走本地修复代理；prompt 缓存近乎失效 → resume 会话 token 按轮次
   全量上下文重复计费；504 失败请求不返回 usage（真实消耗略高于账面）；**API 探测必须用
   curl**（Python urllib 被 503 指纹拦截）。
4. **1 小时预算的结构性代价**：复杂题（L2 多数、L1-007/008 等）draft+plan 即耗尽预算、
   0 评测；这批题的会话已保留，按"候选阶段单独计时"口径可续跑。
5. **双 A800/H100 提示**：评测器 JIT 编译目标 `TORCH_CUDA_ARCH_LIST=9.0`（H100）。
6. 终局精测的判废重试：外部 GPU 进程污染会消耗 final 名额（rc=3），按 task.json
   版本化上调后重试（`scripts/run_finals.py` 自动处理）。

## 复现要点

```bash
# 建题（30 工作区）→ 每题限时运行 → 精测
python3 scripts/prepare_batch.py --tag <tag>
python3 scripts/timer_driver.py --campaign ... --task-budget-minutes 60
python3 scripts/run_finals.py
# 账本不变式校验（counter == ledger 行数）
python3 scripts/reconcile_ledger.py --campaign ...
```
