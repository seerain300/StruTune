# KDA (Kernel Design Agents) — A800 首轮实验归档

> 实验线：KDA agent 工作流（draft → plan → 候选迭代 → 证据账本 → 终局精测），
> 在 A800-SXM4-80GB 80GB (sm_80, 主机 g0056) 上对 30 道 Triton kernel 优化题的完整首轮运行。
> 运行窗口：2026-09-16 21:42 → 2026-09-18（含 09-17 夜间 retry 轮与 09-18 不限 token 续跑/边界形状修复轮）。
> 本归档即 H100 重跑实验（`../kda-h100/`）README 中所引"**A800 旧机同口径终局**"基线的原始数据。

## 1. 实验协议

- **任务集**：30 题 = FlashInfer-Bench 10 + SOL-ExecBench L1 10 + SOL-ExecBench L2 10，
  共 738 workloads（`experiment_manifest.json`，sha256 冻结）。
- **Agent**：Claude Opus 4.8（effort=xhigh，中转站经本地 SSE 修复代理），每阶段独立会话
  （`--resume` 续接；09-18 起每候选 fresh session 以规避无缓存接口的重复计费），
  工具白名单仅文件操作 + KernelWiki skill + 评测入口；prompt 见 `scripts/run_task_stage.py`。
- **时间/额度预算**：原始协议每题 100 万字软限 / 165 万硬限（token budget 口径）；
  09-18 续跑轮解除限额，改 timer_driver 外部计时（60 分钟窗口 + 20 分钟宽限，agent 不知情）。
- **评测协议**（`evaluators/`，与 task.json 的 evaluator_sha256 锁定）：
  - **feedback（搜索内）**：原始协议 = 固定 5 抽样 workload / 100 iterations（seed 0 抽样）；
    09-18 起 L1-005 切换全量粗测做边界形状诊断（版本化，v1 备份在 control/ 下）。
  - **final（终局权威）**：全量 workload / warmup 3–10 / **100 iterations**，参考实现
    同进程成对实时计时（无缓存）；独占空卡 + 外部进程监控，外来 GPU 进程即判废重试。
- **合规门控**：静态检查（必须 `@triton.jit`；禁 torch 计算算子/compile/CPU/NumPy 回退）
  + 最优候选人工审计（tokenize 扫描含方法形式；结果：26 题中 25 干净，
  `L2-012 c006` 含 torch 路由回退，详见 §5）。
- **token 口径**：budget = 未缓存输入 + 缓存写 + 缓存读 + 输出（实验预算口径，非计费）；
  504 失败请求不返回 usage、不入账，真实消耗略高于账面。

## 2. 结果总表

见 `summary/results_table.md`（CSV 同目录；逐 workload 毫秒明细
`summary/best_timing_detail.{md,csv}`；最优代码 `summary/best_solutions/`）。要点：

- **20/30 终局精测通过且 >1×**（最高 gqa_paged_decode 539.16×）；
  另 6 题精测通过但 <1×（正确性确认）；4 题未解出，均标注终态，无空白。
- 全程 token 总账 ≈ **1.79 亿**（首轮 3746 万 + 续跑轮约 1.42 亿）。
- L1/005 与 L1/018 首轮"反馈全过、全量翻车"（边界形状数值超差），经 09-18
  全量反馈诊断轮修复后双双终局通过（1.37× / 15.10×）。

## 3. 口径与硬件

- 全部加速比绑定 **A800-SXM4-80GB (sm_80)**；与 H100（`../kda-h100/`）数字对比必须标注硬件。
- feedback（抽样/粗测，部分题中途切换口径）与 final（全量精测/成对实时）数字不可混用；
  本归档总表两列并列，`final_source` 列给出每题终局数字的原始 JSON 路径。
- token 为 budget 口径累计，含会话归档前历史（各 `control/claude-superseded-*/` 对应的
  preparation 留档在 `batch_logs/campaigns/*.preparation.json`）。

## 4. 目录导航

```
summary/                     30 秒查结果层
  results_table.{md,csv}     结果总表（脚本从原始 JSON 生成，非手抄）
  best_timing_detail.{md,csv} 26 题终局精测逐 workload 毫秒明细（677 行）
  best_solutions/            25 份最优代码（含来源/指标头注释；L2-012 因回退剔除）
tasks/<run_id>/              30 题原始产物（不改动原实验工作区，复制归档）
  workspace/                 agent 工作区：CLAUDE.md/TASK.md 合同、task/ 题目定义、
                             docs/{draft,plan}.md、candidates.jsonl 证据账本（append-only）、
                             runs/candidates/<cXXX>/{feedback,final}.json 逐候选评测明细、
                             final/<cXXX>/ 旧版终局评测路径、*.final-backup
  control/                   控制侧：task.json 可信配置（含版本化限额/预算备份 *-v1.*）、
                             state.json 计数器、candidates/<cXXX>/solution.py sha256 冻结快照、
                             claude/*.jsonl 全部会话转录（模型原话）、
                             claude-superseded-*/ 归档会话、observability.json token 记账、
                             reconciliation.jsonl 账本修复留痕
batch_logs/campaigns/        46 个 campaign 级文件：池/看护/计时事件流（*.events.jsonl）、
                             池日志、preparation 留档、watchdog 状态
scripts/                     全部控制脚本（池调度/单阶段/评测器入口/watchdog/账本对账/
                             限时驱动/建题——即本实验流水线本体）
evaluators/                  evaluate.py (FlashInfer) / evaluate_sol.py (SOL)
notes/                       协议、评测接口文档、两版交接手册、过程档案、两份阶段报告
experiment_manifest.json     题目清单（sha256 冻结）
```

查"某题某候选模型说了什么/评测判了什么"：`tasks/<run_id>/control/claude/*.jsonl`
（模型原话，按 `NNNN-<stage>.jsonl` 编号）与 `tasks/<run_id>/workspace/runs/candidates/<cXXX>/`
（逐 workload 判定）。从总表一行的 `run_id`/`best_candidate` 可直接定位到两侧。

## 5. 关键工程事项（影响复现/引用的坑）

1. **评测外来进程防污染**：final 计时期间同卡出现任何外部 GPU 进程即判废（rc=3）。
   本机为共享机器，09-18 补测期间实际触发 8 次（其他用户铺卡作业/周期性探测），
   均被正确拦截后重试，无污染数字混入；判废记录在各 final.log。
2. **静态检查方法形式盲区**：`x.sort(` / `torch.bincount` / `.cumsum(` 等方法形式
   不在禁止正则内（正则只匹配 `torch.sort(` 前缀形式）。`L2-012 c006` 的 MoE 路由
   用 torch 完成即为该盲区实例——引用其 0.86× 时必须标注"含 torch 路由回退"。
3. **反馈抽样协议的结构缺陷**：固定 5 抽样下，错误恰好落在未抽到形状时 agent 无法
   发现（L1-005 反复盲修的根因）；协议文本写了"样本优先覆盖边界"但 seed=0 随机抽样
   未执行之。修复路径：全量 workload 粗测反馈（L1-005 两窗修复实录见
   notes/reports/kda_30tasks_final_20260918.md §8），已固化为下一轮默认（prepare_batch.py）。
4. **无缓存接口的会话成本**：中转站 prompt 缓存近乎失效（命中 2–16%），`--resume`
   长会话按"轮数 × 全量历史"重复计费（实测放大 10–46 倍，L2-040 一题 3345 万字）。
   续跑轮改每候选 fresh session + 磁盘产物传上下文。
5. **SSE 流式截断**：中转站 Anthropic 流缺结束事件 → 本地代理转非流式重建 SSE；
   代价是单回合生成超 ~300s 即 504（长 draft 类任务通过率低）。
6. **账本不变式**：`state.candidate_evaluations` == `candidates.jsonl` 行数，全程以
   `scripts/reconcile_ledger.py` 维护；修复留痕在各题 control/reconciliation.jsonl。
7. **评测 Python 环境绑定**：FlashInfer 用 ziming 用户的 conda env（mtmc）、SOL 用
   SOL-ExecBench 自带 venv——见 scripts/evaluate_candidate.py 顶部硬编码，换机需改。
