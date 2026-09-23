# StruTune

Kernel tuning experiments on shared H100/A800 machines. Each experiment line
lives in its own top-level directory (互不干扰)：

| 目录 | 实验线 | 一句话 |
|---|---|---|
| [dr-kernel-h100/](dr-kernel-h100/) | drkernel-8b STTS @ H100 | 10 题 STTS 评测产物 |
| [dr-kernel-a800/](dr-kernel-a800/) | drkernel-8b STTS @ A800 | 30 题 8-sample×10-iter + 3-turn baseline |
| [kda-h100/](kda-h100/) | KDA @ H100 | 30 题首轮，17 题零回退终评 |
| [kda-a800/](kda-a800/) | KDA @ A800 | 30 题首轮（g0056，sm_80） |
| [ksearch-h100/](ksearch-h100/) | K-Search world-model @ H100 | 30 题，21 题有效终评 geomean-of-geomean 21.10x，守卫审计含 9 题违规留档 |
| [ksearch-a800/](ksearch-a800/) | K-Search @ A800 | 30 题全量评测 + 消融 + torch 回退审计 |

## 仓库用法

- **查结果**：各实验线 `summary/results_table.md`（CSV 同目录）；逐 workload 计时
  看 `summary/best_timing_detail.*`；最优代码在 `summary/best_solutions/`
- **查过程**：`tasks/<题目>/<批次>/` 下有每轮候选代码（rounds/）、逐轮评测明细
  （eval/）、状态账本（world_model/）、终局复评（final/）、token 记账
  （usage.jsonl）——从总表一行可追溯原始 JSON
- **复现**：各实验线 README 含协议、口径说明与复现入口；脚本在 `scripts/`

## 新实验线归档规范

新目录照 `dr-kernel-h100/` 样板，语义六要素齐全（任务合同 / 每候选解 /
每候选评测明细 / 计数状态 / token 记账 / 终局复评），目录名可按实验线形态映射：

```
<实验名>-<硬件>/
├── README.md            # 自足：协议、结果总表、口径（加速比与硬件绑定）、导航、复现要点
├── summary/             # results_table.md+csv、best_solutions/、（可选）best_timing_detail
├── tasks/<题目>/<批次>/ # 原始产物：每轮解代码 + 评测 JSON + 状态账本
├── batch_logs/          # campaign 进度、token 记账、结果流水
├── scripts/ evaluators/ # 流水线脚本与评测器（剔除 __pycache__）
```

规则：原始评测 JSON 全量进归档（summary 只是索引）；未解出的题也进表；不放
refcache / 权重 / 数据集 / 服务日志；整理 = 复制，原工作区保持可续跑。

上传：在上层目录（顶层只含单一实验子目录）`git init` → `fetch origin main` →
`reset --soft origin/main` → 确认 `git status` 无对远程文件的 D 标记 → `add`
本实验子目录 → commit → push。推送后抽查远程根目录其他实验线完好。
