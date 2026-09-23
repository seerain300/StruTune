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

各目录 README 自足（实验协议、结果总表、口径与硬件绑定、目录导航、复现要点）；
快速查结果直接看各目录 `summary/results_table.md`。

## 整理与上传规范（原 HANDOFF_STANDARD.md，2026-09-23 并入）

新实验线归档时照此执行，样板参照 `dr-kernel-h100/`：

**目录结构**：`<实验名>-<硬件>/` 顶层单一目录，内含
`README.md`（自足）/ `summary/`（results_table.md+csv、best_solutions/、可选
best_timing_detail）/ `tasks/<题目>/<批次>/`（原始产物：合同 prompt、每轮解代码、
逐轮评测 JSON、状态账本）/ `batch_logs/` / `scripts/` / `evaluators/`。
结构不必硬套目录名，按语义映射（任务合同/每候选解/每候选评测明细/计数状态/
token 记账/终局复评六要素齐全即可）。

**硬规则**：① 原始评测 JSON 永远全量进归档（summary 只是索引）；② 同一实验内
命名统一；③ 审计链完整（总表一行能追到原始 JSON 与模型原话）；④ 未解出的题
也进表；⑤ 不进归档：refcache/权重/数据集/`__pycache__`/服务日志；⑥ 整理=复制，
原工作区保持可续跑。

**上传流程**（防误删守卫是血泪教训）：
```bash
cd <含单一实验子目录的上层目录> && git init
git config user.name "seerain300" && git config user.email "seerain300@users.noreply.github.com"
git remote add origin https://github.com/seerain300/StruTune.git
git fetch origin main && git reset --soft origin/main && git branch -M main
git status --short | grep '^ D\|^D' && echo "!! 远程文件被标删除，停止排查"   # 守卫必过
git add <实验子目录> && git commit -m "<实验线> artifacts on <硬件>: <一句话>"
git push -u origin main
```
推送后用 GitHub API（带 token）抽查：远程根目录其他实验线完好 + 本实验线
summary/tasks 文件数与本地一致。**禁止在实验子目录内部 git init**（会把归档
内容当仓库根，`add -A` 静默顶掉其他实验线；2026-09-23 kda-h100 与 ksearch-h100
上传均触发过该守卫）。
