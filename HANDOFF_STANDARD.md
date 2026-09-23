# 实验产物整理与上传规范（交接说明）

> 以 drkernel-8b STTS 实验的归档为参照样板：`StruTune` 仓库 `dr-kernel-h100/` 目录
> （https://github.com/seerain300/StruTune，整理完成于 2026-09-22）。
> 其他实验线照此标准整理后，放入同一仓库的各自子目录（如 `kda-h100/`、`ksearch-h100/`）。

## 一、目录结构标准

```
<实验名>-<硬件>/            ← 例：dr-kernel-h100
├── README.md              ← 必须自足：实验背景、协议参数、结果总表、目录导航、复现要点
├── summary/               ← 便捷查询层（最重要，让不熟悉实验的人 30 秒能查到结果）
│   ├── results_table.md   ← 结果总表（人类可读，含口径说明）
│   ├── results_table.csv  ← 同数据 CSV（程序可读）
│   └── best_solutions/    ← 每个解出题的最优代码，文件头注释写明：
│                             来源批次/采样/轮次、feedback 指标、终局复评 speedup
│   （如有毫秒级数据：加 best_timing_detail.md/.csv，逐 workload 列 axes/ref_ms/sol_ms/speedup）
├── tasks/<题目>/<批次>/    ← 原始产物，按"题目"组织（不是按运行日期分 batch 目录）
│   ├── prompt.txt / state.json / summary.json / best_solution.py / final/
│   └── s<k>t<n>/          ← 每轮：response.txt(模型原话) / solution.py / evaluation.json / .log
├── batch_logs/            ← 批级原始日志：campaign.log（进度）、tokens.jsonl（记账）、results.jsonl
├── scripts/               ← 流水线全部脚本（剔除 __pycache__）
├── evaluators/            ← 评测器（若与实验线绑定）
└── task_plan.json 等      ← 题目定义/配置清单（含参考实现）
```

## 二、README.md 必须写清的五件事

1. **实验协议**：轮次/迭代结构、采样参数（温度/top-p/seed）、reward 公式、合规门控方式
2. **结果总表**：每题一行——pass@1、终局 speedup、最优解来源；注明口径
   （如 "final speedup = 官方评测器全新复评，warmup 3 + 100 iters"）
3. **口径与硬件**：加速比与硬件绑定（A800 ≠ H100，引用旧数字必须标注）；缓存评测与
   复评数字不可混用
4. **目录导航**：读者想查"某题某轮模型说了什么/评测判了什么"时按路径能直接找到
5. **关键工程事项**：影响复现的坑（例：vllm 0.23 需在请求里显式传 stop_token_ids；
   评测外来进程防污染机制及触发次数）

## 三、整理规则

- **不改动原始目录**：整理=复制到新归档目录，原实验工作区保持可续跑状态
- **不进归档的东西**：参考延迟缓存 refcache（与硬件绑定且体积大）、模型权重、数据集、
  `__pycache__`、服务日志
- **评测原始输出全保留**：逐轮 evaluation.json（per-workload 明细：status/ref_ms/sol_ms/
  speedup/axes）和终局复评 JSON 一份不丢；summary 只是加索引，不是替代
- **未解出的题也要进表**：标注"未解出"，不留空白让人猜
- **多批次合并展示**：同一题跑过 1 采样和 7 采样的，tasks/ 下两个批次子目录并存，
  summary 总表里合并成一行对比展示

## 四、上传流程（GitHub）

1. 仓库：https://github.com/seerain300/StruTune.git，各实验线一个子目录，互不干扰
2. 认证：fine-grained PAT（Repository access 选 All repositories 或至少勾 StruTune；
   Permissions → Contents → **Read and write**。注意：已创建的 token 不能事后扩大
   仓库范围，需要新建）。凭据已配置在本机 `~/.git-credentials`
3. 步骤：
   ```bash
   cd <归档目录> && git init
   git config user.name "seerain300" && git config user.email "seerain300@users.noreply.github.com"
   git remote add origin https://github.com/seerain300/StruTune.git
   git fetch origin main && git reset --soft origin/main   # 落在远程历史上，避免冲突
   git add -A && git commit -m "<实验线> artifacts on <硬件>: <一句话>"
   git push -u origin main
   ```
4. 推送后用 API 或网页抽查远程目录是否完整（注意 GitHub API 匿名调用有速率限制，带 token 查）

## 五、校验清单（上传前过一遍）

- [ ] 题目目录数、表格行数、best_solutions 文件数三者与实验记录一致
- [ ] 每个 tasks/<题>/<批次>/ 都有 state.json / summary.json / prompt.txt
- [ ] 轮次目录数与各批协议一致（如 1 采样≈50 轮、7 采样≈350–390 轮）
- [ ] 无 __pycache__、无关大文件（du -sh 总大小先看一眼，样板约 107MB）
- [ ] README 表格数字与 results.jsonl 一致（用脚本从原始 JSON 生成表格，不要手抄）
- [ ] 推送成功后远程抽查 summary/ 目录文件齐全

---
样板参考：`dr-kernel-h100/summary/results_table.md`（总表）、
`dr-kernel-h100/summary/best_timing_detail.md`（毫秒明细）、
`dr-kernel-h100/README.md`（完整 README 样式）。
