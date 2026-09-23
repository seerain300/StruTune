# StruTune — drkernel-8b STTS 推理评测产物归档（H100, 2026-09-22）

drkernel-8b（hkust-nlp/drkernel-8b）在 H100 80GB 上对 FlashInfer-Test + SOL-ExecBench 题集
（30 题冻结清单的 11 题子集）的 Triton-only 多轮生成-评测实验完整产物。
流水线由旧 A800 机器的迁移包（`migrate_h100_20260920`）部署，协议与旧机对齐。

## 实验协议

- **STTS 多轮**：10 迭代 × 每段最多 5 轮 user turn，段间 best-window 上下文压缩（keep=4，
  按 reward 求和选最优连续窗口），patience=0（不早停，跑满协议）
- **采样**：temperature=1.0 / top-p=0.95 / max-tokens=8192 / seed=20260918
- **合规门控**：AST 静态检测（decoy kernel 未引用 + host 侧 torch 计算黑名单），不合规直接
  不评、reward=0 并反馈违规原因
- **reward**：0.3×compile + 0.4×correct + 0.3×min(geomean_speedup, 3.0)/3.0
- **反馈评测**：全量 workload，warmup 3 / iters 10，参考延迟用 refcache 缓存（H100 实测）
- **终局报告**：每题 best-of-history 解交官方评测器全新复评（correctness-only + 100-iter 计时）

## 两批实验

| 批次 | 题数 | 采样/题 | 说明 |
|---|---|---|---|
| batch1（1sample） | 7 | 1 | 首轮探针：gemm / rmsnorm / mla_paged_decode + SOL 008/053/058/030 |
| batch2（7samples） | 4 | 7 | batch1 失败题加采样重跑：mla / 053 / 058 / 030 |

## 结果总表（见 `summary/results_table.md` / `.csv`）

| 题目 | 1采样 | 7采样 | 最优 speedup |
|---|---|---|---|
| `flashinfer/gemm_n4096_k4096` | ✅ 0.36× | — | 0.36×（正确但慢于 cuBLAS） |
| `flashinfer/rmsnorm_h4096` | ✅ 1.38× | — | 1.38× |
| `flashinfer/mla_paged_decode_h16_ckv512_kpe64_ps1` | ❌ | ❌ | 未解出（47 wl，8 次轨迹尝试未破） |
| `SOL/L1/008_expert_output_weighted_index_add_accumulation` | ✅ 2.82× | — | 2.82× |
| `SOL/L1/053_gaussian_topk_sparse_activation` | ❌ | ✅ 0.86 pass@1 | **12.94×** |
| `SOL/L1/058_moe_expert_token_radix_sort_with_prefix_sum` | ❌ | ✅ 0.57 pass@1 | **6.75×** |
| `SOL/L2/030_flux_concatenated_sequence_processing_with_split` | ❌ | ✅ 0.43 pass@1 | 0.14×（正确但无加速） |

pass@1 = 全部 workload 正确的采样比例；speedup 为官方评测器全新复评的 geomean。

## 目录导航

```
├── summary/                    ← 便捷查询层
│   ├── results_table.md/.csv   ← 上表数据源（含每题最优解的采样/轮次出处）
│   └── best_solutions/*.py     ← 6 个解出题的最优 Triton 代码（头注释含元信息）
├── tasks/<benchmark>/<题名>/
│   ├── 1sample/ 与 7samples/   ← 原始产物（每题实际跑过的批次）
│   │   ├── prompt.txt          ← 发给模型的完整任务 prompt
│   │   ├── state.json          ← 全部轨迹状态（每采样的逐轮 reward/评测摘要/对话窗口）
│   │   ├── summary.json        ← 该题终局汇总（含 best 代码全文）
│   │   ├── best_solution.py    ← best-of-history 解
│   │   ├── final/              ← 终局官方复评输出（correctness/performance JSON+日志）
│   │   └── s<k>t<n>/           ← 每轮产物：response.txt(模型原话) / solution.py(当轮代码)
│   │                              / evaluation.json(逐轮评测：per-workload 明细+speedup) / evaluation.log
├── batch_logs/                 ← 批级原始日志（campaign.log 进度日志 / tokens.jsonl token 记账 / results.jsonl）
├── scripts/                    ← 流水线全部脚本（campaign/stts/fib_eval/pass1 + 启动/监控）
├── evaluators/                 ← 官方评测器（evaluate.py=FlashInfer, evaluate_sol.py=SOL）
└── task_plan.json              ← 30 题冻结清单（含每题 pytorch 参考实现、definition/workload 路径）
```

## 环境与关键工程事项

- 硬件：单机 H100 80GB ×6；vLLM 服务（GPU 0）+ 评测 GPU 池（2/3/4）
- vLLM 0.23.0（复用现有环境）：`--override-generation-config` 的 eos_token_id 不再作为停止条件，
  脚本已在请求里显式传 `stop_token_ids=[151643, 151645]` 修复（否则回复填满 8192 token）
- **评测防污染**：评测期间每 1 秒探测 GPU 计算进程，发现外来进程立即废弃当前评测
  （避免污染计时）→ 等该卡完全空闲 → 从头重跑，不限次数。两批共触发 43 次防护，
  所有进入反馈的计时均为干净评测
- 参考延迟缓存（refcache）与硬件绑定，不随归档分发；复现时首轮自动重测（约 1 小时/7 题）

## 复现要点

1. 数据集：`flashinfer-test`（1.9G，含 blob 输入）+ `SOL-ExecBench`（`data/benchmark`），
   按 `task_plan.json` 中的 definition/workload 路径放置
2. 模型：`hkust-nlp/drkernel-8b`（须含 chat_template.jinja 与 generation_config.json）
3. 环境：runner（openai client）、flashinfer 评测（flashinfer-bench==0.1.2）、
   SOL 评测（SOL-ExecBench 仓库 venv）
4. 启动：`scripts/start_drkernel_gpu0_kbstyle.sh`（vLLM）→ `scripts/run_stts_probe.sh`
   或 `scripts/run_stts_4tasks_7samples.sh`（断点续跑安全，按题跳过已完成）
