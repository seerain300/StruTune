# K-Search Baseline on A800（MTMC 论文实验线）

> 实验线：K-Search（caoshiyi/K-Search@53c8fab + 本地适配）作为 MTMC 论文的对比 baseline，
> 在 30 题 GPU kernel 优化基准上全自动搜索 Triton kernel。
> 硬件：NVIDIA A800-SXM4-80GB ×8（sm_80），g0050 服务器，2026-09-14 ~ 2026-09-18。
> 加速比数字与本硬件绑定（A800 ≠ H100）；H100 复现见 `notes/k-search_H100_REPRODUCTION_GUIDE.md`（源工作区）。

---

## 1. 实验协议

### 1.1 方法与预算

| 项 | 值 |
|---|---|
| 方法 | K-Search world-model 完整版（决策树式结构搜索，每个 cycle propose→codegen→eval→refine） |
| 生成模型 | AWS-GPT-5.6-Sol @ llmapi.isrc.ac.cn/v1（OpenAI 兼容端点，与 MTMC 主方法同端点同模型） |
| 搜索预算 | 每题 100 轮 = 20 action nodes × 5 attempts/node，stagnation window 5 |
| seed | 0（固定 feedback workload 抽样与本地 RNG） |
| 语言 | Triton（`--language triton`） |

### 1.2 题目集（30 题 / 738 workloads）

| 子集 | 题数 | workloads | 评测器 |
|---|---|---|---|
| FlashInfer-test | 10 | 426 | `evaluators/evaluate.py`（flashinfer-bench 0.1.2） |
| SOL-ExecBench L1 | 10 | 154 | `evaluators/evaluate_sol.py` → SOL 官方 CLI（sol-execbench 1.0.2） |
| SOL-ExecBench L2 | 10 | 158 | 同上 |

题目清单与版本见 `configs/experiment_manifest.json`（v3：094 因 reference 为朴素
Python 逐步循环被剔除、092 顶补；dsa_topk fp8 因 sm_80 无 fp8e4nv 被剔除、dsa_sparse 顶补）。

### 1.3 评测口径（reward）

- **搜索反馈**（优化过程中的信号）：固定 seed 抽样 **5 个 workload**；
  FlashInfer warmup3/iters100/trials5，SOL warmup10/iters100（eval seed 200）。
  reward = mean speedup（ref 与 sol 逐 workload 成对计算）。
- **终局复评**（本归档所有 speedup 数字的唯一口径）：**全量 workloads**、
  官方评测器、同机**独占空卡**、ref 与 sol **同进程成对计时**；
  FlashInfer warmup3/iters100/**trials1**（协议 v0.5），SOL iterations100。
  合规门控：全部 workload 数值正确（definition 内置容差）才 valid，否则 INVALID。
- 正确性判官语义：-inf 对 -inf 视为匹配（见 §5 工程事项 #1）。
- **防作弊**：SOL 官方 driver 自带 reward-hack 检查（monkey-patch/线程注入/惰性输出）；
  防调库守卫见 §3 torch 回退审计。

### 1.4 批次

| 批次 | 内容 | 备注 |
|---|---|---|
| formal_20260914 | FlashInfer 10 + SOL L1 10 | 主批次；其中 gqa_paged_prefill 重跑过一次（判官 bug 修复后，旧运行见 incident 目录） |
| formal2_solL2_20260916 | SOL L2 10 | GPU 池化并发（flock） |
| ablation_strict_nolib_20260917 | L1 中 5 题重跑（强化禁调库 prompt） | 002/007/092 中止（LLM 无视守卫），005/020 完成 |
| ablation_full_feedback_20260917 | 020 从头跑（15wl 全量反馈） | 守卫为默认版 |
| ablation_resume5_fullfb_20260917 | 058 从主批 WM 续跑 5 轮（16wl 全量反馈） | 修复 batch=1 抽样盲区 bug |

---

## 2. 结果总表（终局复评口径，全部 workload）

| # | 任务 | bench | valid | pass | geomean | arith | torch 回退 | 反馈 best | tokens | 解来源 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | dsa_sparse_attention_h16_ckv512_kpe64 | FlashInfer | ✅ | 23/23 | 20.72 | 21.45 | 干净 | 21.0 | 1.62M | formal |
| 2 | gdn_decode_qk4_v8 | FlashInfer | ✅ | 54/54 | 225.25 | 437.05 | 干净 | 338.4 | 1.39M | formal |
| 3 | gdn_prefill_qk4_v8 | FlashInfer | ✅ | 100/100 | 195.33 | 231.58 | 干净 | 227.7 | 1.87M | formal |
| 4 | gemm_n4096_k4096 | FlashInfer | ✅ | 43/43 | **0.84** | 0.84 | **A·核心靠库(matmul)** | 0.84 | 0.85M | formal |
| 5 | gqa_paged_decode_h32_kv8 | FlashInfer | ✅ | 48/48 | 124.87 | 154.77 | 干净 | 178.9 | 3.18M | formal |
| 6 | gqa_paged_prefill_causal_h32_kv8 | FlashInfer | ✅ | 38/38 | 392.86 | 717.08 | 干净 | 873.8 | 1.79M | formal(重跑) |
| 7 | gqa_ragged_prefill_causal_h32_kv8 | FlashInfer | ✅ | 21/21 | 11.07 | 13.44 | 干净 | 21.5 | 1.57M | formal |
| 8 | mla_paged_decode_h16_ckv512 | FlashInfer | ✅ | 47/47 | 28.40 | 32.60 | 干净 | 36.8 | 1.94M | formal |
| 9 | mla_paged_prefill_causal_h16_ckv512 | FlashInfer | ✅ | 38/38 | 173.17 | 296.01 | 干净 | 342.4 | 1.82M | formal |
| 10 | rmsnorm_h4096 | FlashInfer | ✅ | 14/14 | 3.47 | 4.42 | 干净 | 4.9 | 0.95M | formal |
| 11 | 002_vae_conv3x3_groupnorm | SOL-L1 | ✅ | 20/20 | 1.39 | 1.40 | **A·核心靠库(conv2d×2)** | 1.44 | 2.17M | formal |
| 12 | 005_conv_gated_projection | SOL-L1 | ✅ | 16/16 | 1.42 | 1.42 | 干净(消融替代†) | 1.56 | 1.31M | formal+strict† |
| 13 | 007_hyena_fft_size_padding_rfft | SOL-L1 | ✅ | 16/16 | 1.38 | 1.39 | **A·核心靠库(rfft×2)** | 1.31 | 1.21M | formal |
| 14 | 008_expert_weighted_index_add | SOL-L1 | ✅ | 16/16 | 2.70 | 2.88 | 干净 | 3.26 | 0.93M | formal |
| 15 | 018_fused_rope_qk_norm_kv_cache | SOL-L1 | ✅ | 13/13 | 21.24 | 22.44 | 干净 | 23.2 | 1.60M | formal |
| 16 | 020_patch_merger_spatial_shuffle | SOL-L1 | ✅ | 15/15 | 2.16 | 2.25 | **C·混合贡献(linear×2)** | 2.92 | 2.26M | formal |
| 17 | 053_gaussian_topk_sparse | SOL-L1 | ✅ | 12/12 | 31.81 | 49.52 | 干净 | 33.3 | 1.51M | formal |
| 18 | 058_moe_radix_sort_prefix_sum | SOL-L1 | ✅ | **16/16** | 10.32 | 10.33 | 干净 | 11.3 | 1.21M | formal+resume5* |
| 19 | 070_mamba2_intra_chunk | SOL-L1 | ✅ | 14/14 | 192.99 | 225.33 | 干净 | 375.5 | 1.78M | formal |
| 20 | 092_gqa_attention_qk_norm | SOL-L1 | ✅ | 16/16 | 2.84 | 2.90 | B·自研为主(linear×4) | 3.15 | 1.55M | formal |
| 21 | 012_moe_batched_execution | SOL-L2 | ✅ | 16/16 | 1.44 | 1.44 | C·待消融(bmm×3) | 1.45 | 1.77M | formal2 |
| 22 | 015_audio_sinusoidal | SOL-L2 | ❌ | 15/16 | 1.03 | 1.03 | **A·核心靠库(conv2d×3)** | 1.21 | 1.43M | formal2 |
| 23 | 030_flux_concat_split | SOL-L2 | ✅ | 16/16 | 1.38 | 1.38 | 干净 | 1.42 | 1.24M | formal2 |
| 24 | 036_convnextv2_nhwc_bwd | SOL-L2 | ✅ | 14/14 | 364.24 | 479.28 | B·自研为主(mm×4) | 614.0 | 3.42M | formal2 |
| 25 | 040_altup_bwd | SOL-L2 | ✅ | 16/16 | 17.40 | 18.31 | 干净 | 23.2 | 3.35M | formal2 |
| 26 | 043_mamba_chunk_scan | SOL-L2 | ✅ | 16/16 | 9.17 | 10.80 | 干净 | 8.5 | 2.43M | formal2 |
| 27 | 049_topk_routing | SOL-L2 | ✅ | 16/16 | 8.94 | 8.97 | 干净 | 10.7 | 1.77M | formal2 |
| 28 | 051_hyena_forward | SOL-L2 | ✅ | 16/16 | 2.35 | 2.49 | C·待消融(addmm,rfft) | 2.28 | 3.02M | formal2 |
| 29 | 057_residual_coupling | SOL-L2 | ❌ | 9/16 | 1.04 | 1.04 | 干净 | 1.05 | 1.80M | formal2 |
| 30 | 080_moe_shared_expert_bwd | SOL-L2 | ✅ | 16/16 | 6.03 | 6.24 | B·自研为主(matmul×7) | 4.8 | 1.37M | formal2 |
| - | 094_time_decay（v3 剔除） | SOL-L1 | — | — | — | — | 干净 | 18347* | — | excluded |

\* 058 主批终评 15/16 INVALID（batch=1 workload 数值失败——该 shape 不在反馈抽样集内，
搜索从未见过）；全量反馈续跑 5 轮后修复为 16/16 valid（10.32x）。本表按修复后数字展示，
两版原始 JSON 都在 `tasks/058_.../` 下的两个批次目录。
† 005 主行用强化守卫消融解（ablation_strict_nolib_20260917）替代展示：formal 解调
F.linear×2（1.57x）；全 Triton 消融解终评 16/16 valid、1.42x（-10%）。两版 JSON 均在
`tasks/005_.../` 下；formal 版源码保留于 `best_solutions/005_...@formal_linear.py`。
\* 094 反馈 best 18347x 为抽样口径（ref 为朴素 Python 循环），因评测成本 4h+/题于 manifest v3 剔除。

**汇总**：valid 28/30（93%）；geomean 范围 0.84x~392.86x；全批 token 总消耗 ~54M
（input 38.8M / cached 0.98M / output 15.3M / reasoning 4.4M，4,426 次调用）。
机器可读版：`summary/results_table.csv`。

### 2.1 torch 回退审计（人工逐行读 run() 源码裁定）

分类标准："如果把库调用换成最朴素实现，这题还能拿到这个加速比吗？"

| 分类 | 数量 | 题 | 特征 |
|---|---|---|---|
| 干净 | 20 | 005†(消融替代) | run() 零库调用，加速全部来自自研 Triton |
| B·自研为主 | 3 | 092/036/080 | 库只做投影/小 GEMM 辅助 |
| C·混合/待消融 | 3 | 020/012/051 | 020 消融证明库贡献 32%（1.46x vs 2.16x） |
| A·核心靠库 | 4 | gemm/002/015/007 | 核心计算调 cuBLAS/cuDNN/cuFFT；终数全部 ≤1.39x |

† 005 formal 解原判 B·自研为主（linear×2，1.57x）；消融去除 F.linear 后全 Triton 终评
1.42x（-10%），主表按消融版展示，归入干净类。

A 类 4 题终数全部 ≤1.39x——加速上限被库调用封死；与 gemm 的"r1-59 自研全败、r60 起退化调库"
行为史一致（见 `tasks/gemm_.../formal_20260914/campaign_stdout.log`）。
上游 K-Search 的防回退守卫（HEAD commit 53c8fab）只注入了 KernelBench 后端，
FlashInfer/SOL 路径无守卫——这是官方代码在本任务源上的原生行为，如实记录。

### 2.2 消融实验

| 实验 | 配置 | 结果 |
|---|---|---|
| 强化守卫（BANNED 清单+REJECTED 警告） | 005/020 重跑 | 成功消除 F.linear（全 Triton）：005 1.57→1.42x（-10%，**已替代进主表**）、020 2.16→1.46x（-32%）；002(conv2d)/007(rfft)/092(linear) LLM 无视守卫仍调库（写不出正确的 Triton conv/FFT），中止 |
| 全量反馈（15wl 替代 5wl 抽样） | 020 从头 100 轮 | 2.16→2.20x（+2%），提升微弱 |
| 全量反馈续跑 | 058 从主批 WM 续 5 轮 | **15/16 INVALID → 16/16 valid**：batch=1 抽样盲区 bug 5 轮修复——证明"不知道"≠"不会修" |

详见 `summary/ablations_table.md`。

---

## 3. 口径与硬件注意事项

1. **加速比 = reference_latency / solution_latency，同机同进程成对计时**。
   reference 是基准集自带的官方实现（不是优化基线）：多数 SOL 题 ref 为朴素 PyTorch，
   因此 070/036 等高倍 speedup 是"融合 Triton vs 朴素实现"的比值；gemm 的 ref 是
   cuBLAS（torch.matmul），所以 0.84x 意味着未超过库。两类数字不可直接互比。
2. **A800 数字**：本归档全部数字在 A800-SXM4-80GB 上测得。H100 复现需重建
   reference 延迟缓存（key 含硬件名），数字会不同。
3. **搜索反馈 vs 终局复评**：反馈是 5wl 抽样口径（列"反馈 best"），只在搜索中作信号；
   本 README §2 表格的 speedup 全部是终局全量复评口径，两者不可混用。
4. **trials 语义**：FlashInfer 终评 trials=1（协议 v0.5，2026-09-15 用户确认）；
   SOL 官方 driver 本身即单次校验+计时。

---

## 4. 目录导航

```
ksearch-a800/
├── summary/
│   ├── results_table.md/.csv    ← §2 总表（脚本从原始 JSON 生成，勿手改）
│   ├── ablations_table.md       ← 消融对照
│   └── best_solutions/          ← 每 valid 题的最终解 .py（文件头注释含来源/指标/审计分类）
│       ├── <题名>.py            ← 主批解（28 个；005 为消融替代版）
│       └── <题名>@<批次>.py     ← 消融/对照代表解（5 个，含 005@formal_linear 调库版）
├── tasks/<题名>/<批次名>/       ← 原始产物（按题目组织；同题多批次并存）
│   ├── campaign_stdout.log      ← 完整搜索日志：WM 决策树状态 + 每轮反馈摘要（想看"模型被要求
│   │                              做什么/判了什么"从这里追；FlashInfer 题的逐轮摘要也在此）
│   ├── stdout_*.log             ← 进程原始 stdout（含 WM JSON 渲染）
│   ├── usage.jsonl              ← token 记账（逐调用：input/cached/output/reasoning）
│   ├── solution.json            ← K-Search 返回的最终解（唯一保存的那个 best）
│   ├── solution_db.jsonl        ← 每个 cycle 最优解的完整代码+eval（中间解审计链）
│   ├── world_model.json         ← 终态世界模型（决策树）
│   ├── feedback_traces/         ← 逐轮评测明细 JSONL（SOL 题，100 个/题：per-workload
│   │                              status/latency/ref_latency/speedup；FlashInfer 题无，
│   │                              用 campaign_stdout.log 的摘要行）
│   ├── final/                   ← 终局复评：evaluation.json(FI) 或 performance.json(SOL)
│   │                              + candidate.json（含 token 汇总）+ solution.sol.json
│   └── final_eval.log / exit_code / DONE
├── batch_logs/                  ← 批级 campaign 日志 + token_summary.json（30+消融汇总）
├── scripts/                     ← 全部流水线脚本 + K-Search_mods/（本地修改的 8 个源文件）
│                                  + sitecustomize.py（-inf 判官补丁，勿删）+ audit_torch_fallback.py
├── evaluators/                  ← evaluate.py / evaluate_sol.py（sha256 溯源见 PROVENANCE.txt）
└── configs/                     ← experiment_manifest.json（题目范围 v3）+ protocol.md（协议全文）
```

**审计链**：总表一行 → `tasks/<题>/<批>/final/*.json`（终评明细）→ `solution_db.jsonl`
（中间解）→ `campaign_stdout.log`（每轮决策与反馈）→ `usage.jsonl`（每次 LLM 调用）。

特殊目录：
- `tasks/gqa_paged_prefill_.../incident_broken_judge_evidence/`：判官 bug 冤案的证据运行
  （当时幸存解全量复评 305x；修复后重跑 393x）
- `tasks/094_..._excluded/`：v3 剔除题的全部尝试（含超时 bug 事故现场）

---

## 5. 关键工程事项（影响复现）

1. **-inf 判官坑（最重要）**：flashinfer_bench 原版正确性检查对输出含 ±inf 的候选直接判
   死，而 gqa_paged_prefill 的 reference LSE 合法含 -inf（空 causal 前缀）→ 正确解
   （含 reference 本身）100 轮全灭。修复 = monkey-patch 对齐官方评测器的 -inf==-inf 语义，
   经 `scripts/sitecustomize.py`（PYTHONPATH 注入 isolated-runner 的 spawn 子进程）生效。
   **复现必须带上这个 shim。**
2. **ziming 评测器本地副本**：原始路径（ziming/MTMC-baseline/agent-generation/）曾被移走，
   归档内 `evaluators/` 是逐字节副本（sha256 见 PROVENANCE.txt）。
3. **SOL 慢 ref 超时**：`evaluate_sol.py --timeout` 是整批 workloads 一个子进程的总超时
   （默认 300s）。reference 含逐时间步 Python 循环的题全量评测需数小时（094 即因
   4h+/题被剔除）。上线新题先跑 reference-as-candidate 探针测 ref 延迟。
4. **GPU 独占**：评测必须独占空卡（同卡租户任务会污染计时）；启动前后各查一次
   `nvidia-smi -i N --query-compute-apps`。
5. **端点缓存行为**（受控实验实测）：LLM 网关为逐字节精确匹配缓存（改 1 token/截断/中部
   改全不命中；命中≈input−3；无部分前缀缓存），K-Search 的动态 prompt 命中率仅 ~3%，
   多 key 无用（缓存不被并发请求驱逐）。
6. **reference 延迟缓存**：搜索反馈侧 ref 延迟首轮落盘复用（省 71h/全批）；终局复评
   不用缓存（同进程成对计时）。缓存 key 含计时参数与硬件名。分析见
   `notes/REF_CACHE_ANALYSIS.md`（源工作区）。
7. **ksearch-run.sh 的 key 优先级**：调用方 LLM_API_KEY > K_SEARCH_KEY(llm.env) > 兜底；
   llm.env 的 export 等号两边不能有空格。
8. **GPU 池化**（formal2 批起）：每卡一把 flock 文件锁，多任务并发共享卡池（benchmark
   独占、LLM 阶段不占卡）；实现 `scripts/K-Search_mods/k_search/utils/gpu_pool.py`。
   编排脚本的"自动复扫空卡"仅在 `--gpus auto` 模式生效（手动指定卡是硬约束——曾因
   复扫越权把 2 卡扩成 6 卡）。
9. **运维规矩**（踩坑实录）：杀进程只按精确 PID（宽 grep 三次误伤）；脱管启动
   （setsid/nohup）一律绝对路径；tee 目标目录先建。

---

## 6. 复现入口

```bash
# 冒烟（单题 2 轮）
CUDA_VISIBLE_DEVICES=<空卡> KSEARCH_STRICT_NO_LIB=1 KSEARCH_RUN_TAG=smoke \
  KSEARCH_MAX_ROUNDS=2 bash scripts/ksearch-run.sh rmsnorm_h4096 0 --wm

# 终局复评
python scripts/ksearch_final_eval.py --task rmsnorm_h4096 \
  --run-dir baseline/ksearch/experiments/smoke/rmsnorm_h4096/run_seed0

# 全流程（30 题 + 评测 + 报告）见 scripts/ksearch_campaign.sh / ksearch_pool_campaign.sh /
# ksearch_final_eval.py / ksearch_merged_report.py
```

K-Search 本地修改的 8 个源文件在 `scripts/K-Search_mods/`（相对仓库根路径一致，
直接覆盖上游 caoshiyi/K-Search@53c8fab 同名文件即可）。
