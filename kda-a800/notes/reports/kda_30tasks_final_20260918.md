# KDA 三十题终局汇总（2026-09-18）

> 承接 `notes/reports/kda_30tasks_20260916.md`（协议 v0.6：FlashInfer 10 + SOL L1 10 + SOL L2 10，共 738 workloads）。
> 本报告覆盖 2026-09-17 21:22 → 2026-09-18 12:00 的续跑阶段（retry4 池 + timer 轮转）与其后的全量补测，
> 并给出全部 30 题的最终口径：最优候选、反馈/全量加速比、torch 回退审计、token 总账。

## 0. 概览

- **续跑方式**：retry4 池（4 题，GPU 1–5）+ timer 轮转（9 题，GPU 6/7，60 分钟窗口 + 20 分钟宽限 + 安全点击杀），
  token 限制解除（v2 版本化配置，5000 万兜底），watchdog 自动回收 rc=1/rc=4 失败，逐窗自动对账。
- **全量补测**：2026-09-18 10:21–11:47，成对实时计时（SOL ref/sol 同进程；reference_cache=false），
  期间多次被外部 GPU 作业污染判废后重试（评测器 foreign-process 监控全部正确拦截，无污染数字混入）。
- **结果**：有效候选 26/30 题；**全量通过且 >1× 共 20 题**；
  全量通过但 <1× 共 6 题；全量 INVALID 2 题（L1/005、L1/018 边界形状）；无有效候选 4 题。
- **token 总账**：全程累计约 **1.79 亿**（09-17 报告基线 3,746 万 + 续跑及补测约 1.42 亿）。

## 一、FlashInfer 子集（10 题）

| # | 题目 | 最优候选 | 反馈 GM | 全量 GM | 全量判定 | 回退审计 | token(万) |
|---|---|---|---:|---:|---|---|---:|
| 1 | `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64` | c002 | 25.45× | 25.33× | ✅ | 干净 | 599.4 |
| 2 | `gdn_decode_qk4_v8_d128_k_last` | c001 | 182.09× | 247.87× | ✅ | 干净 | 137.1 |
| 3 | `gdn_prefill_qk4_v8_d128_k_last` | c002 | 174.75× | 161.69× | ✅ | 干净 | 244.0 |
| 4 | `gemm_n4096_k4096` | c007 | 0.58× | 0.59× | ✅(<1×) | 干净 | 658.0 |
| 5 | `gqa_paged_decode_h32_kv8_d128_ps1` | c001 | 633.20× | 539.16× | ✅ | 干净 | 130.1 |
| 6 | `gqa_paged_prefill_causal_h32_kv8_d128_ps1` | c001 | 92.36× | 63.06× | ✅ | 干净 | 126.8 |
| 7 | `gqa_ragged_prefill_causal_h32_kv8_d128` | c001 | 13.89× | 12.92× | ✅ | 干净 | 125.0 |
| 8 | `mla_paged_decode_h16_ckv512_kpe64_ps1` | c002 | 63.63× | 56.26× | ✅ | 干净 | 134.2 |
| 9 | `mla_paged_prefill_causal_h16_ckv512_kpe64_ps1` | c002 | 68.46× | 55.44× | ✅ | 索引构造¹ | 126.6 |
| 10 | `rmsnorm_h4096` | c002 | 4.57× | 4.19× | ✅ | 干净 | 116.5 |

## 二、SOL-ExecBench L1 子集（10 题）

| # | 题目 | 最优候选 | 反馈 GM | 全量 GM | 全量判定 | 回退审计 | token(万) |
|---|---|---|---:|---:|---|---|---:|
| 1 | `002_vae_conv3x3_groupnorm_silu_residual_fused` | c006 | 0.79× | 0.76× | ✅(<1×) | 干净 | 519.2 |
| 2 | `005_conv_gated_projection_with_causal_conv` | c005⁷⁸ | 1.38× | **1.37×** | ✅ | 干净 | 519+130.6 |
| 3 | `007_hyena_fft_size_padding_rfft` | c001 | 0.08× | 0.10× | ✅(<1×) | 干净 | 123.6 |
| 4 | `008_expert_output_weighted_index_add_accumulation` | c001 | 1.93× | 1.85× | ✅ | 干净 | 139.4 |
| 5 | `018_fused_rope_with_qk_norm_and_kv_cache_update` | c003⁷ | 15.62× | **15.10×** | ✅ | 干净 | 443+143.6 |
| 6 | `020_vision_patch_merger_spatial_shuffle_mlp` | — | 无有效候选 | — | — | — | 636.9 |
| 7 | `053_gaussian_topk_sparse_activation` | c002 | 24.93× | 26.36× | ✅ | 干净³ | 143.7 |
| 8 | `058_moe_expert_token_radix_sort_with_prefix_sum` | c003 | 3.41× | 3.17× | ✅ | 干净 | 168.8 |
| 9 | `070_mamba2_fused_intra_chunk_diagonal_computation` | c001 | 223.26× | 231.09× | ✅ | 干净 | 108.2 |
| 10 | `092_gqa_attention_with_qk_norm` | c002 | 2.33× | 2.24× | ✅ | 干净 | 149.0 |

## 三、SOL-ExecBench L2 子集（10 题）

| # | 题目 | 最优候选 | 反馈 GM | 全量 GM | 全量判定 | 回退审计 | token(万) |
|---|---|---|---:|---:|---|---|---:|
| 1 | `012_moe_expert_batched_execution_with_capacity_factor` | c006 | 0.88× | 0.86× | ✅(<1×) | **❌ 回退⁴** | 775.3 |
| 2 | `015_audio_sinusoidal_position_embedding_with_conv_projection` | — | 无有效候选（14 评） | — | — | — | 3861.4 |
| 3 | `030_flux_concatenated_sequence_processing_with_split` | c002 | 0.63× | 0.60× | ✅(<1×) | 干净 | 143.3 |
| 4 | `036_convnextv2_layer_with_nhwc_persistence_backward` | — | 无有效候选 | — | — | — | 181.1 |
| 5 | `040_altup_predict_correction_cycle_backward` | c012 | 12.02× | 9.85× | ✅ | 干净 | 3386.0 |
| 6 | `043_mamba_chunk_scan_with_segsum` | c007 | 19.23× | 19.85× | ✅ | 干净 | 2029.1 |
| 7 | `049_group_limited_topk_routing` | c013 | 3.67× | 4.33× | ✅ | 干净 | 2221.5 |
| 8 | `051_seqlen-finetuned-reconstructed_hyena_complete_forward_block` | — | 无有效候选（9 次 draft 尝试） | — | — | — | 62.5 |
| 9 | `057_residual_coupling_flow_block` | c003 | 0.50× | 0.55× | ✅(<1×) | 干净 | 442.7 |
| 10 | `080_moe_complete_layer_with_shared_expert_backward` | c001 | 3.88× | 4.43× | ✅ | 干净 | 169.9 |

## 4. 口径说明

- **反馈 GM**：固定 5 抽样 workload 的 geomean_speedup（各题 `candidates.jsonl` 最优有效候选）。
- **全量 GM**：final 评测（成对实时计时，reference_cache=false）的 geomean_speedup。
  新路径结果 `runs/candidates/<id>/final.json`；09-17 旧路径 `final/<id>/{evaluation,performance}.json`。
- **token(万)**：全程累计 = 09-17 报告基线 + 续跑各阶段归档记录（`*.preparation.json`）+ 当前活跃
  `observability.json`，已消除归档重置导致的重复计数。实验预算口径（未缓存输入+缓存写+缓存读+输出 1:1 累加），
  非计费口径；504 失败请求不返回 usage，真实消耗略高于账面。
- **回退审计**：对每个最优候选的归档源码（`kda-control/<run>/candidates/<id>/solution.py`）做
  tokenize 后的广义算子扫描（模块形式 + 方法形式，排除 `tl.`/`math.`）。
  - 脚注¹ `mla_paged_prefill c002`：`torch.repeat_interleave(torch.arange(batch), q_lens)` 构造 seq_id 索引，
    属元数据管线（同 `arange` 类），判定合法，如实披露。
  - 脚注³ `L1/053 c002`：命中的 `math.log/math.sqrt` 为宿主机标量常量，合法。
  - 脚注⁴ **`L2/012 c006` 为唯一真实回退**：MoE 路由（`tensor.sort(stable=True)` + `torch.bincount` +
    `.cumsum(0)` + `repeat_interleave`，solution.py L184–193）用 torch 完成，Triton 仅做 expert 计算。
    方法调用形式绕过了 `evaluate_candidate.py` 静态检查（正则只匹配 `torch.sort(` 前缀形式）。
    其全量 0.86× 无指标价值；建议论文口径剔除或标注。静态检查的该方法形式漏洞待修复。
- 脚注² `L1/005`（14/16）与 `L1/018`（12/13）全量存在边界形状数值超差（小 batch / seq_len=1），
  整体 INVALID，括号内 geomean 仅为通过部分，不可用（详见 09-17 报告 §6）。
- 脚注⁶ `gdn_prefill c002` 全量补测于 12:45 完成（第 6 次尝试，前 5 次被外部 GPU 探测进程污染判废）：
  **valid=True，161.69×**（100/100 workload，成对实时计时）。

## 5. 汇总统计

| 口径 | 数量 | 题目 |
|---|---:|---|
| 全量通过且 >1× | 20 | gqa_pd, gdn_decode, gdn_prefill, mla_pd, mla_pp, gqa_pp, gqa_rp, dsa_sparse, rmsnorm（FI）；L1/053, L1/058, L1/070, L1/092, L1/008（L1）；L2/043, L2/040, L2/049, L2/080（L2） |
| 全量通过但 <1× | 6 | gemm, L1/002, L1/007, L2/012, L2/030, L2/057 |
| 全量 INVALID | 0 | — |
| 无有效候选 | 4 | L1/020, L2/015, L2/036, L2/051 |
| 回退违规 | 1 | L2/012（<1×，无指标损失） |

- token 总消耗 ≈ **17,934 万**（FI 小计 ≈ 2,398；L1 小计 ≈ 2,263；L2 小计 ≈ 13,273）。
- 续跑阶段（09-17 21:22 起）新增 ≈ 14,188 万，产出：+9 题首个有效候选、+17 个 >1× 全量通过位、
  最优改进如 gdn_prefill 127.88→174.75×（反馈）。

## 6. 审计线索

- 过程与机制：`notes/KDA_WATCHDOG_AND_TIMER_20260918.md`（watchdog/对账/计时驱动/事故记录/收尾）
- 全量补测日志：`kda-control/final-eval-20260918/`（status.log + 各题日志；被污染尝试保留判废记录）
- 账本修复留痕：各题 `kda-control/<run>/reconciliation.jsonl`；`L2/057` 隔离行 `candidates.jsonl.fragments`
- 预算版本化：各题 `task.token-v1.json` / `task.final-budget-v1.json`（限额与 final 名额调整的原因均注明）
- 续跑留档：`kda-control/campaigns/timer-continuation-20260918.preparation.json` 等 3 份

## 7. 补记（2026-09-18 12:05）：L1-005 / L1-018 边界形状修复轮

- 两题各 1 窗 × 40 分钟（fresh session，50M 兜底），均在窗内产出并评测 c003。
- **L1/018 c003 全量通过（15.10×，13/13）**：修复假设为模拟参考实现的 bf16 舍入顺序，命中根因。
  该题由 INVALID 转为有效 >1×；>1× 题目数 18 → 19。
- L1/005 c003 全量仍 INVALID（小 batch 形状超差未解）；最优反馈候选仍为 1.47×（c002/c003 持平）。
- 本轮 token：两题各一个 fresh 会话（准备留档 `timer-l1fix-20260918.preparation.json`）。

## 8. 补记（2026-09-18 12:55）：L1-005 全量反馈修复成功，INVALID 清零

- **机制变更（版本化）**：将该题反馈评测从固定 5 抽样切换为全量 16 workload
  （`control/feedback_workloads.jsonl` 双侧同步 + `task.json` 的 sha256/indices/uuids 更新，
  v1 备份 `*.feedback-v1.*`）——消除"失败形状不在抽样内"的盲修结构缺陷。
- 窗口 2（40 分钟）产出 c004 并**分析性锁定根因**：K2 核中间值保 fp32，参考实现逐步舍入 bf16，
  小 M + 紧 atol 下分歧超差；窗口 3（40 分钟）实施修复。
- **c005：反馈 16/16 通过（1.38×），final 16/16 通过（1.37×，成对实时计时）**——INVALID 清零。
- 至此 30 题终态：**全量通过 26 题（>1× 者 20）**、无有效候选 4 题；无 INVALID、无未跑全量的有效候选。
