# K-Search 30 题合并结果报告（formal_h100 + formal_solL2_h100）

- 生成时间：2026-09-23 22:18（v2：torch 回退改为人工审计结论）
- 方法：K-Search world-model 完整版，seed0，每题 100 轮（20 节点 × 5 attempt）
- LLM：AWS-GPT-5.6-Sol @ llmapi.isrc.ac.cn/v1
- 硬件：A800-SXM4-80GB；GPU 池化（flock 并发）+ 亲和模式混合
- 搜索反馈：固定 seed 抽样 5 workload（FlashInfer 3/100/5；SOL 10/100）
- 最终评测：全量 workload；FlashInfer evaluate.py（warmup3/iters100/trials1）；SOL 官方 CLI（iterations100）
- 题集：FlashInfer 10 + SOL L1 10（092 顶补 094）+ SOL L2 10 = **30 题 / 738 workloads**

## 1. 指标与分类定义

| 指标 | 定义 |
|---|---|
| valid / pass / geomean | 全量评测：全部正确=valid；geomean=各 workload speedup 几何平均（ref 与 sol 同进程成对计时）|
| 反馈 best | 搜索期间 5 个抽样 workload 上的最优 mean speedup |
| 内核(定义/launch) | `@triton.jit` 装饰的函数数 / run() 内通过 `_kernel[grid](...)` 实际启动次数——定义了但未 launch 的是死代码 |

**torch 回退人工审计分类**（逐题读 run() 源码，核心判断：「如果把库调用换成最朴素实现，这题还能拿到这个加速比吗？」）：

| 分类 | 含义 | 判定依据 |
|---|---|---|
| **A·核心靠库** | 核心计算调 cuDNN/cuBLAS/cuFFT，自研 Triton 只做周边融合或干脆是死代码 | 库调用在主路径 + 处理题目核心操作 + 加速比通常 ≤1.4x |
| **B·自研为主** | 库调用只做辅助步骤（投影/小 GEMM），计算热点由 Triton 实现 | 库调用不在热点路径 + Triton launch 数 > 0 + 内核覆盖核心操作 |
| **C·待消融** | 无法从代码结构直接判断库和自研的贡献比例 | 需把库调用替换为朴素实现后重测 |
| **干净** | run() 内零库调用 | AST 全扫描无命中 |

## 2. 30 题总表（按题目组）

| # | 任务 | bench | valid | pass | geomean | torch 回退 | 库调用 | 内核(定义/launch) |
|---|---|---|---|---|---|---|---|---|
| 1 | dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 | FlashInfer | ✅ | 23/23 | 157.71 | 干净 | - | 1/1 |
| 2 | gdn_decode_qk4_v8_d128_k_last | FlashInfer | ✅ | 54/54 | 868.65 | 干净 | - | 1/1 |
| 3 | gdn_prefill_qk4_v8_d128_k_last | FlashInfer | ✅ | 100/100 | 230.34 | 干净 | - | 1/1 |
| 4 | gemm_n4096_k4096 | FlashInfer | ✅ | 43/43 | 0.47 | A·核心靠库 | matmul | 1/0 |
| 5 | gqa_paged_decode_h32_kv8_d128_ps1 | FlashInfer | ✅ | 48/48 | 389.40 | 干净 | - | 1/1 |
| 6 | gqa_paged_prefill_causal_h32_kv8_d128_ps1 | FlashInfer | ✅ | 38/38 | 555.89 | 干净 | - | 1/1 |
| 7 | gqa_ragged_prefill_causal_h32_kv8_d128 | FlashInfer | ✅ | 21/21 | 11.12 | 干净 | - | 1/1 |
| 8 | mla_paged_decode_h16_ckv512_kpe64_ps1 | FlashInfer | ✅ | 47/47 | 38.73 | 干净 | - | 2/2 |
| 9 | mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 | FlashInfer | ✅ | 38/38 | 173.47 | 干净 | - | 2/2 |
| 10 | rmsnorm_h4096 | FlashInfer | ✅ | 14/14 | 3.61 | 干净 | - | 1/1 |
| 11 | 005_conv_gated_projection_with_causal_conv | SOL-L1 | ✅ | 16/16 | 1.48 | B·自研为主 | linear×2 | 1/1 |
| 12 | 008_expert_output_weighted_index_add_accumulation | SOL-L1 | ✅ | 16/16 | 5.68 | 干净 | - | 2/2 |
| 13 | 018_fused_rope_with_qk_norm_and_kv_cache_update | SOL-L1 | ✅ | 13/13 | 18.97 | 干净 | - | 1/1 |
| 14 | 020_vision_patch_merger_spatial_shuffle_mlp | SOL-L1 | ✅ | 15/15 | 1.82 | C·混合贡献 | linear×2 | 5/6 |
| 15 | 053_gaussian_topk_sparse_activation | SOL-L1 | ✅ | 12/12 | 35.90 | 干净 | - | 4/4 |
| 16 | 058_moe_expert_token_radix_sort_with_prefix_sum | SOL-L1 | ✅ | 16/16 | 10.32 | 干净 | - | 3/3 |
| 17 | 070_mamba2_fused_intra_chunk_diagonal_computation | SOL-L1 | ✅ | 14/14 | 184.70 | 干净 | - | 2/2 |
| 18 | 030_flux_concatenated_sequence_processing_with_split | SOL-L2 | ✅ | 16/16 | 1.57 | 干净 | - | 1/1 |
| 19 | 040_altup_predict_correction_cycle_backward | SOL-L2 | ✅ | 16/16 | 8.46 | 干净 | - | 5/5 |
| 20 | 043_mamba_chunk_scan_with_segsum | SOL-L2 | ✅ | 16/16 | 3.20 | 干净 | - | 4/4 |
| 21 | 049_group_limited_topk_routing | SOL-L2 | ✅ | 16/16 | 7.01 | 干净 | - | 3/3 |
| 22 | 051_seqlen-finetuned-reconstructed_hyena_complete_forward_block | SOL-L2 | ✅ | 16/16 | 4.15 | C·待消融 | addmm×2,rfft×2,irfft×1 | 7/1 |
| 23 | 080_moe_complete_layer_with_shared_expert_backward | SOL-L2 | ✅ | 16/16 | 5.30 | B·自研为主 | matmul×7,addmm×2 | 2/2 |

## 3. 有库调用的 11 题逐题分析

**gemm_n4096_k4096**（A·核心靠库，0.47x）
- 库调用：matmul | Triton 内核：定义 1 个 / launch 0 次
- run() 主体整体调 torch.matmul(A,B.T)。1 个 Triton 内核定义但 0 次有效 launch（死代码）。r1–59 自研全败、r60 起退化为调与 reference 同款的 cuBLAS。0.84x = 库调用+包装开销。

**002_vae_conv3x3_groupnorm_silu_residual_fused**（A·核心靠库，-x）
- 库调用：conv2d×2 | Triton 内核：定义 6 个 / launch 0 次
- 两个 conv3x3 本体调 F.conv2d（题目核心计算）。6 个 Triton 内核定义（GroupNorm/SiLU/残差）但 run() 内 0 次 launch——全部是死代码。1.39x 来自'调库+GroupNorm 融合'的外围组合。

**005_conv_gated_projection_with_causal_conv**（B·自研为主，1.48x）
- 库调用：linear×2 | Triton 内核：定义 1 个 / launch 1 次
- F.linear 做输入/输出投影。核心 causal_conv+gating 由 Triton _packed_conv_gate_kernel 实现（1 内核 1 launch）。1.57x 来自 gating kernel。

**007_hyena_fft_size_padding_rfft**（A·核心靠库，-x）
- 库调用：rfft×2 | Triton 内核：定义 2 个 / launch 2 次
- RFFT 本体（题目名即 rfft）调 torch.fft.rfft。Triton _direct_rfft_kernel 有 2 次 launch 但仅覆盖部分路径（padding/归一化段），RFFT 核心走 cuFFT。1.38x。

**020_vision_patch_merger_spatial_shuffle_mlp**（C·混合贡献，1.82x）
- 库调用：linear×2 | Triton 内核：定义 5 个 / launch 6 次
- F.linear 做 MLP fc1/fc2 投影。spatial shuffle/patch merge 由 Triton 实现。消融：去掉 F.linear 后 1.46x（-32%），证明库贡献约 1/3。全量反馈版 2.20x（+2%）。

**092_gqa_attention_with_qk_norm**（B·自研为主，-x）
- 库调用：linear×4 | Triton 内核：定义 2 个 / launch 2 次
- F.linear 做 q/k/v/o 投影（前置/后置辅助步骤）。attention 本体由 Triton 实现（2 内核 2 launch）。2.84x 来自 attention 优化，投影走库合理。

**012_moe_expert_batched_execution_with_capacity_factor**（C·待消融，-x）
- 库调用：bmm×3 | Triton 内核：定义 4 个 / launch 4 次
- torch.bmm×3 做专家的 gate/up/down 批量矩阵乘。Triton 4 内核 4 launch 做路由/scatter。1.44x 太温和——如果 bmm 是计算大头，加速来自路由融合而非专家计算本身。需消融确认。

**015_audio_sinusoidal_position_embedding_with_conv_projection**（A·核心靠库，-x）
- 库调用：conv2d×3,addmm,linear | Triton 内核：定义 3 个 / launch 3 次
- 三个 conv2d 本体调库 + addmm/linear 投影调库。Triton 只做 gelu/transpose/scale（3 内核 3 launch）。1.03x ≈ 零加速。reference 也调同样的库，解等于没优化。15/16 有一个数值失败。

**036_convnextv2_layer_with_nhwc_persistence_backward**（B·自研为主，-x）
- 库调用：mm×4 | Triton 内核：定义 9 个 / launch 10 次
- torch.mm 做 pwconv 权重梯度的小 GEMM（合理分步）。GRN 反传/LayerNorm 反传/dwconv 梯度/NHWC 布局全部由 9 个 Triton 内核（10 次 launch）实现。364x 的主体来自消掉 reference 的 permute/clone/for 循环。

**051_seqlen-finetuned-reconstructed_hyena_complete_forward_block**（C·待消融，4.15x）
- 库调用：addmm×2,rfft×2,irfft×1 | Triton 内核：定义 7 个 / launch 1 次
- 7 个 Triton 内核定义但只有 1 次 launch——6 个是死代码。FFT 走库，addmm 投影走库。2.35x 的来源需实验验证（可能来自 1 个 Triton kernel + reference 本身的低效）。

**080_moe_complete_layer_with_shared_expert_backward**（B·自研为主，5.30x）
- 库调用：matmul×7,addmm×2 | Triton 内核：定义 2 个 / launch 2 次
- matmul/addmm 做 shared expert 梯度和 router 梯度（代码行数多但非计算热点）。routed expert 的 scatter/gather/融合由 Triton 实现（2 内核 2 launch）。6x 来自 routed 部分。

## 4. 失败题注记

| 题 | 终数 | 原因 |
|---|---|---|
| 058 | 15/16 | batch=1/seq=2080 INCORRECT_NUMERICAL（排序 kernel 边界形状 bug）|
| 057 | 9/16 | 反馈 5 全过但全量 7 shape 失败——泛化缺口 |
| 015 | 15/16 | 1 workload 数值失败（伴随核心靠库）|
| gemm | 43/43 正确但 0.84x | A 类：run() 主体调 cuBLAS |
| 094（v3 剔除） | — | reference 朴素 Python 循环，评测 4h+/题 |

## 5. 搜索进度与反馈 best

| 任务 | 轮次 | 反馈 best |
|---|---|---|
| dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 | 246 | 27.429 |
| gdn_decode_qk4_v8_d128_k_last | 262 | 688.179 |
| gdn_prefill_qk4_v8_d128_k_last | 263 | 344.855 |
| gemm_n4096_k4096 | 267 | 0.910 |
| gqa_paged_decode_h32_kv8_d128_ps1 | 275 | 722.986 |
| gqa_paged_prefill_causal_h32_kv8_d128_ps1 | 339 | 1684.142 |
| gqa_ragged_prefill_causal_h32_kv8_d128 | 259 | 15.818 |
| mla_paged_decode_h16_ckv512_kpe64_ps1 | 348 | 55.151 |
| mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 | 352 | 288.295 |
| rmsnorm_h4096 | 247 | 4.625 |
| 002_vae_conv3x3_groupnorm_silu_residual_fused | 113 | 1.203 |
| 005_conv_gated_projection_with_causal_conv | 104 | 1.502 |
| 007_hyena_fft_size_padding_rfft | 105 | 3.348 |
| 008_expert_output_weighted_index_add_accumulation | 106 | 5.286 |
| 018_fused_rope_with_qk_norm_and_kv_cache_update | 118 | 24.989 |
| 020_vision_patch_merger_spatial_shuffle_mlp | 110 | 2.123 |
| 053_gaussian_topk_sparse_activation | 118 | 67.637 |
| 058_moe_expert_token_radix_sort_with_prefix_sum | 107 | 16.064 |
| 070_mamba2_fused_intra_chunk_diagonal_computation | 116 | 361.184 |
| 092_gqa_attention_with_qk_norm | 113 | 3.902 |
| 012_moe_expert_batched_execution_with_capacity_factor | 126 | 1.758 |
| 015_audio_sinusoidal_position_embedding_with_conv_projection | 105 | 1.043 |
| 030_flux_concatenated_sequence_processing_with_split | 161 | 1.623 |
| 036_convnextv2_layer_with_nhwc_persistence_backward | 105 | 57.264 |
| 040_altup_predict_correction_cycle_backward | 126 | 9.917 |
| 043_mamba_chunk_scan_with_segsum | 130 | 6.121 |
| 049_group_limited_topk_routing | 119 | 7.630 |
| 051_seqlen-finetuned-reconstructed_hyena_complete_forward_block | 112 | 5.547 |
| 057_residual_coupling_flow_block | 108 | 1.330 |
| 080_moe_complete_layer_with_shared_expert_backward | 105 | 6.089 |

## 6. Token 消耗

| 任务 | LLM calls | input | cached | output | reasoning | 总计 |
|---|---|---|---|---|---|---|
| dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 | 278 | 2,666,182 | 1,578,759 | 1,065,452 | 398,484 | 3,731,634 |
| gdn_decode_qk4_v8_d128_k_last | 445 | 4,016,363 | 2,121,216 | 1,101,521 | 256,859 | 5,117,884 |
| gdn_prefill_qk4_v8_d128_k_last | 513 | 4,315,200 | 2,237,652 | 1,716,909 | 456,144 | 6,032,109 |
| gemm_n4096_k4096 | 309 | 2,399,547 | 1,554,112 | 680,064 | 215,856 | 3,079,611 |
| gqa_paged_decode_h32_kv8_d128_ps1 | 361 | 3,498,331 | 1,843,584 | 1,218,711 | 206,937 | 4,717,042 |
| gqa_paged_prefill_causal_h32_kv8_d128_ps1 | 393 | 3,544,637 | 1,530,368 | 1,613,138 | 343,603 | 5,157,775 |
| gqa_ragged_prefill_causal_h32_kv8_d128 | 354 | 4,004,171 | 1,875,584 | 1,453,673 | 318,410 | 5,457,844 |
| mla_paged_decode_h16_ckv512_kpe64_ps1 | 415 | 3,883,467 | 1,587,456 | 1,947,412 | 386,258 | 5,830,879 |
| mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 | 423 | 4,303,921 | 1,621,189 | 1,954,449 | 415,792 | 6,258,370 |
| rmsnorm_h4096 | 272 | 2,278,735 | 1,457,408 | 338,033 | 106,868 | 2,616,768 |
| 002_vae_conv3x3_groupnorm_silu_residual_fused | 153 | 2,152,317 | 789,888 | 549,540 | 174,626 | 2,701,857 |
| 005_conv_gated_projection_with_causal_conv | 132 | 1,756,416 | 584,448 | 367,460 | 108,908 | 2,123,876 |
| 007_hyena_fft_size_padding_rfft | 140 | 1,561,062 | 646,272 | 318,545 | 101,758 | 1,879,607 |
| 008_expert_output_weighted_index_add_accumulation | 157 | 1,616,435 | 731,008 | 338,452 | 141,807 | 1,954,887 |
| 018_fused_rope_with_qk_norm_and_kv_cache_update | 158 | 1,779,789 | 529,392 | 701,537 | 109,237 | 2,481,326 |
| 020_vision_patch_merger_spatial_shuffle_mlp | 153 | 2,021,846 | 419,456 | 687,169 | 79,047 | 2,709,015 |
| 053_gaussian_topk_sparse_activation | 150 | 1,718,038 | 424,704 | 698,828 | 96,407 | 2,416,866 |
| 058_moe_expert_token_radix_sort_with_prefix_sum | 135 | 1,059,615 | 286,976 | 474,107 | 58,685 | 1,533,722 |
| 070_mamba2_fused_intra_chunk_diagonal_computation | 178 | 2,096,949 | 692,591 | 416,214 | 102,044 | 2,513,163 |
| 092_gqa_attention_with_qk_norm | 157 | 1,434,971 | 304,644 | 682,603 | 61,728 | 2,117,574 |
| 012_moe_expert_batched_execution_with_capacity_factor | 156 | 1,942,590 | 780,365 | 506,726 | 166,084 | 2,449,316 |
| 015_audio_sinusoidal_position_embedding_with_conv_projection | 146 | 1,731,084 | 665,600 | 331,234 | 136,740 | 2,062,318 |
| 030_flux_concatenated_sequence_processing_with_split | 205 | 2,024,180 | 933,376 | 445,686 | 205,094 | 2,469,866 |
| 036_convnextv2_layer_with_nhwc_persistence_backward | 149 | 2,373,137 | 707,840 | 541,475 | 147,097 | 2,914,612 |
| 040_altup_predict_correction_cycle_backward | 172 | 3,416,006 | 989,620 | 1,207,101 | 176,747 | 4,623,107 |
| 043_mamba_chunk_scan_with_segsum | 163 | 2,825,795 | 1,091,905 | 783,174 | 146,096 | 3,608,969 |
| 049_group_limited_topk_routing | 152 | 2,244,903 | 826,112 | 662,971 | 162,430 | 2,907,874 |
| 051_seqlen-finetuned-reconstructed_hyena_complete_forward_block | 157 | 2,227,332 | 794,880 | 504,915 | 163,404 | 2,732,247 |
| 057_residual_coupling_flow_block | 149 | 1,721,671 | 670,592 | 385,251 | 148,941 | 2,106,922 |
| 080_moe_complete_layer_with_shared_expert_backward | 153 | 2,087,439 | 769,920 | 413,007 | 124,700 | 2,500,446 |
| **合计** | **6878** | **74,702,129** | **31,046,917** | **24,105,357** | **5,716,791** | **98,807,486** |

## 7. 观察

- **A 类 4 题终数全部 ≤1.39x**：gemm 0.84 / 002 1.39 / 015 1.03 / 007 1.38——加速上限被库调用封死
- **B 类 4 题加速比 1.57~364x**：库调用只做辅助步骤，加速主体来自 Triton 自研（消融验证：005 去掉 F.linear 后仅 -10%）
- **C 类 2+1=3 题**加速比 1.44~2.35x——020 经消融证明库贡献 32%（1.46x vs 2.16x），从 B 改判 C
- **C 类 2 题**（012/051）加速比 1.44x / 2.35x——需消融实验（把库调用替换为朴素实现后重测）才能归因
- **干净 19 题**加速比 1.38~393x——零库调用，加速全部来自自研
- **A 类的共同特征**：题目核心操作即库函数擅长域（GEMM/conv/FFT），LLM 选择调库而非自研——与 gemm 的 r59 全败后 r60 转向调库行为一致
- **守卫效果**：上游防回退守卫（v0.4 起注入 SOL）阻止了 run() 主体整体调库（gemm 式躺平），但无法阻止'热点自研+辅助调库'的混合策略（B/C 类）
- **内核数 ≠ 自研贡献**：002 定义了 6 个内核但 0 次 launch（全死代码）；051 定义了 7 个但只 launch 1 个——必须数 launch 次数
