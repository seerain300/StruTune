# KDA A800 (g0056) 30 题终局结果总表

- 批次：`formal-kda-20260916`（30 题，2026-09-16 21:42 → 09-18，含 09-18 续跑/修复轮）
- **feedback_geomean**：搜索内反馈（原始协议：固定 5 抽样 / 100 iterations；L1-005 于 09-18 中途切全量粗测）——排序信号
- **final_geomean**：终局精测（全量 workload / warmup 3–10 / **100 iterations** / 参考实现同进程成对实时计时，无缓存）——**权威口径**
- 硬件：A800-SXM4-80GB (sm_80)；token 为 budget 口径（未缓存输入+缓存写+缓存读+输出），**全题总账**（含归档会话历史）
- 最优候选回退审计：26 题有效候选中 25 干净；`L2-012 c006` 含 torch 路由回退（0.86×，无指标损失）

- **20 题终局精测通过且 >1×**；有效候选 26/30；未解出 4 题均标注终态；合计 token ≈ 18516 万

> **token 列阅读提示**：前 4 题（L2-015/040/049/043，合计 1.15 亿、占总量 62%）的消耗由两个已定因的机制放大，不反映同等工作量：① 中转站 prompt 缓存失效（命中 1–16%），续跑会话每轮全量重发历史（实测放大 10–46 倍）；② L2-015 的 3861 万系评测器不回传编译错误 traceback，导致 14 个候选对同一个 Triton constexpr 错误盲二分（详见 README 工程事项 §5.8）。若需"工作量归一"口径，可参考各题评次数列。

| 组 | 题 | 评次 | 最优候选 | 反馈 | **终局精测** | 终态 | 会话(活/档) | token(万) |
|---|---|---:|---|---:|---:|---|---|---:|
| flashinfer | dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 | 5 | c002 | 25.45× | **25.33×** | operator_stopped | 5/8 | 599 |
| flashinfer | gdn_decode_qk4_v8_d128_k_last | 2 | c001 | 182.09× | **247.87×** | token_limit | 4/2 | 137 |
| flashinfer | gdn_prefill_qk4_v8_d128_k_last | 2 | c002 | 174.75× | **161.69×** | token_limit | 3/9 | 244 |
| flashinfer | gemm_n4096_k4096 | 8 | c007 | 0.58× | 0.59× | search_complete | 3/8 | 658 |
| flashinfer | gqa_paged_decode_h32_kv8_d128_ps1 | 1 | c001 | 633.20× | **539.16×** | token_limit | 3/2 | 130 |
| flashinfer | gqa_paged_prefill_causal_h32_kv8_d128_ps1 | 1 | c001 | 92.36× | **63.06×** | token_limit | 3/2 | 127 |
| flashinfer | gqa_ragged_prefill_causal_h32_kv8_d128 | 1 | c001 | 13.89× | **12.92×** | token_limit | 5/0 | 125 |
| flashinfer | mla_paged_decode_h16_ckv512_kpe64_ps1 | 2 | c002 | 63.63× | **56.26×** | token_limit | 4/0 | 134 |
| flashinfer | mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 | 2 | c002 | 68.46× | **55.44×** | token_limit | 6/0 | 127 |
| flashinfer | rmsnorm_h4096 | 2 | c002 | 4.57× | **4.19×** | token_limit | 6/0 | 116 |
| sol_execbench | L1/002_vae_conv3x3_groupnorm_silu_residual_fused | 6 | c006 | 0.79× | 0.76× | operator_stopped | 3/6 | 519 |
| sol_execbench | L1/005_conv_gated_projection_with_causal_conv | 5 | c003 | 1.47× | **1.36×** | running? | 3/6 | 623 |
| sol_execbench | L1/007_hyena_fft_size_padding_rfft | 1 | c001 | 0.08× | 0.10× | token_limit | 6/0 | 124 |
| sol_execbench | L1/008_expert_output_weighted_index_add_accumulation | 2 | c001 | 1.93× | **1.85×** | token_limit | 5/0 | 139 |
| sol_execbench | L1/018_fused_rope_with_qk_norm_and_kv_cache_update | 3 | c003 | 15.62× | **15.10×** | running? | 1/6 | 234 |
| sol_execbench | L1/020_vision_patch_merger_spatial_shuffle_mlp | 3 | — | — | — | operator_stopped | 3/5 | 637 |
| sol_execbench | L1/053_gaussian_topk_sparse_activation | 2 | c002 | 24.93× | **26.36×** | token_limit | 6/0 | 144 |
| sol_execbench | L1/058_moe_expert_token_radix_sort_with_prefix_sum | 3 | c003 | 3.41× | **3.17×** | token_limit | 6/0 | 169 |
| sol_execbench | L1/070_mamba2_fused_intra_chunk_diagonal_computation | 2 | c001 | 223.26× | **231.09×** | token_limit | 6/0 | 108 |
| sol_execbench | L1/092_gqa_attention_with_qk_norm | 2 | c002 | 2.33× | **2.24×** | token_limit | 4/0 | 149 |
| sol_execbench | L2/012_moe_expert_batched_execution_with_capacity_factor | 7 | c006 | 0.88× | 0.86× | search_complete | 6/4 | 775 |
| sol_execbench | L2/015_audio_sinusoidal_position_embedding_with_conv_projection | 14 | — | — | — | operator_stopped | 12/4 | 3861 ¹ |
| sol_execbench | L2/030_flux_concatenated_sequence_processing_with_split | 2 | c002 | 0.63× | 0.60× | token_limit | 5/0 | 143 |
| sol_execbench | L2/036_convnextv2_layer_with_nhwc_persistence_backward | 0 | — | — | — | operator_stopped | 3/2 | 181 |
| sol_execbench | L2/040_altup_predict_correction_cycle_backward | 15 | c012 | 12.02× | **9.85×** | search_complete | 17/6 | 3386 ¹ |
| sol_execbench | L2/043_mamba_chunk_scan_with_segsum | 10 | c007 | 19.23× | **19.85×** | search_complete | 12/6 | 2029 ¹ |
| sol_execbench | L2/049_group_limited_topk_routing | 13 | c013 | 3.67× | **4.33×** | operator_stopped | 12/3 | 2222 ¹ |
| sol_execbench | L2/051_seqlen-finetuned-reconstructed_hyena_complete_forward_block | 0 | — | — | — | operator_stopped | 2/9 | 63 |
| sol_execbench | L2/057_residual_coupling_flow_block | 4 | c003 | 0.50× | 0.55× | operator_stopped | 5/3 | 443 |
| sol_execbench | L2/080_moe_complete_layer_with_shared_expert_backward | 1 | c001 | 3.88× | **4.43×** | token_limit | 2/3 | 170 |

> ¹ 上标 1 = 消耗受上述机制放大，见顶部阅读提示。
