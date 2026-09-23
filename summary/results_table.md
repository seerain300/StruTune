# KDA H100 首轮实验结果总表

- 批次：`formal-kda-h100-20260920`（30 题，每题 1 小时含 draft/plan，反馈全量粗测 w2/i10）
- **feedback_geomean**：搜索内粗测（warmup 2 / 10 iterations，排序信号）
- **final_geomean**：终局精测（全量 workload / warmup 3–10 / **100 iterations** / 参考实现同进程成对实时计时，无缓存）——**权威口径**
- 硬件：H100 80GB (sm_90)；token 为 budget 口径（未缓存输入+缓存写+缓存读+输出）累计
- 17/30 精测有效；最优候选回退审计 17/17 干净（零 torch 回退）

| 组 | 题 | 评次 | 最优候选 | 反馈粗测 | **终局精测** | 终态 | token(万) |
|---|---|---:|---|---:|---:|---|---:|
| flashinfer | dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 | 5 | c005 | 22.92× | **24.31×** | search_complete | 539 |
| flashinfer | gdn_decode_qk4_v8_d128_k_last | 4 | c004 | 288.48× | **293.35×** | time_budget | 532 |
| flashinfer | gdn_prefill_qk4_v8_d128_k_last | 2 | c002 | 130.25× | **131.90×** | time_budget | 437 |
| flashinfer | gemm_n4096_k4096 | 10 | c004 | 0.58× | **0.54×** | time_budget | 1246 |
| flashinfer | gqa_paged_decode_h32_kv8_d128_ps1 | 6 | c004 | 521.27× | **520.22×** | time_budget | 544 |
| flashinfer | gqa_paged_prefill_causal_h32_kv8_d128_ps1 | 0 | — | —× | — | time_budget | 60 |
| flashinfer | gqa_ragged_prefill_causal_h32_kv8_d128 | 6 | c004 | 11.93× | **12.45×** | search_complete | 775 |
| flashinfer | mla_paged_decode_h16_ckv512_kpe64_ps1 | 2 | c002 | 51.46× | **47.19×** | time_budget | 417 |
| flashinfer | mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 | 3 | c003 | 112.57× | **118.64×** | time_budget | 568 |
| flashinfer | rmsnorm_h4096 | 3 | c002 | 4.24× | **4.37×** | token_limit | 153 |
| sol_execbench | L1-002_vae_conv3x3_groupnorm_silu_residual_fused | 2 | — | —× | — | time_budget | 204 |
| sol_execbench | L1-005_conv_gated_projection_with_causal_conv | 5 | c003 | 1.50× | **1.55×** | time_budget | 406 |
| sol_execbench | L1-007_hyena_fft_size_padding_rfft | 1 | c001 | 0.07× | **0.06×** | time_budget | 35 |
| sol_execbench | L1-008_expert_output_weighted_index_add_accumulation | 0 | — | —× | — | exhausted_windows | 393 |
| sol_execbench | L1-018_fused_rope_with_qk_norm_and_kv_cache_update | 3 | c003 | 14.64× | **15.06×** | time_budget | 173 |
| sol_execbench | L1-020_vision_patch_merger_spatial_shuffle_mlp | 0 | — | —× | — | time_budget | 94 |
| sol_execbench | L1-053_gaussian_topk_sparse_activation | 6 | c006 | 31.45× | **32.92×** | token_limit | 627 |
| sol_execbench | L1-058_moe_expert_token_radix_sort_with_prefix_sum | 3 | c003 | 3.47× | **3.66×** | time_budget | 392 |
| sol_execbench | L1-070_mamba2_fused_intra_chunk_diagonal_computation | 5 | c001 | 362.24× | **400.79×** | search_complete | 367 |
| sol_execbench | L1-092_gqa_attention_with_qk_norm | 4 | c003 | 2.67× | **2.72×** | time_budget | 497 |
| sol_execbench | L2-012_moe_expert_batched_execution_with_capacity_factor | 0 | — | —× | — | time_budget | 137 |
| sol_execbench | L2-015_audio_sinusoidal_position_embedding_with_conv_projection | 2 | — | —× | — | time_budget | 101 |
| sol_execbench | L2-030_flux_concatenated_sequence_processing_with_split | 5 | c004 | 1.63× | **1.71×** | search_complete | 479 |
| sol_execbench | L2-036_convnextv2_layer_with_nhwc_persistence_backward | 0 | — | —× | — | time_budget | 65 |
| sol_execbench | L2-040_altup_predict_correction_cycle_backward | 3 | — | —× | — | time_budget | 215 |
| sol_execbench | L2-043_mamba_chunk_scan_with_segsum | 2 | — | —× | — | time_budget | 407 |
| sol_execbench | L2-049_group_limited_topk_routing | 4 | — | —× | — | time_budget | 602 |
| sol_execbench | L2-051_seqlen-finetuned-reconstructed_hyena_complete_forward_block | 0 | — | —× | — | time_budget | 50 |
| sol_execbench | L2-057_residual_coupling_flow_block | 1 | — | —× | — | time_budget | 71 |
| sol_execbench | L2-080_moe_complete_layer_with_shared_expert_backward | 2 | c001 | 3.26× | — | time_budget | 307 |
