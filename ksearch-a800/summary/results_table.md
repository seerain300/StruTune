| # | 任务 | bench | valid | pass | geomean speedup | arith | torch 回退 | 反馈 best | tokens | 解来源批次 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 | FlashInfer | ✅ | 23/23 | 20.720 | 21.453 | 干净(-) | 20.999 | 1,624,052 | formal_20260914 |
| 2 | gdn_decode_qk4_v8_d128_k_last | FlashInfer | ✅ | 54/54 | 225.251 | 437.052 | 干净(-) | 338.365 | 1,389,861 | formal_20260914 |
| 3 | gdn_prefill_qk4_v8_d128_k_last | FlashInfer | ✅ | 100/100 | 195.329 | 231.581 | 干净(-) | 227.748 | 1,869,496 | formal_20260914 |
| 4 | gemm_n4096_k4096 | FlashInfer | ✅ | 43/43 | 0.839 | 0.842 | A·核心靠库(matmul) | 0.840 | 853,791 | formal_20260914 |
| 5 | gqa_paged_decode_h32_kv8_d128_ps1 | FlashInfer | ✅ | 48/48 | 124.867 | 154.771 | 干净(-) | 178.854 | 3,182,294 | formal_20260914 |
| 6 | gqa_paged_prefill_causal_h32_kv8_d128_ps1 | FlashInfer | ✅ | 38/38 | 392.862 | 717.077 | 干净(-) | 873.784 | 1,792,216 | formal_20260914 |
| 7 | gqa_ragged_prefill_causal_h32_kv8_d128 | FlashInfer | ✅ | 21/21 | 11.066 | 13.437 | 干净(-) | 21.539 | 1,573,412 | formal_20260914 |
| 8 | mla_paged_decode_h16_ckv512_kpe64_ps1 | FlashInfer | ✅ | 47/47 | 28.400 | 32.599 | 干净(-) | 36.789 | 1,935,702 | formal_20260914 |
| 9 | mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 | FlashInfer | ✅ | 38/38 | 173.171 | 296.009 | 干净(-) | 342.446 | 1,820,164 | formal_20260914 |
| 10 | rmsnorm_h4096 | FlashInfer | ✅ | 14/14 | 3.471 | 4.417 | 干净(-) | 4.853 | 952,384 | formal_20260914 |
| 11 | 002_vae_conv3x3_groupnorm_silu_residual_fused | SOL-L1 | ✅ | 20/20 | 1.389 | 1.402 | A·核心靠库(conv2d×2) | 1.437 | 2,173,042 | formal_20260914 |
| 12 | 005_conv_gated_projection_with_causal_conv | SOL-L1 | ✅ | 16/16 | 1.420 | 1.423 | 干净(消融替代†) | 1.559 | 1,311,758 | formal_20260914 + ablation_strict_nolib |
| 13 | 007_hyena_fft_size_padding_rfft | SOL-L1 | ✅ | 16/16 | 1.375 | 1.386 | A·核心靠库(rfft×2) | 1.309 | 1,207,483 | formal_20260914 |
| 14 | 008_expert_output_weighted_index_add_accumulation | SOL-L1 | ✅ | 16/16 | 2.695 | 2.876 | 干净(-) | 3.259 | 926,741 | formal_20260914 |
| 15 | 018_fused_rope_with_qk_norm_and_kv_cache_update | SOL-L1 | ✅ | 13/13 | 21.241 | 22.435 | 干净(-) | 23.174 | 1,604,465 | formal_20260914 |
| 16 | 020_vision_patch_merger_spatial_shuffle_mlp | SOL-L1 | ✅ | 15/15 | 1.464 | 1.527 | 干净(消融替代‡) | 2.923 | 2,255,698 | formal_20260914 + ablation_strict_nolib |
| 17 | 053_gaussian_topk_sparse_activation | SOL-L1 | ✅ | 12/12 | 31.807 | 49.519 | 干净(-) | 33.329 | 1,506,558 | formal_20260914 |
| 18 | 058_moe_expert_token_radix_sort_with_prefix_sum | SOL-L1 | ✅ | 16/16 | 10.324 | 10.422 | 干净(-) | 11.307 | 1,205,494 | formal_20260914 + ablation_resume5 |
| 19 | 070_mamba2_fused_intra_chunk_diagonal_computation | SOL-L1 | ✅ | 14/14 | 192.994 | 225.331 | 干净(-) | 375.506 | 1,779,550 | formal_20260914 |
| 20 | 092_gqa_attention_with_qk_norm | SOL-L1 | ✅ | 16/16 | 2.843 | 2.898 | B·自研为主(linear×4) | 3.151 | 1,550,189 | formal_20260914 |
| 21 | 012_moe_expert_batched_execution_with_capacity_factor | SOL-L2 | ✅ | 16/16 | 1.440 | 1.442 | C·待消融(bmm×3) | 1.454 | 1,767,233 | formal2_solL2_20260916 |
| 22 | 015_audio_sinusoidal_position_embedding_with_conv_projection | SOL-L2 | ❌ | 15/16 | 1.026 | 1.026 | A·核心靠库(conv2d×3,addmm,linear) | 1.207 | 1,429,040 | formal2_solL2_20260916 |
| 23 | 030_flux_concatenated_sequence_processing_with_split | SOL-L2 | ✅ | 16/16 | 1.379 | 1.384 | 干净(-) | 1.423 | 1,239,761 | formal2_solL2_20260916 |
| 24 | 036_convnextv2_layer_with_nhwc_persistence_backward | SOL-L2 | ✅ | 14/14 | 364.242 | 479.277 | B·自研为主(mm×4) | 614.033 | 3,422,715 | formal2_solL2_20260916 |
| 25 | 040_altup_predict_correction_cycle_backward | SOL-L2 | ✅ | 16/16 | 17.403 | 18.310 | 干净(-) | 23.223 | 3,350,041 | formal2_solL2_20260916 |
| 26 | 043_mamba_chunk_scan_with_segsum | SOL-L2 | ✅ | 16/16 | 9.169 | 10.796 | 干净(-) | 8.523 | 2,425,533 | formal2_solL2_20260916 |
| 27 | 049_group_limited_topk_routing | SOL-L2 | ✅ | 16/16 | 8.940 | 8.972 | 干净(-) | 10.673 | 1,773,814 | formal2_solL2_20260916 |
| 28 | 051_seqlen-finetuned-reconstructed_hyena_complete_forward_block | SOL-L2 | ✅ | 16/16 | 2.351 | 2.489 | C·待消融(addmm×2,rfft/irfft) | 2.282 | 3,024,273 | formal2_solL2_20260916 |
| 29 | 057_residual_coupling_flow_block | SOL-L2 | ❌ | 9/16 | 1.044 | 1.044 | 干净(-) | 1.050 | 1,804,318 | formal2_solL2_20260916 |
| 30 | 080_moe_complete_layer_with_shared_expert_backward | SOL-L2 | ✅ | 16/16 | 6.027 | 6.239 | B·自研为主(matmul×7,addmm×2) | 4.831 | 1,368,648 | formal2_solL2_20260916 |
| - | 094_time_decay_exponential_stabilization（v3 剔除） | SOL-L1 | — | — | — | — | 干净 | 18347* | 见 batch_logs | excluded_v3（ref 为朴素 Python 循环，评测 4h+/题） |
† 005 主行用强化守卫消融解（ablation_strict_nolib_20260917）替代展示：原 formal 解调 F.linear×2
（B·自研为主，1.565x）；消融重跑 43 轮后全 Triton 解为 1.420x（-10%）。终评 JSON 两版均在
`tasks/005_.../` 两个批次目录。反馈 best 1.559 为 formal 批口径（消融批 43 轮内 best 1.362，
因守卫中止未满 100 轮，tokens 列为 formal 批消耗）。
‡ 020 主行同样用强化守卫消融解替代展示：原 formal 解调 F.linear×2（C·混合贡献，2.160x，
消融证明库贡献 32%）；全 Triton 消融解为 1.464x（51/100 轮中止，终评 15/15 valid）。
终评 JSON 两版均在 `tasks/020_.../` 两个批次目录。反馈 best 2.923 为 formal 批口径
（消融批 51 轮内 best 2.229，tokens 列为 formal 批消耗 2,255,698；消融批 948,286）。
