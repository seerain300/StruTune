# dr-kernel-a800 结果总表

- 模型：hkust-nlp/drkernel-8b（vLLM 推理，权重与官方 main @990fe42 一致）
- 硬件：NVIDIA A800-SXM4-80GB（生成 2×vLLM；评测走 GPU 池）
- **final_geomean_speedup 口径：官方评测器全新复评（全 workload、参考重计时、warmup 3 + 100 iters），A800 硬件绑定**
- best_feedback_geomean：轮级反馈口径（参考延迟缓存 + warmup 3 + 10 iters），仅用于轨迹内选择，不可与 final 混用
- pass_at_1：该批次 8 条采样中"任一轮全部 workload 正确"的采样占比

## 批次一 stts（STTS：8 采样 × 最多 10 迭代 × 每迭代 5 轮，2026-09-19~20）

| 任务 | pass@1 | best反馈geomean | **终局speedup(A800)** | 解出 |
|---|---|---|---|---|
| flashinfer/rmsnorm_h4096 | 88% | 3.71 | 3.30 | ✅ |
| SOL/L1/053_gaussian_topk_sparse_activation | 62% | 10.98 | 3.22 | ✅ |
| SOL/L1/008_expert_output_weighted_index_add_accumulation | 88% | 2.16 | 2.19 | ✅ |
| SOL/L1/058_moe_expert_token_radix_sort_with_prefix_sum | 12% | 0.66 | 0.65 | · |
| flashinfer/gemm_n4096_k4096 | 88% | 0.60 | 0.55 | · |
| flashinfer/mla_paged_decode_h16_ckv512_kpe64_ps1 | 12% | 0.21 | 0.21 | · |
| SOL/L2/030_flux_concatenated_sequence_processing_with_split | 25% | 0.15 | 0.14 | · |
| SOL/L1/002_vae_conv3x3_groupnorm_silu_residual_fused | 0% | — | — | ✗ |
| SOL/L1/005_conv_gated_projection_with_causal_conv | 0% | — | — | ✗ |
| SOL/L1/007_hyena_fft_size_padding_rfft | 0% | — | — | ✗ |
| SOL/L1/018_fused_rope_with_qk_norm_and_kv_cache_update | 0% | — | — | ✗ |
| SOL/L1/020_vision_patch_merger_spatial_shuffle_mlp | 0% | — | — | ✗ |
| SOL/L1/070_mamba2_fused_intra_chunk_diagonal_computation | 0% | — | — | ✗ |
| SOL/L1/092_gqa_attention_with_qk_norm | 0% | — | — | ✗ |
| SOL/L2/012_moe_expert_batched_execution_with_capacity_factor | 0% | — | — | ✗ |
| SOL/L2/015_audio_sinusoidal_position_embedding_with_conv_projection | 0% | — | — | ✗ |
| SOL/L2/036_convnextv2_layer_with_nhwc_persistence_backward | 0% | — | — | ✗ |
| SOL/L2/040_altup_predict_correction_cycle_backward | 0% | — | — | ✗ |
| SOL/L2/043_mamba_chunk_scan_with_segsum | 0% | — | — | ✗ |
| SOL/L2/049_group_limited_topk_routing | 0% | — | — | ✗ |
| SOL/L2/051_seqlen-finetuned-reconstructed_hyena_complete_forward_block | 0% | — | — | ✗ |
| SOL/L2/057_residual_coupling_flow_block | 0% | — | — | ✗ |
| SOL/L2/080_moe_complete_layer_with_shared_expert_backward | 0% | — | — | ✗ |
| flashinfer/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 | 0% | — | — | ✗ |
| flashinfer/gdn_decode_qk4_v8_d128_k_last | 0% | — | — | ✗ |
| flashinfer/gdn_prefill_qk4_v8_d128_k_last | 0% | — | — | ✗ |
| flashinfer/gqa_paged_decode_h32_kv8_d128_ps1 | 0% | — | — | ✗ |
| flashinfer/gqa_paged_prefill_causal_h32_kv8_d128_ps1 | 0% | — | — | ✗ |
| flashinfer/gqa_ragged_prefill_causal_h32_kv8_d128 | 0% | — | — | ✗ |
| flashinfer/mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 | 0% | — | — | ✗ |

## 批次二 stts3turn（基线：8 采样 × 3 轮反馈，2026-09-18）

| 任务 | pass@1 | 终局speedup(A800) | 解出 |
|---|---|---|---|
| flashinfer/rmsnorm_h4096 | 50% | 3.20 | ✅ |
| SOL/L1/008_expert_output_weighted_index_add_accumulation | 38% | 1.71 | ✅ |
| flashinfer/gemm_n4096_k4096 | 75% | 0.43 | · |
| SOL/L1/053_gaussian_topk_sparse_activation | 12% | 0.10 | · |
| SOL/L1/058_moe_expert_token_radix_sort_with_prefix_sum | 12% | 0.01 | · |
| SOL/L1/002_vae_conv3x3_groupnorm_silu_residual_fused | 0% | — | ✗ |
| SOL/L1/005_conv_gated_projection_with_causal_conv | 0% | — | ✗ |
| SOL/L1/007_hyena_fft_size_padding_rfft | 0% | — | ✗ |
| SOL/L1/018_fused_rope_with_qk_norm_and_kv_cache_update | 0% | — | ✗ |
| SOL/L1/020_vision_patch_merger_spatial_shuffle_mlp | 0% | — | ✗ |
| SOL/L1/070_mamba2_fused_intra_chunk_diagonal_computation | 0% | — | ✗ |
| SOL/L1/092_gqa_attention_with_qk_norm | 0% | — | ✗ |
| SOL/L2/012_moe_expert_batched_execution_with_capacity_factor | 0% | — | ✗ |
| SOL/L2/015_audio_sinusoidal_position_embedding_with_conv_projection | 0% | — | ✗ |
| SOL/L2/030_flux_concatenated_sequence_processing_with_split | 0% | — | ✗ |
| SOL/L2/036_convnextv2_layer_with_nhwc_persistence_backward | 0% | — | ✗ |
| SOL/L2/040_altup_predict_correction_cycle_backward | 0% | — | ✗ |
| SOL/L2/043_mamba_chunk_scan_with_segsum | 0% | — | ✗ |
| SOL/L2/049_group_limited_topk_routing | 0% | — | ✗ |
| SOL/L2/051_seqlen-finetuned-reconstructed_hyena_complete_forward_block | 0% | — | ✗ |
| SOL/L2/057_residual_coupling_flow_block | 0% | — | ✗ |
| SOL/L2/080_moe_complete_layer_with_shared_expert_backward | 0% | — | ✗ |
| flashinfer/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 | 0% | — | ✗ |
| flashinfer/gdn_decode_qk4_v8_d128_k_last | 0% | — | ✗ |
| flashinfer/gdn_prefill_qk4_v8_d128_k_last | 0% | — | ✗ |
| flashinfer/gqa_paged_decode_h32_kv8_d128_ps1 | 0% | — | ✗ |
| flashinfer/gqa_paged_prefill_causal_h32_kv8_d128_ps1 | 0% | — | ✗ |
| flashinfer/gqa_ragged_prefill_causal_h32_kv8_d128 | 0% | — | ✗ |
| flashinfer/mla_paged_decode_h16_ckv512_kpe64_ps1 | 0% | — | ✗ |
| flashinfer/mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 | 0% | — | ✗ |
