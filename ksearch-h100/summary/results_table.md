# K-Search H100 结果总表

- 口径：终评 = 全量 workload，warmup3 + 100 iterations，参考与候选同进程成对计时（无缓存）；
  唯一例外 gdn_prefill（† 非标口径，见注记列）
- 违规留档 = 退出解含 BANNED torch 调用（cuDNN/cuBLAS/cuFFT 委托），按协议不终评；
  token 列为该题有效搜索消耗（含违规题——搜索是真实发生的）
- Token 口径：该题全部完成评测轮窗口内的 LLM 调用记账（重试损耗已扣）

| 组 | 任务 | 状态 | pass | 终评 geomean | 轮次 | 输入(M) | 输出(M) | 注记 |
|---|---|---|---|---|---|---|---|---|
| FlashInfer | dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 | 已终评 | 23/23 | 157.71x | 100 | 2.67 | 1.07 |  |
| FlashInfer | gdn_decode_qk4_v8_d128_k_last | 已终评 | 54/54 | 868.65x | 100 | 4.02 | 1.10 |  |
| FlashInfer | gdn_prefill_qk4_v8_d128_k_last | 已终评 | 100/100 | 230.34x | 102 | 4.32 | 1.72 | 非标口径†：warmup3+20iters+参考延迟走磁盘缓存（全量参考单遍 521min 不经济；候选侧实时计时，100/100 全过） |
| FlashInfer | gemm_n4096_k4096 | 已终评 | 43/43 | 0.47x | 104 | 2.40 | 0.68 | 0.47x 为真实结果：r1-59 自研全败，退出解=调 cuBLAS（与 reference 同款），无自研可用 |
| FlashInfer | gqa_paged_decode_h32_kv8_d128_ps1 | 已终评 | 48/48 | 389.40x | 100 | 3.50 | 1.22 |  |
| FlashInfer | gqa_paged_prefill_causal_h32_kv8_d128_ps1 | 已终评 | 38/38 | 555.89x | 106 | 3.54 | 1.61 |  |
| FlashInfer | gqa_ragged_prefill_causal_h32_kv8_d128 | 已终评 | 21/21 | 11.12x | 100 | 4.00 | 1.45 |  |
| FlashInfer | mla_paged_decode_h16_ckv512_kpe64_ps1 | 已终评 | 47/47 | 38.73x | 100 | 3.88 | 1.95 |  |
| FlashInfer | mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 | 已终评 | 38/38 | 173.47x | 105 | 4.30 | 1.95 |  |
| FlashInfer | rmsnorm_h4096 | 已终评 | 14/14 | 3.61x | 100 | 2.28 | 0.34 |  |
| SOL-L1 | 002_vae_conv3x3_groupnorm_silu_residual_fused | 违规留档 | — | — | 104 | 2.15 | 0.55 | F.conv2d×2（conv 本体调库，6 个 Triton 内核全部死代码） |
| SOL-L1 | 005_conv_gated_projection_with_causal_conv | 已终评 | 16/16 | 1.48x | 100 | 1.76 | 0.37 |  |
| SOL-L1 | 007_hyena_fft_size_padding_rfft | 违规留档 | — | — | 100 | 1.56 | 0.32 | torch.fft.rfft×2（FFT 本体调 cuFFT） |
| SOL-L1 | 008_expert_output_weighted_index_add_accumulation | 已终评 | 16/16 | 5.68x | 100 | 1.62 | 0.34 |  |
| SOL-L1 | 018_fused_rope_with_qk_norm_and_kv_cache_update | 已终评 | 13/13 | 18.97x | 100 | 1.78 | 0.70 |  |
| SOL-L1 | 020_vision_patch_merger_spatial_shuffle_mlp | 已终评 | 15/15 | 1.82x | 103 | 2.02 | 0.69 |  |
| SOL-L1 | 053_gaussian_topk_sparse_activation | 已终评 | 12/12 | 35.90x | 100 | 1.72 | 0.70 |  |
| SOL-L1 | 058_moe_expert_token_radix_sort_with_prefix_sum | 已终评 | 16/16 | 13.00x | 100 | 1.06 | 0.47 |  |
| SOL-L1 | 070_mamba2_fused_intra_chunk_diagonal_computation | 已终评 | 14/14 | 184.70x | 101 | 2.10 | 0.42 |  |
| SOL-L1 | 092_gqa_attention_with_qk_norm | 违规留档 | — | — | 100 | 1.43 | 0.68 | F.linear×4（q/k/v/o 投影调库；attention 本体 Triton） |
| SOL-L2 | 012_moe_expert_batched_execution_with_capacity_factor | 违规留档 | — | — | 100 | 1.94 | 0.51 | torch.bmm×3 |
| SOL-L2 | 015_audio_sinusoidal_position_embedding_with_conv_projection | 违规留档 | — | — | 102 | 1.73 | 0.33 | F.conv2d×5 |
| SOL-L2 | 030_flux_concatenated_sequence_processing_with_split | 已终评 | 16/16 | 1.57x | 141 | 2.02 | 0.45 | 预算超支 141 轮；best@31=1.62x 轮次代码丢失，退出保存解 1.57x |
| SOL-L2 | 036_convnextv2_layer_with_nhwc_persistence_backward | 违规留档 | — | — | 101 | 2.37 | 0.54 | @torch.compile + torch.mm×2 |
| SOL-L2 | 040_altup_predict_correction_cycle_backward | 已终评 | 16/16 | 8.46x | 100 | 3.42 | 1.21 |  |
| SOL-L2 | 043_mamba_chunk_scan_with_segsum | 已终评 | 16/16 | 3.20x | 104 | 2.83 | 0.78 | 退出解违规(torch.compile+cumsum+bmm)；表中数字为搜索期 best-round 干净解(3.20x)复评；该 best-round 代码丢失注记见 docs/RUNBOOK |
| SOL-L2 | 049_group_limited_topk_routing | 已终评 | 16/16 | 7.01x | 100 | 2.24 | 0.66 |  |
| SOL-L2 | 051_seqlen-finetuned-reconstructed_hyena_complete_forward_block | 违规留档 | — | — | 100 | 2.23 | 0.50 | F.conv2d/F.linear/F.conv1d/torch.fft 共 11 处（纯 torch 委托） |
| SOL-L2 | 057_residual_coupling_flow_block | 违规留档 | — | — | 100 | 1.72 | 0.39 | F.conv1d×3 |
| SOL-L2 | 080_moe_complete_layer_with_shared_expert_backward | 违规留档 | — | — | 104 | 2.09 | 0.41 | torch.mm×4 |

- **已终评 21 题，geomean-of-geomean = 21.10x**（含 gdn_prefill 非标口径；
  剔除后 20 题 = 18.72x）；违规留档 9 题
- 生成时间：2026-09-23 22:21；数字直接来自各题 unified/ 评测 JSON