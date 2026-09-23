#!/usr/bin/env bash
# campaign 监护 v2：每 5 分钟检查三组 campaign 是否还有"活的任务进程"
# （判断依据=任务进程存活，而非主 bash 存活——主 bash 死了但任务还活着时
# 不许重启，避免重复拉起；2026-09-20 夜间的进程滚雪球正是这个盲区造成）。
# 另加重启预算：每 campaign 每小时最多 4 次，超限只记日志不再拉起。
# 注：外部看门狗已退役（其 ts 解析 bug 曾致随机误杀；挂死防护由
# FI 进程内超时 + SOL CLI 自带超时承担）。
WS=/home/ziming/ksearch_h100_portable
FI_TASKS="dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 gdn_decode_qk4_v8_d128_k_last gdn_prefill_qk4_v8_d128_k_last gemm_n4096_k4096 gqa_paged_decode_h32_kv8_d128_ps1 gqa_paged_prefill_causal_h32_kv8_d128_ps1 gqa_ragged_prefill_causal_h32_kv8_d128 mla_paged_decode_h16_ckv512_kpe64_ps1 mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 rmsnorm_h4096"
L1_TASKS="L1/002_vae_conv3x3_groupnorm_silu_residual_fused L1/005_conv_gated_projection_with_causal_conv L1/007_hyena_fft_size_padding_rfft L1/008_expert_output_weighted_index_add_accumulation L1/018_fused_rope_with_qk_norm_and_kv_cache_update L1/020_vision_patch_merger_spatial_shuffle_mlp L1/053_gaussian_topk_sparse_activation L1/058_moe_expert_token_radix_sort_with_prefix_sum L1/070_mamba2_fused_intra_chunk_diagonal_computation L1/092_gqa_attention_with_qk_norm"
L2_TASKS="L2/012_moe_expert_batched_execution_with_capacity_factor L2/015_audio_sinusoidal_position_embedding_with_conv_projection L2/030_flux_concatenated_sequence_processing_with_split L2/036_convnextv2_layer_with_nhwc_persistence_backward L2/040_altup_predict_correction_cycle_backward L2/043_mamba_chunk_scan_with_segsum L2/049_group_limited_topk_routing L2/051_seqlen-finetuned-reconstructed_hyena_complete_forward_block L2/057_residual_coupling_flow_block L2/080_moe_complete_layer_with_shared_expert_backward"

tasks_alive() {  # $1 = 空格分隔的任务名列表（取前3个特征词足够判活）
  for t in $1; do
    if pgrep -f "ksearch[-]run[.]sh ${t} " >/dev/null 2>&1 || pgrep -f -- "--definition ${t}([^a-z_]|$)" >/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

restart_count_today() { grep "$(date '+%m-%d') .*RESTART $1 " /tmp/ksearch_supervisor.log 2>/dev/null | wc -l; }

while true; do
  for entry in "formal_h100:$FI_TASKS:0,4:1:KSEARCH_INVISIBLE_ALLOW_MB=6000:par6" "formal_sol_h100:$L1_TASKS:4,5:6:KSEARCH_POOL_CLAIM_EMPTY=0:sol_l1" "formal_solL2_h100:$L2_TASKS:5:4:KSEARCH_POOL_CLAIM_EMPTY=0:sol_l2"; do
    IFS=':' read -r tag tasks pool conc extra logname <<< "$entry"
    if ! tasks_alive "$tasks"; then
      n=$(restart_count_today "$tag")
      if [ "$n" -ge 4 ]; then
        echo "$(date '+%m-%d %H:%M') SKIP $tag (重启预算耗尽 $n/4)" >> /tmp/ksearch_supervisor.log
        continue
      fi
      echo "$(date '+%m-%d %H:%M') RESTART $tag (第 $((n+1))/4 次)" >> /tmp/ksearch_supervisor.log
      if [ -n "$extra" ]; then
        env KSEARCH_STRICT_NO_LIB=1 $extra nohup bash $WS/scripts/ksearch_pool_campaign.sh \
          --pool "$pool" --concurrency "$conc" --tag "$tag" --tasks $tasks \
          >> /tmp/${logname}_campaign.log 2>&1 &
      else
        env KSEARCH_STRICT_NO_LIB=1 nohup bash $WS/scripts/ksearch_pool_campaign.sh \
          --pool "$pool" --concurrency "$conc" --tag "$tag" --tasks $tasks \
          >> /tmp/${logname}_campaign.log 2>&1 &
      fi
    fi
  done
  sleep 300
done
