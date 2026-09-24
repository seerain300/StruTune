# 消融实验对照表

| 题 | 消融批次 | valid | pass | geomean | 对照（原版终数） | 结论 |
|---|---|---|---|---|---|---|
| 005_conv_gated_projection_with_causal_conv | ablation_strict_nolib_20260917 | ✅ | 16/16 | 1.420 | 1.57x(可调库) | 全 Triton 后 -10%，已替代进主表（见 results_table †） |
| 020_vision_patch_merger_spatial_shuffle_mlp | ablation_strict_nolib_20260917 | ✅ | 15/15 | 1.464 | 2.16x(可调库) | 全 Triton 后 -32%，库贡献 1/3，已替代进主表（见 results_table ‡） |
| 020_vision_patch_merger_spatial_shuffle_mlp | ablation_full_feedback_20260917 | ✅ | 15/15 | 2.198 | 2.16x(5wl 抽样) | 全量反馈 +2% |
| 058_moe_expert_token_radix_sort_with_prefix_sum | ablation_resume5_fullfb_20260917 | ✅ | 16/16 | 10.324 | 15/16 INVALID(抽样盲区) | 全量反馈 5 轮修复→16/16 |

被中止的消融（LLM 无视守卫仍调库，无继续意义）：002/007/092 @ strict_nolib（数据在 tasks/ 对应批次目录）
