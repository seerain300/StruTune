# drkernel-8b STTS 结果总表（H100, 2026-09-22）

协议：STTS 10 迭代 × 5 轮/段，best-window keep=4，patience=0，temperature=1.0 / top-p=0.95 / max-tokens=8192。
final speedup = 官方评测器全新复评（correctness-only + 100-iter 计时）。

| 题目 | wl | 1采样 pass@1 | 1采样 final | 7采样 pass@1 | 7采样 final | 最优批次 | 最优 speedup | 最优解来源 |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `flashinfer/gemm_n4096_k4096` | 43 | 1.00 | 0.36× | — | — | 1sample | 0.36× | s0t10 |
| `flashinfer/rmsnorm_h4096` | 14 | 1.00 | 1.38× | — | — | 1sample | 1.38× | s0t33 |
| `flashinfer/mla_paged_decode_h16_ckv512_kpe64_ps1` | 47 | 0.00 | — | 0.00 | — | — | — | — |
| `SOL/L1/008_expert_output_weighted_index_add_accumulation` | 16 | 1.00 | 2.82× | — | — | 1sample | 2.82× | s0t54 |
| `SOL/L1/053_gaussian_topk_sparse_activation` | 12 | 0.00 | — | 0.86 | 12.94× | 7samples | 12.94× | s0t5 |
| `SOL/L1/058_moe_expert_token_radix_sort_with_prefix_sum` | 16 | 0.00 | — | 0.57 | 6.75× | 7samples | 6.75× | s1t0 |
| `SOL/L2/030_flux_concatenated_sequence_processing_with_split` | 16 | 0.00 | — | 0.43 | 0.14× | 7samples | 0.14× | s4t2 |
