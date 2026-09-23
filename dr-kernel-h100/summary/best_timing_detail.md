# 最优解逐 workload 计时明细（终局官方复评：warmup 3 + 100 iters，CUDA event 均值）

每个 workload 一行：axes = 输入形状配置；ref_ms = PyTorch 参考实现毫秒；
sol_ms = 模型 Triton 内核毫秒；speedup = ref_ms / sol_ms。

## flashinfer/gemm_n4096_k4096（1sample，geomean 0.36×，43/43 PASSED）

| # | axes | ref_ms | sol_ms | speedup | status |
|---:|---|---:|---:|---:|---|
| 1 | `{"M": 256}` | 0.0341 | 0.1211 | 0.28× | PASSED |
| 2 | `{"M": 248}` | 0.0347 | 0.1204 | 0.29× | PASSED |
| 3 | `{"M": 240}` | 0.0329 | 0.1223 | 0.27× | PASSED |
| 4 | `{"M": 232}` | 0.0335 | 0.1218 | 0.27× | PASSED |
| 5 | `{"M": 224}` | 0.0315 | 0.1163 | 0.27× | PASSED |
| 6 | `{"M": 216}` | 0.0327 | 0.1135 | 0.29× | PASSED |
| 7 | `{"M": 208}` | 0.0312 | 0.1053 | 0.30× | PASSED |
| 8 | `{"M": 200}` | 0.0320 | 0.1040 | 0.31× | PASSED |
| 9 | `{"M": 192}` | 0.0303 | 0.0810 | 0.37× | PASSED |
| 10 | `{"M": 184}` | 0.0315 | 0.0815 | 0.39× | PASSED |
| 11 | `{"M": 176}` | 0.0300 | 0.0814 | 0.37× | PASSED |
| 12 | `{"M": 168}` | 0.0309 | 0.0812 | 0.38× | PASSED |
| 13 | `{"M": 160}` | 0.0294 | 0.0803 | 0.37× | PASSED |
| 14 | `{"M": 152}` | 0.0310 | 0.0802 | 0.39× | PASSED |
| 15 | `{"M": 144}` | 0.0289 | 0.0808 | 0.36× | PASSED |
| 16 | `{"M": 136}` | 0.0297 | 0.0794 | 0.37× | PASSED |
| 17 | `{"M": 128}` | 0.0284 | 0.0959 | 0.30× | PASSED |
| 18 | `{"M": 120}` | 0.0283 | 0.0721 | 0.39× | PASSED |
| 19 | `{"M": 112}` | 0.0282 | 0.0719 | 0.39× | PASSED |
| 20 | `{"M": 104}` | 0.0280 | 0.0722 | 0.39× | PASSED |
| 21 | `{"M": 96}` | 0.0280 | 0.0720 | 0.39× | PASSED |
| 22 | `{"M": 88}` | 0.0279 | 0.0737 | 0.38× | PASSED |
| 23 | `{"M": 80}` | 0.0278 | 0.0708 | 0.39× | PASSED |
| 24 | `{"M": 72}` | 0.0285 | 0.0711 | 0.40× | PASSED |
| 25 | `{"M": 64}` | 0.0272 | 0.0658 | 0.41× | PASSED |
| 26 | `{"M": 56}` | 0.0282 | 0.0655 | 0.43× | PASSED |
| 27 | `{"M": 48}` | 0.0274 | 0.0658 | 0.42× | PASSED |
| 28 | `{"M": 40}` | 0.0269 | 0.0649 | 0.41× | PASSED |
| 29 | `{"M": 32}` | 0.0269 | 0.0648 | 0.42× | PASSED |
| 30 | `{"M": 24}` | 0.0268 | 0.0649 | 0.41× | PASSED |
| 31 | `{"M": 16}` | 0.0289 | 0.0648 | 0.45× | PASSED |
| 32 | `{"M": 8}` | 0.0267 | 0.0643 | 0.41× | PASSED |
| 33 | `{"M": 4}` | 0.0283 | 0.0642 | 0.44× | PASSED |
| 34 | `{"M": 2}` | 0.0283 | 0.0642 | 0.44× | PASSED |
| 35 | `{"M": 1}` | 0.0284 | 0.0648 | 0.44× | PASSED |
| 36 | `{"M": 7}` | 0.0284 | 0.0647 | 0.44× | PASSED |
| 37 | `{"M": 35}` | 0.0273 | 0.0650 | 0.42× | PASSED |
| 38 | `{"M": 972}` | 0.0669 | 0.3290 | 0.20× | PASSED |
| 39 | `{"M": 70}` | 0.0282 | 0.0779 | 0.36× | PASSED |
| 40 | `{"M": 2053}` | 0.1639 | 0.6027 | 0.27× | PASSED |
| 41 | `{"M": 8192}` | 0.6306 | 2.5983 | 0.24× | PASSED |
| 42 | `{"M": 2379}` | 0.1893 | 0.6893 | 0.27× | PASSED |
| 43 | `{"M": 15}` | 0.0309 | 0.0687 | 0.45× | PASSED |

## flashinfer/rmsnorm_h4096（1sample，geomean 1.38×，14/14 PASSED）

| # | axes | ref_ms | sol_ms | speedup | status |
|---:|---|---:|---:|---:|---|
| 1 | `{"batch_size": 7}` | 0.0790 | 0.0848 | 0.93× | PASSED |
| 2 | `{"batch_size": 1}` | 0.0774 | 0.0840 | 0.92× | PASSED |
| 3 | `{"batch_size": 34}` | 0.0778 | 0.0849 | 0.92× | PASSED |
| 4 | `{"batch_size": 170}` | 0.0765 | 0.0844 | 0.91× | PASSED |
| 5 | `{"batch_size": 14418}` | 0.9794 | 0.3397 | 2.88× | PASSED |
| 6 | `{"batch_size": 11832}` | 0.8338 | 0.2901 | 2.87× | PASSED |
| 7 | `{"batch_size": 64}` | 0.0776 | 0.0842 | 0.92× | PASSED |
| 8 | `{"batch_size": 16}` | 0.0775 | 0.0825 | 0.94× | PASSED |
| 9 | `{"batch_size": 10827}` | 0.7497 | 0.2700 | 2.78× | PASSED |
| 10 | `{"batch_size": 8804}` | 0.6308 | 0.2291 | 2.75× | PASSED |
| 11 | `{"batch_size": 63}` | 0.0813 | 0.0840 | 0.97× | PASSED |
| 12 | `{"batch_size": 79}` | 0.0779 | 0.0843 | 0.92× | PASSED |
| 13 | `{"batch_size": 14509}` | 0.9948 | 0.3410 | 2.92× | PASSED |
| 14 | `{"batch_size": 15}` | 0.0773 | 0.0831 | 0.93× | PASSED |

## SOL/L1/008_expert_output_weighted_index_add_accumulation（1sample，geomean 2.82×，16/16 PASSED）

| # | axes | ref_ms | sol_ms | speedup | status |
|---:|---|---:|---:|---:|---|
| 1 | `{"batch_size": 2, "seq_len": 1024}` | 0.4154 | 0.1021 | 4.07× | PASSED |
| 2 | `{"batch_size": 4, "seq_len": 541}` | 0.4395 | 0.1075 | 4.09× | PASSED |
| 3 | `{"batch_size": 2, "seq_len": 128}` | 0.0555 | 0.0409 | 1.36× | PASSED |
| 4 | `{"batch_size": 1, "seq_len": 8192}` | 1.7413 | 0.4028 | 4.32× | PASSED |
| 5 | `{"batch_size": 4, "seq_len": 512}` | 0.4163 | 0.1019 | 4.08× | PASSED |
| 6 | `{"batch_size": 1, "seq_len": 512}` | 0.1084 | 0.0546 | 1.98× | PASSED |
| 7 | `{"batch_size": 16, "seq_len": 256}` | 0.8512 | 0.1977 | 4.30× | PASSED |
| 8 | `{"batch_size": 2, "seq_len": 512}` | 0.2102 | 0.0683 | 3.08× | PASSED |
| 9 | `{"batch_size": 4, "seq_len": 256}` | 0.2107 | 0.0685 | 3.08× | PASSED |
| 10 | `{"batch_size": 1, "seq_len": 131}` | 0.0307 | 0.0390 | 0.79× | PASSED |
| 11 | `{"batch_size": 1, "seq_len": 1024}` | 0.2102 | 0.0689 | 3.05× | PASSED |
| 12 | `{"batch_size": 2, "seq_len": 1879}` | 0.7789 | 0.1824 | 4.27× | PASSED |
| 13 | `{"batch_size": 2, "seq_len": 256}` | 0.1092 | 0.0534 | 2.05× | PASSED |
| 14 | `{"batch_size": 1, "seq_len": 256}` | 0.0559 | 0.0438 | 1.27× | PASSED |
| 15 | `{"batch_size": 64, "seq_len": 128}` | 1.7446 | 0.4029 | 4.33× | PASSED |
| 16 | `{"batch_size": 32, "seq_len": 256}` | 1.7482 | 0.4036 | 4.33× | PASSED |

## SOL/L1/053_gaussian_topk_sparse_activation（7samples，geomean 12.94×，12/12 PASSED）

| # | axes | ref_ms | sol_ms | speedup | status |
|---:|---|---:|---:|---:|---|
| 1 | `{"batch_size": 1, "seq_len": 512, "intermediate_size": 12288}` | 12.4958 | 0.0661 | 189.15× | PASSED |
| 2 | `{"batch_size": 4, "seq_len": 2048, "intermediate_size": 12288}` | 13.8558 | 1.0361 | 13.37× | PASSED |
| 3 | `{"batch_size": 32, "seq_len": 128, "intermediate_size": 12288}` | 1.2738 | 0.4753 | 2.68× | PASSED |
| 4 | `{"batch_size": 2, "seq_len": 211, "intermediate_size": 8192}` | 0.5852 | 0.1539 | 3.80× | PASSED |
| 5 | `{"batch_size": 1, "seq_len": 8192, "intermediate_size": 4096}` | 13.8068 | 0.3838 | 35.98× | PASSED |
| 6 | `{"batch_size": 1, "seq_len": 1024, "intermediate_size": 16384}` | 12.6952 | 0.1761 | 72.10× | PASSED |
| 7 | `{"batch_size": 16, "seq_len": 1163, "intermediate_size": 8192}` | 2.9234 | 1.4513 | 2.01× | PASSED |
| 8 | `{"batch_size": 4, "seq_len": 541, "intermediate_size": 8192}` | 0.8052 | 0.1740 | 4.63× | PASSED |
| 9 | `{"batch_size": 4, "seq_len": 449, "intermediate_size": 4096}` | 0.6471 | 0.1179 | 5.49× | PASSED |
| 10 | `{"batch_size": 64, "seq_len": 1024, "intermediate_size": 8192}` | 25.8703 | 13.6309 | 1.90× | PASSED |
| 11 | `{"batch_size": 2, "seq_len": 131, "intermediate_size": 4096}` | 13.1519 | 0.0210 | 627.00× | PASSED |
| 12 | `{"batch_size": 2, "seq_len": 293, "intermediate_size": 12288}` | 0.6314 | 0.1171 | 5.39× | PASSED |

## SOL/L1/058_moe_expert_token_radix_sort_with_prefix_sum（7samples，geomean 6.75×，16/16 PASSED）

| # | axes | ref_ms | sol_ms | speedup | status |
|---:|---|---:|---:|---:|---|
| 1 | `{"batch_size": 8, "seq_len": 256}` | 5.0025 | 0.1412 | 35.42× | PASSED |
| 2 | `{"batch_size": 64, "seq_len": 128}` | 5.3645 | 0.1871 | 28.67× | PASSED |
| 3 | `{"batch_size": 2, "seq_len": 1024}` | 0.2196 | 0.0915 | 2.40× | PASSED |
| 4 | `{"batch_size": 32, "seq_len": 128}` | 5.0361 | 0.1131 | 44.52× | PASSED |
| 5 | `{"batch_size": 4, "seq_len": 512}` | 5.2909 | 0.1325 | 39.92× | PASSED |
| 6 | `{"batch_size": 1, "seq_len": 2048}` | 0.1948 | 0.1280 | 1.52× | PASSED |
| 7 | `{"batch_size": 4, "seq_len": 544}` | 0.1923 | 0.1295 | 1.49× | PASSED |
| 8 | `{"batch_size": 2, "seq_len": 1056}` | 0.2179 | 0.1421 | 1.53× | PASSED |
| 9 | `{"batch_size": 2, "seq_len": 1088}` | 0.2296 | 0.1429 | 1.61× | PASSED |
| 10 | `{"batch_size": 2, "seq_len": 2048}` | 5.3602 | 0.1455 | 36.83× | PASSED |
| 11 | `{"batch_size": 1, "seq_len": 2080}` | 5.0171 | 0.0921 | 54.50× | PASSED |
| 12 | `{"batch_size": 1, "seq_len": 2112}` | 5.0394 | 0.1025 | 49.15× | PASSED |
| 13 | `{"batch_size": 2, "seq_len": 1120}` | 0.2056 | 0.0990 | 2.08× | PASSED |
| 14 | `{"batch_size": 1, "seq_len": 4096}` | 0.2269 | 0.1436 | 1.58× | PASSED |
| 15 | `{"batch_size": 4, "seq_len": 1024}` | 0.2275 | 0.1447 | 1.57× | PASSED |
| 16 | `{"batch_size": 8, "seq_len": 288}` | 0.2211 | 0.1468 | 1.51× | PASSED |

## SOL/L2/030_flux_concatenated_sequence_processing_with_split（7samples，geomean 0.14×，16/16 PASSED）

| # | axes | ref_ms | sol_ms | speedup | status |
|---:|---|---:|---:|---:|---|
| 1 | `{"batch_size": 2, "text_seq_len": 128, "img_seq_len": 256}` | 0.4849 | 5.7024 | 0.09× | PASSED |
| 2 | `{"batch_size": 1, "text_seq_len": 256, "img_seq_len": 512}` | 0.4378 | 5.8354 | 0.08× | PASSED |
| 3 | `{"batch_size": 2, "text_seq_len": 131, "img_seq_len": 293}` | 0.4682 | 4.9652 | 0.09× | PASSED |
| 4 | `{"batch_size": 1, "text_seq_len": 128, "img_seq_len": 256}` | 0.2313 | 1.3061 | 0.18× | PASSED |
| 5 | `{"batch_size": 1, "text_seq_len": 77, "img_seq_len": 4096}` | 4.6606 | 34.7405 | 0.13× | PASSED |
| 6 | `{"batch_size": 1, "text_seq_len": 512, "img_seq_len": 2048}` | 1.3344 | 9.0974 | 0.15× | PASSED |
| 7 | `{"batch_size": 1, "text_seq_len": 77, "img_seq_len": 1024}` | 0.5666 | 3.7622 | 0.15× | PASSED |
| 8 | `{"batch_size": 4, "text_seq_len": 77, "img_seq_len": 1024}` | 2.7671 | 15.1529 | 0.18× | PASSED |
| 9 | `{"batch_size": 1, "text_seq_len": 1024, "img_seq_len": 4096}` | 5.8701 | 43.6858 | 0.13× | PASSED |
| 10 | `{"batch_size": 16, "text_seq_len": 128, "img_seq_len": 256}` | 5.9283 | 36.8349 | 0.16× | PASSED |
| 11 | `{"batch_size": 32, "text_seq_len": 256, "img_seq_len": 512}` | 13.1720 | 88.7743 | 0.15× | PASSED |
| 12 | `{"batch_size": 4, "text_seq_len": 256, "img_seq_len": 512}` | 1.9435 | 10.5717 | 0.18× | PASSED |
| 13 | `{"batch_size": 2, "text_seq_len": 1423, "img_seq_len": 1489}` | 3.2390 | 20.3978 | 0.16× | PASSED |
| 14 | `{"batch_size": 1, "text_seq_len": 1087, "img_seq_len": 1163}` | 1.2674 | 8.3491 | 0.15× | PASSED |
| 15 | `{"batch_size": 4, "text_seq_len": 211, "img_seq_len": 449}` | 1.3568 | 9.1061 | 0.15× | PASSED |
| 16 | `{"batch_size": 4, "text_seq_len": 128, "img_seq_len": 256}` | 0.8463 | 5.2110 | 0.16× | PASSED |

## flashinfer/mla_paged_decode（未解出，无终局复评计时）

