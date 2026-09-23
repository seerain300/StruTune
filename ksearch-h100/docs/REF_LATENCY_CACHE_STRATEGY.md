# Reference 延迟缓存策略（30 题完整分析）

> 2026-09-18 | 基于 formal_20260914 + formal2_solL2_20260916 全量评测实际数据计算。
> 公式：trials=1, iter=20, warmup=3(FI)/10(SOL), sol_ms = ref_ms / geomean_speedup。

---

## 1. 缓存机制

搜索反馈的 benchmark 每轮需要候选的 speedup = ref_latency / sol_latency。
ref_latency 对同一 workload 是确定性的（同代码、同 GPU、同参数），重复测试纯属浪费。

**现有实现**（两个后端均已内置）：
- 第 1 轮：`benchmark_reference=True`，同时测 ref + sol → ref 延迟落盘
- 第 2~100 轮：`benchmark_reference=False`，只测 sol → speedup = 缓存 ref / 实测 sol
- 缓存文件：`baseline/ksearch*/.ref_latency_cache/<题名>_<warmup>_<iter>_<seed>.json`
- 最终评测不走缓存（协议要求同进程成对计时 ref + sol）

**代价**：第 1 轮比后续轮慢（要测 ref），一次性成本。
**收益**：后续 99 轮每轮节省 `n_wl × calls × ref_ms` 的时间。

---

## 2. 30 题分档表

### 第一档：必须缓存（ref > 100ms，不缓存每轮多等数分钟至半小时）

| # | 任务 | bench | wl | ref(ms) | sol(ms) | 不缓存每轮 | 缓存后每轮 | 99轮节省 | 加速比 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | gdn_prefill_qk4_v8 | FI | 100 | 671.8 | 3.4 | 25.9 min | 7.9s | **42.5h** | 195x |
| 2 | mla_paged_prefill_causal | FI | 38 | 465.8 | 2.7 | 6.8 min | 2.4s | **11.2h** | 173x |
| 3 | 036_convnextv2_nhwc_bwd | SOL | 14 | 454.7 | 1.3 | 3.3 min | 5.5s | **5.3h** | 364x |
| 4 | gqa_paged_prefill_causal | FI | 38 | 198.8 | 0.5 | 2.9 min | 0.4s | **4.8h** | 393x |
| 5 | gqa_paged_decode_h32 | FI | 48 | 144.0 | 1.2 | 2.7 min | 1.3s | **4.4h** | 125x |

这 5 题的共同特征：**reference 是朴素/暴力实现**（Python 循环、逐 permute、未融合），导致 ref 延迟极高。K-Search 生成的候选实现了大幅加速（125~393x），因此 sol 极快。如果不缓存 ref，每轮 benchmark 时间的 95%+ 都花在重复测 ref 上。

### 第二档：值得缓存（ref 10~100ms，99 轮省 10~60 分钟）

| # | 任务 | bench | wl | ref(ms) | sol(ms) | 不缓存每轮 | 缓存后 | 99轮节省 |
|---|---|---|---|---|---|---|---|---|
| 6 | 040_altup_bwd | SOL | 16 | 30.9 | 1.8 | 20.7s | 5.9s | 24.5m |
| 7 | gdn_decode_qk4_v8 | FI | 54 | 29.0 | 0.1 | 36.2s | 0.2s | 59.4m |
| 8 | 070_mamba2_intra_chunk | SOL | 14 | 21.4 | 0.1 | 14.0s | 5.0s | 14.8m |
| 9 | 012_moe_batched_exec | SOL | 16 | 18.3 | 12.7 | 19.9s | 11.1s | 14.5m |
| 10 | 002_vae_conv3x3 | SOL | 20 | 10.6 | 7.6 | 15.9s | 9.6s | 10.5m |
| 11 | mla_paged_decode_h16 | FI | 47 | 10.4 | 0.4 | 11.6s | 0.4s | 18.5m |

### 第三档：可选/不需要（ref < 10ms，省的时间可忽略）

| # | 任务 | ref(ms) | 99轮节省 | 结论 |
|---|---|---|---|---|
| 12 | 057_residual_coupling | 8.9 | 7.1m | 可选 |
| 13 | 015_audio_sinusoidal | 5.3 | 4.2m | 可选 |
| 14 | 092_gqa_attention_qk_norm | 4.3 | 3.4m | 可选 |
| 15~28 | （14 题 ref 1~3.5ms） | <3.5 | <3m | 不需要 |
| 29 | rmsnorm_h4096 | 0.7 | 0.4m | 不需要 |
| 30 | gemm_n4096_k4096 | 0.1 | 0.1m | 不需要 |

---

## 3. 汇总

| 档次 | 题数 | ref 范围 | 99 轮累计节省 | 建议 |
|---|---|---|---|---|
| 必须缓存 | **5** | 144~672ms | **68.2h** | 不缓存则搜索不可行 |
| 值得缓存 | **6** | 10~31ms | **2.5h** | 开了没坏处 |
| 可选/不需要 | **19** | 0.1~9ms | <1h | 随意 |
| **合计** | **30** | | **~71h** | |

| 全批成本 | 值 |
|---|---|
| 建缓存一次性（第 1 轮自动完成） | **44 分钟** |
| 全批 99 轮 benchmark（缓存后） | **8.1 小时** |
| 全批 100 轮 LLM 调用（真正的瓶颈） | **~100 小时** |

---

## 4. 操作建议

### 当前实现（已生效，无需改动）

缓存自动开启。第 1 轮建缓存时每题多花几秒~几分钟，后续 99 轮受益。

### H100 复现时

缓存文件按硬件型号隔离（FlashInfer 侧 key 含 GPU 名），A800 的缓存在 H100 上不会命中。两种选择：

| 方案 | 操作 | 适用 |
|---|---|---|
| **自动重建**（推荐） | 什么都不做，第 1 轮自动建 | 全部 30 题 |
| 拷贝第一档缓存 | 只拷 5 个题的 JSON 文件到目标机同路径 | 急于启动、不想等第 1 轮慢 |

第一档缓存文件路径（需要拷贝的仅这 5 个）：
```
baseline/ksearch/.ref_latency_cache/
  gdn_prefill_qk4_v8_d128_k_last_w3_i20_t1_<GPU名>.json
  mla_paged_prefill_causal_h16_ckv512_kpe64_ps1_w3_i20_t1_<GPU名>.json
  gqa_paged_prefill_causal_h32_kv8_d128_ps1_w3_i20_t1_<GPU名>.json
  gqa_paged_decode_h32_kv8_d128_ps1_w3_i20_t1_<GPU名>.json
baseline/ksearch-sol-execbench/.ref_latency_cache/
  036_convnextv2_layer_with_nhwc_persistence_backward_w10_i20_s200.json
```

### 注意事项

1. **缓存 key 包含计时参数**：改 warmup/iterations/seed 后 key 变化，自动重建，不会错配旧缓存
2. **SOL 缓存 key 不含硬件名**：跨机器拷贝 SOL 缓存时需确保计时参数一致，否则 key 不匹配自动重建
3. **最终评测不用缓存**：论文数字走同进程成对计时（协议 §3），缓存只影响搜索反馈信号
4. **ref 延迟是确定性的**（同代码+同GPU+同参数），缓存值可安全复用于 speedup 计算

---

## 5. v0.6.H100 增补（2026-09-20）：按题目的缓存开关

- `experiment_manifest.json` 每题新增 `ref_cache` 字段：真 = 反馈测评"首轮实测落盘、
  后续读缓存"；假 = 每轮重测 ref。**最终精确评测一律不用缓存**（不变）。
- 依据上表"不缓存每轮 >30s"的判定线（A800 实测数据，H100 近似类比），
  `ref_cache=true` 的 6 题：gdn_prefill_qk4_v8_d128_k_last（25.9min）、
  mla_paged_prefill_causal_h16_ckv512_kpe64_ps1（6.8min）、
  L2/036_convnextv2_layer_with_nhwc_persistence_backward（3.3min）、
  gqa_paged_prefill_causal_h32_kv8_d128_ps1（2.9min）、
  gqa_paged_decode_h32_kv8_d128_ps1（2.7min）、gdn_decode_qk4_v8_d128_k_last（36.2s）。
- 实现接线：`ksearch-run.sh` 按题名 case 表导出 `KSEARCH_REF_CACHE=1/0`，
  两个任务后端读该变量门控 ref 缓存的读取/写入。
- 缓存粒度确认：两个后端的缓存均为**逐 workload 记录**（FI 按 workload uuid、
  SOL 按 workload id，每条一个延迟值），非全 wl 汇总。
