# Baseline 评测基准与接口对齐文档（MTMC-baseline 系）

> 用途：任何新 baseline（DRTriton/Dr.Kernel、其他方法）接入同一套 benchmark 时的交接与对齐。
> 维护：weihongren；最后更新 2026-09-16（协议 v0.6）。
> 机器可读范围清单：`/data1/workspace/weihongren/experiment_manifest.json`
> KDA 搜索采样、预算、候选版本与工具审计：`/data1/workspace/weihongren/notes/KDA_EXPERIMENT_PROTOCOL.md`

---

## 1. 题目集（30 题 / 738 workloads，manifest v3）

### 1.1 FlashInfer 子集（10 题 / 426 workloads）

- 数据集根：`/data1/workspace/weihongren/dataset/flashinfer-test`（ziming 原版 `/data1/workspace/ziming/dataset/flashinfer-test` 的逐字节副本；**只读使用，勿写**）
- 结构：`definitions/<类别>/<题名>.json` + `workloads/<类别>/<题名>.jsonl`；部分题的输入是 `blob/workloads/` 下的 safetensors（由评测器按 `--dataset-root` 定位）
- **排除** `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`（sm_80 无 fp8e4nv，编译墙）；**纳入** `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`

| definition 名 | 类别 | workloads |
|---|---|---|
| dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 | dsa_paged | 23 |
| gdn_decode_qk4_v8_d128_k_last | gdn | 54 |
| gdn_prefill_qk4_v8_d128_k_last | gdn | 100 |
| gemm_n4096_k4096 | gemm | 43 |
| gqa_paged_decode_h32_kv8_d128_ps1 | gqa_paged | 48 |
| gqa_paged_prefill_causal_h32_kv8_d128_ps1 | gqa_paged | 38 |
| gqa_ragged_prefill_causal_h32_kv8_d128 | gqa_ragged | 21 |
| mla_paged_decode_h16_ckv512_kpe64_ps1 | mla_paged | 47 |
| mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 | mla_paged | 38 |
| rmsnorm_h4096 | rmsnorm | 14 |

### 1.2 SOL-ExecBench 子集（L1 10 题 / 154 workloads）

- 数据集根：`/data1/workspace/ziming/dataset/SOL-ExecBench`（ziming 目录，只读；评测用其自带 `.venv`）
- 题目路径：`data/benchmark/L1/<题名>/`，每题固定三件套 `definition.json` / `reference.py` / `workload.jsonl`
- **4 题含 custom inputs**（008/018/020/058，`custom_inputs_entrypoint=get_inputs`）：必须走 SOL 官方 CLI 的输入生成，不得换成独立随机输入
- 容差按 workload 内置（`tolerance.max_atol/max_rtol`），无全局统一值

| 题名 | workloads | 备注 |
|---|---|---|
| 002_vae_conv3x3_groupnorm_silu_residual_fused | 20 | |
| 005_conv_gated_projection_with_causal_conv | 16 | |
| 007_hyena_fft_size_padding_rfft | 16 | |
| 008_expert_output_weighted_index_add_accumulation | 16 | custom inputs |
| 018_fused_rope_with_qk_norm_and_kv_cache_update | 13 | custom inputs |
| 020_vision_patch_merger_spatial_shuffle_mlp | 15 | custom inputs |
| 053_gaussian_topk_sparse_activation | 12 | |
| 058_moe_expert_token_radix_sort_with_prefix_sum | 16 | custom inputs |
| 070_mamba2_fused_intra_chunk_diagonal_computation | 14 | |
| 092_gqa_attention_with_qk_norm | 16 | v3 换入（探针实测 ref 2.3ms，健康）；**094 已剔除**（reference 为朴素 Python 逐步循环：单 workload ref 计时 110 次需 ~1h，全量评测 4h+，且 speedup 是对朴素实现的比值——如需复活须配分片评测与超时预案） |

### 1.3 SOL-ExecBench 子集（L2 10 题 / 158 workloads）

- 题目路径：`data/benchmark/L2/<题名>/`，仍使用 SOL 官方 CLI、官方输入生成和 workload 内置容差。
- **6 题含 custom inputs**（012/015/036/051/057/080，`custom_inputs_entrypoint=get_inputs`）：不得绕过官方输入生成。

| 题名 | workloads | 备注 |
|---|---|---|
| 012_moe_expert_batched_execution_with_capacity_factor | 16 | custom inputs |
| 015_audio_sinusoidal_position_embedding_with_conv_projection | 16 | custom inputs |
| 030_flux_concatenated_sequence_processing_with_split | 16 | |
| 036_convnextv2_layer_with_nhwc_persistence_backward | 14 | custom inputs |
| 040_altup_predict_correction_cycle_backward | 16 | |
| 043_mamba_chunk_scan_with_segsum | 16 | |
| 049_group_limited_topk_routing | 16 | |
| 051_seqlen-finetuned-reconstructed_hyena_complete_forward_block | 16 | custom inputs |
| 057_residual_coupling_flow_block | 16 | custom inputs |
| 080_moe_complete_layer_with_shared_expert_backward | 16 | custom inputs |

### 1.4 KDA 候选与评测次数约定

- 先前 vector-add smoke test 只实现 `C1`，是为了验证一次完整的
  `draft → plan → candidate → validate → evaluate → decision` 闭环，并非 KDA
  官方规定每题只能生成一个候选。正式实验应按预算逐候选迭代，达到晋级条件或预算耗尽才停止。
- smoke test 的“5 次候选评测”是对同一个微基准重复执行 5 次，用于计算均值、样本标准差和
  ±1σ 区间；它既不是 5 个 workload，也不是当前正式统一评测的 `trials` 参数。
- 正式论文数字严格采用下文 v0.6 口径：所有 workload 全量评测；FlashInfer
  `warmup=3 / iters=100 / trials=1`；SOL 官方 CLI `iterations=100`。搜索阶段可使用固定 seed
  的反馈子集，但反馈数字不得替代最终全量结果。
- KDA 搜索阶段采用 `random.Random(0).sample(workloads, min(5,N))` 的固定反馈样本，默认每版 kernel 在 5 个 workload 上完成一次
  正确性和性能评测；这整体计为 **1 次候选评测**。每题最多 100 次候选评测。只对最终最佳
  正确候选执行一次全量认证，且全量认证独立记账。
- 每题 token 预算为 1,000,000 软阈值、1,500,000 正常结算阈值和 1,650,000 绝对阈值，定义为
  `input_tokens + output_tokens`。KDA Opus 4.8 代理已改为从完整非流式上游响应重建 SSE，
  Claude Code 和代理日志均可获得非零 output usage；若个别响应仍缺失 usage，必须标记不完整，
  不能把缺失的 output token 当成 0。

---

## 2. 最终统一评测（论文数字唯一来源，协议 v0.6）

**口径：全量 workloads；FlashInfer warmup3 / iters100 / trials1；SOL 显式 warmup10 / iterations100、eval seed 200。**
生成阶段的任何反馈数字（抽样 workload）不得进论文。speedup = ref_latency / sol_latency，同机同进程成对计时。

### 2.1 FlashInfer 题：ziming 的 evaluate.py

脚本：**本地副本 `/data1/workspace/weihongren/evaluators/evaluate.py`**（ziming 的 agent-generation 目录已于 09-15 被他本人移除；本副本取自其 `test/scripts/`，两版本 diff 确认逐字节相同，sha256 见 `evaluators/PROVENANCE.txt`）
运行环境：**mtmc conda env**（`source /data1/workspace/weihongren/activate-eval.sh`）

```bash
CUDA_VISIBLE_DEVICES=<空卡编号> python scripts/evaluate.py \
  --definition <dataset>/definitions/<类别>/<题名>.json \
  --workload    <dataset>/workloads/<类别>/<题名>.jsonl \
  --solution    <候选目录>/solution.py \        # 入口函数 run(*inputs) 返回输出元组（value-returning，非 DPS）
  --entry run \
  --dataset-root /data1/workspace/weihongren/dataset/flashinfer-test \
  --device cuda:0 \
  --warmup 3 --iters 100 --trials 1 \
  --json <候选目录>/evaluation.json
```

- 先正确性后性能：任一 workload 失败 → 整体 INVALID（不计时）
- 输出 `evaluation.json`：`valid / passed / geomean_speedup（优化目标）/ arithmetic_mean_speedup / per_workload[{uuid,status,speedup,ref_ms,sol_ms,axes}]`
- 退出码 0 iff valid
- 便捷包装：`/data1/workspace/weihongren/scripts/unified_eval.py`（K-Search solution JSON → 自动展开 solution.py + 调上表命令 + 汇总 token 到 candidate.json），`--trials` 可调（默认 5，正式用 1）

### 2.2 SOL 题：ziming 的 evaluate_sol.py（官方 CLI 包装）

脚本：**本地副本 `/data1/workspace/weihongren/evaluators/evaluate_sol.py`**（同上，溯源指纹见 PROVENANCE.txt）
运行环境：**必须用 SOL 官方 venv 的 python**（`.venv/bin/python`，脚本按 `sys.executable` 旁的 `sol-execbench` 定位 CLI）

```bash
CUDA_VISIBLE_DEVICES=<空卡编号> /data1/workspace/ziming/dataset/SOL-ExecBench/.venv/bin/python \
  /data1/workspace/ziming/MTMC-baseline/test/scripts/evaluate_sol.py \
  --definition <SOL根>/data/benchmark/L1/<题名>/definition.json \
  --workload    <SOL根>/data/benchmark/L1/<题名>/workload.jsonl \
  --solution    <候选.solution.json 或 solution.py> \
  --output      <候选目录>/performance.json \
  --rerun --iterations 100 --timeout <秒，慢题给足>
```

- `--timeout` 是**整个评测子进程**的超时（默认 300s）；reference 含逐时间步 Python 循环的题全量评测需数小时级（094 即因此类原因于 v3 剔除，勿再引入同类题而不先探针 ref 延迟）
- 输出 `performance.json`：`valid / passed / total / geomean_speedup / arithmetic_mean_speedup / per_workload[{uuid,status,speedup,ref_ms,sol_ms,max_abs,max_rel,axes}]`
- solution.json 为 SOL schema（`spec.languages=["triton"|"pytorch"]`、`entry_point`、`destination_passing_style`）；.py 会被自动包装（推断 DPS）
- 当前采用的 `MTMC-baseline/agent-generation/scripts/evaluate_sol.py` 已支持 `--warmup`；KDA 适配版另外显式支持 `--seed 200` 与反馈阶段 reference latency 缓存。不要依赖默认值。

### 2.3 GPU 与公平性要求

- 计时一律**独占空卡**（显存占用≈0）；用 `CUDA_VISIBLE_DEVICES=<物理卡> + --device cuda:0` 映射
- benchmark 子进程须在宿主机终端环境跑（沙箱内 /dev 被覆盖时 nvidia 相关会失败）
- 各方法候选在**同一台机器、同一评测脚本、同一参数**下测；不得跨 harness 混比 FlashInfer/SOL 数字（分别报告）

---

## 3. 搜索反馈评测（search-based baseline 的优化过程用）

与最终评测的唯一差别是 **workload 数：固定 seed 抽样 5 个**（`random.Random(seed).sample(workloads, min(5,N))`，uuid 序列记录在 run 元数据；seed0/1/2 对应独立重复）。

- 反馈计时（协议 v0.4）：FlashInfer warmup3/iters100/trials5；SOL warmup10/iters100。正式 K-Search run 中 gqa_paged_prefill/094 两题经用户确认改用轻量反馈（trials1 / iters30），只影响搜索信号
- **reference 延迟缓存**（可选加速，不影响最终数字）：首轮实测 ref 延迟落盘（键=题名+计时参数+硬件），后续轮 driver 跳过 ref 计时、客户端用缓存算 speedup；实现见 `K-Search/k_search/tasks/{flashinfer_bench_task,sol_execbench_task}.py`，缓存目录 `baseline/ksearch{,-sol-execbench}/.ref_latency_cache/`
- K-Search 的任务后端（可参考实现）：`k_search/tasks/flashinfer_bench_task.py`（调 flashinfer_bench 包）、`k_search/tasks/sol_execbench_task.py`（子进程调 SOL 官方 CLI，staging: definition+抽样workload+solution+config{warmup,iterations,benchmark_reference,seed=200}）

---

## 4. 产物目录与 token 记账约定

### 4.1 目录结构

```
baseline/<method>/experiments/<tag>/<题名>/run_seed<N>/     # FlashInfer 题
baseline/<method>-sol-execbench/experiments/<tag>/<题名>/run_seed<N>/   # SOL 题
  ├── usage.jsonl            # 每次 LLM 调用的 token 记账（见 4.2）
  ├── campaign_stdout.log    # 全量 stdout（含每轮 round 标记与反馈摘要行）
  ├── ksearch-artifacts/     # K-Search: solutions/<题>/*.json、world_model/、eval/
  ├── unified/               # 最终评测产物 evaluation.json/performance.json + candidate.json
  ├── DONE / exit_code       # 编排完成标记 / 退出码
  └── final_eval.log
```

废弃尝试归档为 `run_seed<N>_attempt1_<口径>_<轮数>r`（token 记录保留但**不并入**新统计）。

### 4.2 token 记账 schema（llm-usage-v1）

工具：monkey-patch OpenAI SDK（参考 `/data1/workspace/weihongren/ksearch-token-run.py`，`KSEARCH_LOG_PROMPT_HASH=1` 可加 prompt sha256 用于缓存取证）。逐调用写 JSONL：

| 字段 | 定义 |
|---|---|
| input_tokens | 该次调用输入 token（=prompt_tokens） |
| input_cached_tokens | provider 报告的缓存命中输入 token（**input 的子集**） |
| input_uncached_tokens | input − cached（provider 报告缓存时） |
| output_tokens | 输出 token（=completion_tokens） |
| reasoning_tokens | provider 单列的推理 token（**output 的子集**，非增量；读 completion_tokens_details.reasoning_tokens） |
| total_tokens | input + output |
| cache_usage_reported / reasoning_usage_reported | provider 是否明确报告（区分"报 0"与"未报告"） |

汇总：`scripts/summarize_llm_usage.py`（含 per-model 分组）；评测侧 `unified_eval.py`/`ksearch_final_eval.py` 的 `aggregate_usage()` 把 run 的 token 并入 candidate.json。
端点缓存行为（已实测）：llmapi 网关为**逐字节精确匹配缓存**（改 1 token/截断/加尾全部不命中；命中≈input−3；无部分前缀缓存；小 prompt 不缓存）。

---

## 5. 环境与版本指纹

| 项 | 值 |
|---|---|
| GPU | 8× NVIDIA A800-SXM4-80GB（sm_80），驱动 580.105.08 |
| 生成+FlashInfer 评测 env | conda `mtmc`：Python 3.11 / torch 2.13.0+cu130 / triton 3.7.1 / flashinfer-bench 0.1.2（`activate-ksearch.sh` / `activate-eval.sh`） |
| SOL 评测 env | SOL 官方 `.venv`：Python 3.12 / torch 2.9.0+cu130 / triton 3.5.0 / sol-execbench 1.0.2 |
| K-Search | caoshiyi/K-Search@53c8fab9a5e8fab2c86610d24fbec5067f90e115 + 本地适配（sol_execbench_task 等，diff 见归档 versions.json） |
| LLM 端点 | AWS-GPT-5.6-Sol @ llmapi.isrc.ac.cn/v1（`llm.env`，权限 600）；OpenAI 兼容 chat.completions |
| 编排 | `scripts/ksearch_campaign.sh`：--gpus auto 自动探测全空卡（阈值 200MiB）+ 动态补位 + 启动前复查 + DONE 断点续跑 |

---

## 6. 协议版本历史

| 版本 | 变更 |
|---|---|
| v0.2 | 初始：反馈 10/10/1（K-Search 官方默认），最终 3/100/5 |
| v0.3 | seed 贯穿反馈抽样；SOL 用官方独立评测接口；manifest v1（dsa_topk 在内） |
| v0.4 | 反馈计时与统一评测器对齐（FlashInfer 3/100/5、SOL 10/100）；**manifest v2：dsa_sparse 换入、dsa_topk 移除**（sm_80 fp8 墙） |
| v0.5 | **最终评测 trials 5→1**（用户确认）；SOL 侧本就是单次校验 |
| v0.6 | 统一范围扩展为 FlashInfer 10 + SOL L1 10 + SOL L2 10，共 30 题 / 738 workloads；明确 KDA 候选次数与正式评测次数的区别 |

## 7. 已知坑清单（新 baseline 接入前必读）

1. KDA 使用的 SOL evaluator 基于 `MTMC-baseline/agent-generation/scripts/evaluate_sol.py`；显式传入 warmup、iterations 和 seed，避免不同历史副本的默认值漂移
2. `--timeout`（SOL）是整批 workloads 一个子进程的总超时，慢题须按 题耗时×workload数 放大
3. FlashInfer 题输入含 safetensors blob，必须传 `--dataset-root` 否则找不到输入
4. trials 概念只在 FlashInfer 侧存在；SOL 固定单次正确性+单轮计时
5. sm_80（A800）不支持 Triton fp8e4nv——生成类 baseline 的 prompt 里可提示改用 int8 加载换算
6. token 记账只记成功返回的调用；SDK 自动重试的失败尝试不计（缓存取证用 prompt_sha256）
7. FlashInfer 与 SOL 的分数分开报告，不跨 harness 比较

---

## 8. 历史正式批次（formal_20260914，旧 20 题范围）状态快照

本节仅记录 2026-09-14 启动的旧范围批次，不代表 v0.6 的 30 题 KDA 范围。

- 18 题搜索完成（100 轮 = 20 节点×5 attempt，WM 完整版，seed0）；gqa_paged_prefill 已以轻量反馈重跑完成（392.9x）；**094 于 v3 剔除（搜索完成但评测中止，原因见题目表），092 搜索进行中（GPU4）**
- 全量评测（trials=1）：15 题已完成（14 valid；058 因 1/16 边界形状数值错误记 INVALID），3 道重题评测进行中
- 报告：`experiments_archive/formal_20260914/ksearch_formal_report.md`（指标定义/成绩/token 表；`scripts/ksearch_formal_report.py` 可重生成）
- 全部完成后执行 `scripts/ksearch_formal_archive.py --tag formal_20260914 --with-eval` 生成完整归档（代码版本/配置/评测/token → `experiments_archive/formal_20260914/`）
