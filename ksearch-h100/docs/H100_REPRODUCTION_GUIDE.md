# K-Search H 卡复现指南（2026-09-18 终版）

> 目标：在另一台 H 系列 GPU（如 H100/H800）上从零复现 K-Search 全流程。
> 按最后一次实验配置写（v0.5 口径 + 强化守卫 + 全量反馈消融可选）。
> API key 在目标机器 bashrc 里，无需额外配置。

---

## 1. 需要拷贝的文件清单

### 1.1 代码（从 weihongren 目录）

```
# K-Search 仓库（含全部本地适配）
K-Search/                          # 整个目录，含未提交的修改
  k_search/tasks/sol_execbench_task.py     # SOL 后端（本地新增）
  k_search/tasks/flashinfer_bench_task.py  # FlashInfer 后端（ref缓存 + -inf补丁 + strict守卫）
  k_search/utils/gpu_pool.py               # GPU 池化（本地新增）
  generate_kernels_and_eval.py             # 入口（已加 sol task source）

# 入口与编排脚本
ksearch-run.sh                     # 统一搜索入口（key 优先级链 + 补丁shim + bench args）
scripts/
  ksearch_campaign.sh              # 编排（auto GPU / 并发 / DONE 续跑）
  ksearch_pool_campaign.sh         # 池化编排（flock GPU 池 + 续跑感知）
  ksearch_final_eval.py            # 最终评测（FlashInfer→unified_eval / SOL→evaluate_sol）
  ksearch_formal_report.py         # 报告生成
  ksearch_formal_archive.py        # 归档（代码版本/配置/token/评测）
  ksearch_l2_report.py             # L2 报告
  ksearch_merged_report.py         # 30 题合并报告（含审计+消融）
  summarize_llm_usage.py           # token 汇总

# 评测器本地副本（不再依赖 ziming 目录）
evaluators/
  evaluate.py                      # FlashInfer 评测器（sha256 见 PROVENANCE.txt）
  evaluate_sol.py                  # SOL 评测器
  PROVENANCE.txt

# -inf 判官补丁 shim
ksearch_patchshim/
  sitecustomize.py                 # 注入 spawn 子进程（勿删！）

# 配置
llm.env                            # API key/URL/model（目标机可能已有；勿拷贝 key，用目标机的）
experiment_manifest.json           # 题目范围定义（v3）
protocol.md                        # 实验协议
```

### 1.2 数据集

```
# FlashInfer 题（副本，ziming 原版只读）
dataset/flashinfer-test/           # ~1.9GB，含 definitions/workloads/blob

# SOL-ExecBench（ziming 目录，含评测 venv）
# 目标机需要独立 clone SOL-ExecBench 仓库并建 venv（见 §2.2）
# 数据部分：
#   data/benchmark/L1/  （L1 题目 definition.json + workload.jsonl + reference.py）
#   data/benchmark/L2/  （L2 题目）
```

### 1.3 conda 环境

```
# 用 conda env export 导出当前 mtmc 环境的 spec
conda activate mtmc
conda env export --no-builds > mtmc_env_spec.yml
# 拷贝 mtmc_env_spec.yml 到目标机
```

---

## 2. 目标机环境搭建

### 2.1 Python 环境（搜索 + FlashInfer 评测）

```bash
# 从 spec 创建
conda env create -f mtmc_env_spec.yml -n mtmc

# 激活
conda activate mtmc

# 验证关键版本
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
# 预期：2.13.0+cu130  NVIDIA H100/H800
python -c "import triton; print(triton.__version__)"   # 3.7.1
python -c "import flashinfer_bench; print('OK')"       # 0.1.2
```

### 2.2 SOL-ExecBench 官方 venv（SOL 题评测用）

```bash
# clone 官方仓库（或从 ziming 目录拷贝）
git clone <SOL-ExecBench repo> ~/SOL-ExecBench

# 建官方 venv（Python 3.12）
cd ~/SOL-ExecBench
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .            # 安装 sol-execbench 1.0.2
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install triton==3.5.0

# 验证
.venv/bin/python -c "import sol_execbench; print('OK')"
ls .venv/bin/sol-execbench  # CLI 可执行文件存在
```

### 2.3 LLM API

```bash
# 目标机 bashrc 里已有（用户说了），确认：
source ~/.bashrc
echo $LLM_API_KEY | head -c 10   # 应输出 key 前缀

# 如需自定义（可选）：
# 编辑 llm.env，设 K_SEARCH_KEY=<目标机的key>
```

### 2.4 路径适配

```bash
# 需要改的路径（在以下文件中全局替换 /data1/workspace/weihongren → <目标机你的目录>）：
#   ksearch-run.sh（WS 变量）
#   scripts/ksearch_campaign.sh（WS 变量）
#   scripts/ksearch_pool_campaign.sh（WS 变量）
#   scripts/ksearch_final_eval.py（WS / SOL_ROOT 变量）
#   scripts/ksearch_formal_report.py（WS / FI / SOL1 / SOL2 变量）
#   scripts/ksearch_formal_archive.py（WS 变量）
#   K-Search/k_search/tasks/sol_execbench_task.py（DEFAULT_SOL_ROOT）
#   K-Search/k_search/tasks/flashinfer_bench_task.py（REF_LATENCY_CACHE_ROOT）
#   ksearch_patchshim/sitecustomize.py（PYTHONPATH 路径）
#   evaluators/ 脚本无硬编码路径（从命令行参数取）
```

---

## 3. 实验配置（最终版 v0.5 + 强化守卫）

### 3.1 搜索参数

```bash
# 预算
KSEARCH_MAX_ROUNDS=100              # 100 轮 = 20 节点 × 5 attempt
KSEARCH_WM_MAX_ACTION_NODES=20
KSEARCH_WM_MAX_ATTEMPTS_PER_NODE=5
KSEARCH_WM_STAGNATION_WINDOW=5      # 默认值

# 强化守卫（必须加）
KSEARCH_STRICT_NO_LIB=1             # MUST NOT + BANNED 清单

# GPU 池（多题并发时）
KSEARCH_GPU_POOL="0,1,2,3"         # 目标机的空卡编号

# seed
--seed 0
```

### 3.2 反馈评测参数

```bash
# FlashInfer 题：ksearch-run.sh 自动传
--warmup-runs 3 --iterations 100 --num-trials 5

# SOL 题：ksearch-run.sh 自动传
--warmup-runs 10 --iterations 100 --num-trials 1

# 反馈 workload：默认抽样 5 个（seed 固定）
# 如需全量反馈（消融用）：
--feedback-workloads <uuid1> <uuid2> ... <uuidN>
```

### 3.3 最终评测参数

```bash
# FlashInfer 题（trials=1，v0.5 口径）
python scripts/ksearch_final_eval.py --task <def名> --run-dir <run目录> \
  --iterations 100 --trials 1
# 底层自动调：evaluators/evaluate.py --warmup 3 --iters 100 --trials 1

# SOL 题
python scripts/ksearch_final_eval.py --task L1/<题名> --run-dir <run目录> \
  --iterations 100 --timeout 7200
# 底层自动调：evaluators/evaluate_sol.py --rerun --iterations 100
# 注意：慢 ref 题（如有）需加大 --timeout
```

---

## 4. 全流程执行命令

### 4.1 前置检查（一次性）

```bash
cd <你的目录>

# 1) 环境检查
source activate-ksearch.sh         # 或手动 conda activate mtmc + PYTHONPATH
python -c "import torch; print(torch.cuda.is_available())"

# 2) SOL venv 检查
ls <SOL根>/.venv/bin/sol-execbench

# 3) API key 检查
source ~/.bashrc && echo "key: ${LLM_API_KEY:0:10}..."

# 4) 判官预检（重要！逐题跑 reference-as-candidate 探针）
python -c "
from k_search.tasks.sol_execbench_task import SolExecBenchTask
from k_search.tasks.task_base import Solution, BuildSpec, SourceFile, SupportedLanguages
t = SolExecBenchTask.from_cli_args(sol_root='<SOL根>',
    definition_name='L1/<题名>', warmup_runs=2, iterations=5,
    feedback_workloads=None, num_feedback_workloads=1, artifacts_dir=None, seed=0)
sol = Solution(name='probe', definition=t.name, author='p',
    spec=BuildSpec(language=SupportedLanguages.TRITON, target_hardware='H100',
                   entry_point='main.py::run', dependencies=[], destination_passing_style=False),
    sources=[SourceFile(path='main.py', content=t._definition['reference'])])
er = t.run_benchmark(solution=sol, round_num=0)
print(f'<题名>: {er.status} ref={er.reference_latency_ms:.1f}ms')
# 应输出 passed；如 INCORRECT_NUMERICAL 说明判官有 bug（-inf 问题），需检查补丁
"

# 5) -inf 补丁验证（FlashInfer 题才需要）
python -c "
from k_search.tasks.flashinfer_bench_task import patch_feedback_checker_inf_sentinels
patch_feedback_checker_inf_sentinels()
from flashinfer_bench.bench.evaluators.default import DefaultEvaluator
print('patched:', getattr(DefaultEvaluator.check_correctness, '_ksearch_inf_patched', False))
# 应输出 True
"
```

### 4.2 FlashInfer 10 题搜索

```bash
# 批量（auto 探卡）
KSEARCH_STRICT_NO_LIB=1 \
KSEARCH_WM_MAX_ACTION_NODES=20 KSEARCH_WM_MAX_ATTEMPTS_PER_NODE=5 \
bash scripts/ksearch_campaign.sh \
  --gpus auto \
  --tasks dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 \
          gdn_decode_qk4_v8_d128_k_last gdn_prefill_qk4_v8_d128_k_last \
          gemm_n4096_k4096 gqa_paged_decode_h32_kv8_d128_ps1 \
          gqa_paged_prefill_causal_h32_kv8_d128_ps1 gqa_ragged_prefill_causal_h32_kv8_d128 \
          mla_paged_decode_h16_ckv512_kpe64_ps1 mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 \
          rmsnorm_h4096 \
  --rounds 100 --seed 0 --wm --tag formal_h100
```

### 4.3 SOL L1 10 题搜索（池化并发）

```bash
KSEARCH_STRICT_NO_LIB=1 \
KSEARCH_WM_MAX_ACTION_NODES=20 KSEARCH_WM_MAX_ATTEMPTS_PER_NODE=5 \
bash scripts/ksearch_pool_campaign.sh \
  --pool "0,1" --concurrency 4 --tag formal_sol_h100 \
  --tasks L1/002_vae_conv3x3_groupnorm_silu_residual_fused \
          L1/005_conv_gated_projection_with_causal_conv \
          L1/007_hyena_fft_size_padding_rfft \
          L1/008_expert_output_weighted_index_add_accumulation \
          L1/018_fused_rope_with_qk_norm_and_kv_cache_update \
          L1/020_vision_patch_merger_spatial_shuffle_mlp \
          L1/053_gaussian_topk_sparse_activation \
          L1/058_moe_expert_token_radix_sort_with_prefix_sum \
          L1/070_mamba2_fused_intra_chunk_diagonal_computation \
          L1/092_gqa_attention_with_qk_norm
```

### 4.4 SOL L2 10 题（可选，第二批）

```bash
KSEARCH_STRICT_NO_LIB=1 \
bash scripts/ksearch_pool_campaign.sh \
  --pool "0,1" --concurrency 4 --tag formal_solL2_h100 \
  --tasks L2/012_moe_expert_batched_execution_with_capacity_factor \
          L2/015_audio_sinusoidal_position_embedding_with_conv_projection \
          L2/030_flux_concatenated_sequence_processing_with_split \
          L2/036_convnextv2_layer_with_nhwc_persistence_backward \
          L2/040_altup_predict_correction_cycle_backward \
          L2/043_mamba_chunk_scan_with_segsum \
          L2/049_group_limited_topk_routing \
          L2/051_seqlen-finetuned-reconstructed_hyena_complete_forward_block \
          L2/057_residual_coupling_flow_block \
          L2/080_moe_complete_layer_with_shared_expert_backward
```

### 4.5 全量评测（搜索完成后逐题跑）

```bash
# FlashInfer 题
for t in dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 \
         gdn_decode_qk4_v8_d128_k_last gdn_prefill_qk4_v8_d128_k_last \
         gemm_n4096_k4096 gqa_paged_decode_h32_kv8_d128_ps1 \
         gqa_paged_prefill_causal_h32_kv8_d128_ps1 gqa_ragged_prefill_causal_h32_kv8_d128 \
         mla_paged_decode_h16_ckv512_kpe64_ps1 mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 \
         rmsnorm_h4096; do
  CUDA_VISIBLE_DEVICES=<空卡> python scripts/ksearch_final_eval.py \
    --task $t --run-dir baseline/ksearch/experiments/formal_h100/$t/run_seed0 \
    --iterations 100 --trials 1
done

# SOL 题
for t in L1/002_vae_conv3x3_groupnorm_silu_residual_fused L1/005_conv_gated_projection_with_causal_conv \
         L1/007_hyena_fft_size_padding_rfft L1/008_expert_output_weighted_index_add_accumulation \
         L1/018_fused_rope_with_qk_norm_and_kv_cache_update L1/020_vision_patch_merger_spatial_shuffle_mlp \
         L1/053_gaussian_topk_sparse_activation L1/058_moe_expert_token_radix_sort_with_prefix_sum \
         L1/070_mamba2_fused_intra_chunk_diagonal_computation L1/092_gqa_attention_with_qk_norm; do
  CUDA_VISIBLE_DEVICES=<空卡> python scripts/ksearch_final_eval.py \
    --task $t --run-dir baseline/ksearch-sol-execbench/experiments/formal_sol_h100/${t##*/}/run_seed0 \
    --iterations 100 --timeout 7200
done
```

### 4.6 报告与归档

```bash
# 30 题报告
python scripts/ksearch_merged_report.py

# 归档（代码版本指纹 + 配置快照 + token + 评测）
python scripts/ksearch_formal_archive.py --tag formal_h100 --with-eval
```

---

## 5. 必读注意事项（按踩坑频率排序）

| # | 坑 | 对策 |
|---|---|---|
| 1 | **-inf 判官**：reference 输出含 -inf（空 causal 前缀 LSE）→ 正确解被误杀 | `ksearch_patchshim/` 目录勿删；启动前跑 §4.1 的补丁验证 |
| 2 | **杀进程用宽 grep**：`grep "sol"` 会误杀 `--solution` 进程 | 只按精确 PID 杀 |
| 3 | **GPU 独占**：评测必须空卡，其他租户共卡会污染计时 | 启动前+启动后各验证一次 `nvidia-smi -i N --query-compute-apps` |
| 4 | **tee 目录竞争**：`tee -a file` 时目录不存在会断管 | 先 `mkdir -p` + `touch` |
| 5 | **相对路径**：脱管启动（setsid/nohup）时 cwd 不确定 | 全部用绝对路径 |
| 6 | **SOL 慢 ref 超时**：ref 为 Python 循环的题全量评测 4h+ | 逐题预检 ref 延迟；`--timeout` 按需放大；跳过此类题或加分片 |
| 7 | **done 标记**：手动启动的搜索无 DONE → 评测脚本跳过 | 用 campaign/pool_campaign 启动；或手动 `touch DONE` |
| 8 | **守卫效果**：默认守卫挡不住 F.linear/conv2d；强化守卫能挡 F.linear 但挡不住 conv2d/FFT | 必须设 `KSEARCH_STRICT_NO_LIB=1`；conv/FFT 题的守卫无效是 LLM 能力边界 |
| 9 | **ref 缓存**：搜索反馈侧 ref 延迟可缓存（不影响最终评测）；最终评测不用缓存 | 协议要求同进程成对计时 |

---

## 6. H 卡特有的适配点

| 项 | A800 (当前) | H100/H800 (目标) | 需改什么 |
|---|---|---|---|
| target-gpu | A800 | H100 | `ksearch-run.sh` 里 `--target-gpu A800` → `H100`（影响 prompt 措辞） |
| sm 架构 | sm_80 | sm_90 | fp8e4nv 编译墙消失（dsa_topk 可跑）；Triton 自动适配 |
| ref 缓存 key | 含硬件名 | 不同硬件 → 不同 key | 无需操作，首轮自动新建缓存 |
| 预期加速比 | 本报告数字 | 可能更高（HBM3 带宽更大） | 正常，只需记录 |
| evaluate.py `--device` | cuda:0 | 同 | CUDA_VISIBLE_DEVICES 隔离即可 |

---

## 7. 快速冒烟验证（目标机上 10 分钟确认一切正常）

```bash
# 单题 2 轮
CUDA_VISIBLE_DEVICES=0 KSEARCH_STRICT_NO_LIB=1 KSEARCH_RUN_TAG=smoke \
  KSEARCH_MAX_ROUNDS=2 bash ksearch-run.sh rmsnorm_h4096 0 --wm

# 检查产物
ls baseline/ksearch/experiments/smoke/rmsnorm_h4096/run_seed0/
# 应有: usage.jsonl  stdout_*.log  ksearch-artifacts/

# 全量评测
CUDA_VISIBLE_DEVICES=0 python scripts/ksearch_final_eval.py \
  --task rmsnorm_h4096 \
  --run-dir baseline/ksearch/experiments/smoke/rmsnorm_h4096/run_seed0
# 应输出: valid=True pass=14/14 geo=...x
```

如冒烟通过，按 §4.2–4.6 展开全量实验。
