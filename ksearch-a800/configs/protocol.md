# MTMC Baseline 公平实验协议（DRAFT v0.3，待学长冻结）

对应论文 v4 大纲 §5.1/§6；冻结前所有数字只算 smoke，不进论文。
v0.3 更新：K-Search seed 真正贯穿 feedback 抽样；SOL-ExecBench 使用官方独立评测接口。

## 1. 硬件与环境

| 项 | 固定值 |
|---|---|
| GPU | 8 × NVIDIA A800-SXM4-80GB（sm_80），驱动 580.105.08 |
| nvcc | CUDA 12.4（/usr/local/cuda-12.4） |
| 统一评测环境 | ziming conda env `mtmc`：torch 2.13.0+cu130 / triton 3.7.1 / flashinfer-bench 0.1.2 / openai 3.3.1（`activate-eval.sh`） |
| K-Search 生成/反馈环境 | ziming `mtmc`：Python 3.11 / torch 2.13.0+cu130 / triton 3.7.1（与 FlashInfer 最终评测同环境） |
| DRTriton 生成环境 | DRTriton/venv：python 3.10 + vLLM（Dr.Kernel-8B bf16，单卡） |
| SOL-ExecBench 评测环境 | SOL 官方 `.venv`：Python 3.12 / torch 2.9.0+cu130 / triton 3.5.0 / sol-execbench 1.0.2 |

- 生成环境允许各方法用官方依赖。FlashInfer-test 最终评测走 `mtmc` 统一 evaluator；
  SOL-ExecBench 题目必须走 SOL 官方 CLI/schema，二者结果分开报告，不能跨 harness 混合比较。
- 任务集使用 ziming 原版的逐字节副本 `/data1/workspace/weihongren/dataset/flashinfer-test`（防写坏原目录；K-Search 直接加载已验证 11/11）。
- GPU benchmark 一律在宿主机终端/有 NVIDIA 设备节点的环境执行（沙箱内 /dev 被覆盖）。
- 若 K-Search（triton 3.5 口径）生成的 kernel 在 triton 3.7.1 下编译失败：记录失败率；连续大面积失败时上报学长决定是否另固定评测版本（当前默认不另设）。

## 2. 任务集（范围已确认，协议参数待冻结）

机器可读名单见 `experiment_manifest.json`。

- FlashInfer-test：**v2 范围（2026-09-14 换题）**：排除 `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`
  （sm_80 上 Triton fp8e4nv 编译墙，smoke 0/5），纳入 `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`；
  共 10 definitions / 426 workloads。
- SOL-ExecBench：指定的 10 个 L1 problems / 154 workloads：002、005、007、008、018、020、
  053、058、070、092（v3：094 剔除——reference 为朴素 Python 逐步循环，单次评测 4h+ 且
  speedup 是对朴素实现的比值；092_gqa_attention_with_qk_norm 顶替，完整路径见 manifest）。
- 两个选定子集没有重叠，共 20 definitions / 580 workloads。

- FlashInfer definition/workload 以 UUID 为准，任何一侧改动都需重新冻结。
- SOL 中含 custom inputs 的题必须使用 SOL 官方输入生成与 evaluator，不得转为普通独立随机输入。
- SOL 的 search seed 为 0/1/2，控制反馈 workload 与搜索随机性；正式 evaluator 的输入生成 seed
  对所有方法和候选固定为官方默认 200，避免不同搜索重复在不同输入张量上比较。

## 3. 评测参数（冻结）

- **v0.5（2026-09-15 用户确认）**：最终统一评测 warmup 3、iterations 100、**trials 1**（原 v0.3 冻结为 trials 5；FlashInfer 侧 evaluate.py 传 `--trials 1`，SOL 侧官方 CLI 本身即单次校验+计时）。搜索反馈仍为 trials 5（与已完成 18 题一致，口径在反馈层不影响最终数字）
- 正确性容差：按各 definition 内置容差；reference 为 flashinfer-bench 官方 reference
- speedup = reference_latency / solution_latency（同机同进程成对计时）

## 4. 预算（四重上限；数值待学长冻结，当前建议值）

| 预算项 | 正式值（2026-09-14 用户确认） |
|---|---|
| 总 LLM token | ≤ 2,000,000 / task / seed（20 题推算中位 ~1.43M/题，总计 ~29M 已确认合理） |
| 候选数 | ≤ 100 / task / seed（= 20 action nodes × 5 attempts，正式批次 v2 预算） |
| benchmark 调用次数 | ≤ 400 / task / seed（实际 ~101） |
| wall-clock search | 快题 ≤ 5h；5 道慢题（gdn_prefill/mla_paged_prefill/gqa_paged_prefill/L1-094/gqa_paged_decode）接受超 12h，已排在批次最后 |

- 当前阶段每题只运行 1 次独立搜索（seed 0）。SOL 原生后端接通后先做 5-round pilot；正式单次
  搜索使用 20 rounds。seed 1/2 暂缓，待 seed0 成本、稳定性和论文统计需求明确后再决定是否补跑。
- K-Search 的 seed 固定 Python/NumPy/Torch 本地 RNG、feedback workload 抽样及运行元数据；
  远端 LLM 未提供可验证的确定性保证，因此 seed 表示独立重复而非逐 token 重放。

## 5. LLM / 模型配置（冻结）

| 方法 | 生成模型 | 配置 |
|---|---|---|
| K-Search | AWS-GPT-5.6-Sol @ llmapi.isrc.ac.cn/v1（与 MTMC 相同） | temperature 用 API 默认（代码未设置）；language=triton；target-gpu=A800；**v0.4（2026-09-14）：搜索反馈计时参数与统一评测器对齐——FlashInfer 题 warmup 3 / iterations 100 / trials 5（evaluate.py 口径），SOL 题 warmup 10（SOL 官方默认）/ iterations 100（evaluate_sol.py 口径）；与最终统一评测的唯一差别是 workload 数（反馈=固定 seed 抽样 5 个，最终=全量）**；SOL 侧 reference 延迟首轮缓存、后续轮复用（只影响反馈速度up计算，不改变计时口径）；`--no-save-results`；usage 由 ksearch-token-run.py 逐调用记账（input/cached/output/reasoning 四类） |
| DRTriton | **本地 Dr.Kernel-8B**（/data1/hf_models/drkernel-8b，Qwen3-8B bf16） | vLLM greedy(首样本 temperature=0)；采样档 temperature=1.0 top_p=1.0；max_tokens 8192；prompt_style=original（smoke 时与 openai 对比后定） |

- DRTriton-7B 官方权重未发布（论文声明接收后放出，HF 无 checkpoint）。
- **论文标注：DRTriton official pipeline (Optimization-AI/DRTriton@64d9325) applied to Dr.Kernel-8B**——最终命名待学长确认。

## 6. DRTriton 支持矩阵（input adapter 实测，CPU 等价校验全过）

| 档位 | 支持算子 | 说明 |
|---|---|---|
| single generation | 11/11 | fused_operator 内逐字调用官方 reference；get_inputs 按 representative workload（最小规模）合成，满足全部 reference 约束断言 |
| N 次采样 | 11/11 | 同上，--rollout_n N（官方：第 1 个 greedy + N-1 个 temperature 采样） |
| test-time search | **2/11：rmsnorm_h4096（14 子问题）、gemm_n4096_k4096（1 子问题）** | 官方 TTS 要求 fused_operator 为扁平 tensor_N 赋值链；其余 9 算子 reference 含 for/if（GDN 循环、attention 逐 batch 循环、DSA topk 循环）无法扁平化 → **TTS 档 unsupported，如实记录**。rmsnorm/gemm 的 TTS 用 fp32 脚手架（官方 subproblem 机制仅支持 fp32），最终正确性仍以统一 evaluator 真实 dtype 判定 |

适配产物：`DRTriton/data/flashinfer_tasks.jsonl`（manifest）、`flashinfer_wrapped.jsonl`（single/N）、`flashinfer_flat.jsonl`（TTS）。

## 7. 版本指纹

| 方法 | 代码版本 | 模型 |
|---|---|---|
| K-Search | caoshiyi/K-Search@53c8fab9a5e8fab2c86610d24fbec5067f90e115 | AWS-GPT-5.6-Sol（API） |
| DRTriton 管线 | github.com/Optimization-AI/DRTriton@64d9325b2308535da1cda073ad47d3d61b40432b | Dr.Kernel-8B（本地权重，hkust-nlp/drkernel-8b，arXiv:2602.05885） |
| 统一评测 | ziming/MTMC-baseline/agent-generation（只读复用） | — |
| MTMC（对照） | mtmc-pipeline-api-v4 | 同 LLM endpoint |

## 8. 试跑顺序（GPU 释放后）

```bash
# 1) K-Search smoke（rmsnorm，先无 world-model 再完整版）
./ksearch-run.sh rmsnorm_h4096 0            # 普通 iterative
./ksearch-run.sh rmsnorm_h4096 0 --wm       # world-model 完整版
# 统一评测（mtmc env）
source activate-eval.sh
python scripts/unified_eval.py --task rmsnorm_h4096 \
  --ksearch-json <artifacts 里最新 solution json> \
  --run-dir baseline/ksearch/rmsnorm_h4096/run_seed0 --limit 3

# 2) DRTriton smoke
./drtriton-run.sh single 0                   # 11 任务 greedy 生成
python scripts/drtriton_collect.py --mode single --seed 0 --limit 3
./drtriton-run.sh sample 0                   # N=8 采样档
./drtriton-run.sh tts 0                      # rmsnorm+gemm TTS 链

# 3) gemm → gqa_paged_decode → gdn_decode → 全 11；每档 3 seeds；正式数字干净进程重测
```

注意：K-Search 与 Dr.Kernel-8B 推理分卡进行；Dr.Kernel-8B bf16 ≈16GB 单卡可容。

## 9. 指标（v4 §5.3/5.4）

per-workload latency/speedup；regime/overall geomean；worst-workload；correctness pass rate；总 token；token-to-hit；speedup/token；benchmark 次数；wall-clock；3 seed 均值/中位数/离散度。

## 待学长确认

- [ ] 预算四项具体数值（§4）；N 次采样的 N
- [ ] "DRTriton pipeline applied to Dr.Kernel-8B" 的论文命名；是否补发邮件向 DRTriton 作者（TAMU/Oracle）要权重
- [ ] TTS 档 9/11 unsupported 的论文表述
- [ ] K-Search 若无法按相同接口/预算复现 → 是否启用 v4 预案（只进 Related Work）
- [ ] 评测 triton 版本固定 3.7.1（mtmc env）
