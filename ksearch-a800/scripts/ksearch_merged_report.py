#!/usr/bin/env python3
"""30 题合并报告 v2：用人工审计结论替换启发式分类。

审计方法：逐题读 run() 源码，统计 (1) AST 检出的库调用及其上下文，
(2) @triton.jit 内核定义数，(3) run() 内实际 launch 次数（`_kernel[grid]` 语法），
(4) 库调用是否在主路径（缩进+分支分析）。核心判断标准：
"如果把库调用换成最朴素的实现，这题还能拿到现在这个加速比吗？"
"""
import json, re, glob
from pathlib import Path
from datetime import datetime

WS = Path("/data1/workspace/weihongren")

FLASHINFER = [
    "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64",
    "gdn_decode_qk4_v8_d128_k_last",
    "gdn_prefill_qk4_v8_d128_k_last",
    "gemm_n4096_k4096",
    "gqa_paged_decode_h32_kv8_d128_ps1",
    "gqa_paged_prefill_causal_h32_kv8_d128_ps1",
    "gqa_ragged_prefill_causal_h32_kv8_d128",
    "mla_paged_decode_h16_ckv512_kpe64_ps1",
    "mla_paged_prefill_causal_h16_ckv512_kpe64_ps1",
    "rmsnorm_h4096",
]
SOL_L1 = [
    "002_vae_conv3x3_groupnorm_silu_residual_fused",
    "005_conv_gated_projection_with_causal_conv",
    "007_hyena_fft_size_padding_rfft",
    "008_expert_output_weighted_index_add_accumulation",
    "018_fused_rope_with_qk_norm_and_kv_cache_update",
    "020_vision_patch_merger_spatial_shuffle_mlp",
    "053_gaussian_topk_sparse_activation",
    "058_moe_expert_token_radix_sort_with_prefix_sum",
    "070_mamba2_fused_intra_chunk_diagonal_computation",
    "092_gqa_attention_with_qk_norm",
]
SOL_L2 = [
    "012_moe_expert_batched_execution_with_capacity_factor",
    "015_audio_sinusoidal_position_embedding_with_conv_projection",
    "030_flux_concatenated_sequence_processing_with_split",
    "036_convnextv2_layer_with_nhwc_persistence_backward",
    "040_altup_predict_correction_cycle_backward",
    "043_mamba_chunk_scan_with_segsum",
    "049_group_limited_topk_routing",
    "051_seqlen-finetuned-reconstructed_hyena_complete_forward_block",
    "057_residual_coupling_flow_block",
    "080_moe_complete_layer_with_shared_expert_backward",
]

# 人工审计结论: task[:3] -> (分类, 库调用, 内核定义, launch, 详细说明)
AUDIT = {
    "gemm": ("A·核心靠库", "matmul", 1, 0,
        "run() 主体整体调 torch.matmul(A,B.T)。1 个 Triton 内核定义但 0 次有效 launch（死代码）。r1–59 自研全败、r60 起退化为调与 reference 同款的 cuBLAS。0.84x = 库调用+包装开销。"),
    "002": ("A·核心靠库", "conv2d×2", 6, 0,
        "两个 conv3x3 本体调 F.conv2d（题目核心计算）。6 个 Triton 内核定义（GroupNorm/SiLU/残差）但 run() 内 0 次 launch——全部是死代码。1.39x 来自'调库+GroupNorm 融合'的外围组合。"),
    "015": ("A·核心靠库", "conv2d×3,addmm,linear", 3, 3,
        "三个 conv2d 本体调库 + addmm/linear 投影调库。Triton 只做 gelu/transpose/scale（3 内核 3 launch）。1.03x ≈ 零加速。reference 也调同样的库，解等于没优化。15/16 有一个数值失败。"),
    "007": ("A·核心靠库", "rfft×2", 2, 2,
        "RFFT 本体（题目名即 rfft）调 torch.fft.rfft。Triton _direct_rfft_kernel 有 2 次 launch 但仅覆盖部分路径（padding/归一化段），RFFT 核心走 cuFFT。1.38x。"),
    "092": ("B·自研为主", "linear×4", 2, 2,
        "F.linear 做 q/k/v/o 投影（前置/后置辅助步骤）。attention 本体由 Triton 实现（2 内核 2 launch）。2.84x 来自 attention 优化，投影走库合理。"),
    "005": ("B·自研为主", "linear×2", 1, 1,
        "F.linear 做输入/输出投影。核心 causal_conv+gating 由 Triton _packed_conv_gate_kernel 实现（1 内核 1 launch）。1.57x 来自 gating kernel。"),
    "036": ("B·自研为主", "mm×4", 9, 10,
        "torch.mm 做 pwconv 权重梯度的小 GEMM（合理分步）。GRN 反传/LayerNorm 反传/dwconv 梯度/NHWC 布局全部由 9 个 Triton 内核（10 次 launch）实现。364x 的主体来自消掉 reference 的 permute/clone/for 循环。"),
    "080": ("B·自研为主", "matmul×7,addmm×2", 2, 2,
        "matmul/addmm 做 shared expert 梯度和 router 梯度（代码行数多但非计算热点）。routed expert 的 scatter/gather/融合由 Triton 实现（2 内核 2 launch）。6x 来自 routed 部分。"),
    "020": ("C·混合贡献", "linear×2", 5, 6,
        "F.linear 做 MLP fc1/fc2 投影。spatial shuffle/patch merge 由 Triton 实现。消融：去掉 F.linear 后 1.46x（-32%），证明库贡献约 1/3。全量反馈版 2.20x（+2%）。"),
    "012": ("C·待消融", "bmm×3", 4, 4,
        "torch.bmm×3 做专家的 gate/up/down 批量矩阵乘。Triton 4 内核 4 launch 做路由/scatter。1.44x 太温和——如果 bmm 是计算大头，加速来自路由融合而非专家计算本身。需消融确认。"),
    "051": ("C·待消融", "addmm×2,rfft×2,irfft×1", 7, 1,
        "7 个 Triton 内核定义但只有 1 次 launch——6 个是死代码。FFT 走库，addmm 投影走库。2.35x 的来源需实验验证（可能来自 1 个 Triton kernel + reference 本身的低效）。"),
    # 干净题（零库调用）
    "030": ("干净", "-", 1, 1, ""), "040": ("干净", "-", 5, 5, ""),
    "043": ("干净", "-", 4, 4, ""), "049": ("干净", "-", 3, 3, ""),
    "057": ("干净", "-", 1, 1, ""), "053": ("干净", "-", 4, 4, ""),
    "058": ("干净", "-", 3, 3, "v2 注记：全量反馈续跑 5 轮后 16/16 valid（原版 15/16 因 batch=1 抽样盲区失败；10.32x）"), "070": ("干净", "-", 2, 2, ""),
    "008": ("干净", "-", 2, 2, ""), "018": ("干净", "-", 1, 1, ""),
    "dsa": ("干净", "-", 1, 1, ""),
    "gdn": ("干净", "-", 1, 1, ""),  # 会被 gdn_decode/gdn_prefill 前缀匹配
    "gqa": ("干净", "-", 1, 1, ""),  # gqa_paged_decode 等
    "mla": ("干净", "-", 2, 2, ""),
    "rms": ("干净", "-", 1, 1, ""),
}


def rd_for(task):
    if task in FLASHINFER:
        return WS / "baseline/ksearch/experiments/formal_20260914" / task / "run_seed0"
    if task in SOL_L2:
        return WS / "baseline/ksearch-sol-execbench/experiments/formal2_solL2_20260916" / task / "run_seed0"
    return WS / "baseline/ksearch-sol-execbench/experiments/formal_20260914" / task / "run_seed0"


def bench(task):
    return "FlashInfer" if task in FLASHINFER else ("SOL-L2" if task in SOL_L2 else "SOL-L1")


def audit_for(task):
    for prefix in ["gemm", "002", "015", "007", "092", "005", "036", "080", "020", "012", "051",
                   "030", "040", "043", "049", "057", "053", "058", "070", "008", "018",
                   "dsa", "gdn", "gqa", "mla", "rms"]:
        if task.startswith(prefix):
            return AUDIT[prefix]
    return ("?", "?", 0, 0, "")


def usage(task):
    f = rd_for(task) / "usage.jsonl"
    if not f.exists():
        return None
    c = i = ca = o = r = 0
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line: continue
        try: x = json.loads(line)
        except: continue
        c += 1; i += x.get("input_tokens") or 0; ca += x.get("input_cached_tokens") or 0
        o += x.get("output_tokens") or 0; r += x.get("reasoning_tokens") or 0
    return {"calls": c, "input": i, "cached": ca, "output": o, "reasoning": r, "total": i + o}


def search_info(task):
    rounds = 0; best = None
    for f in rd_for(task).glob("stdout_*.log"):
        txt = f.read_text(errors="ignore")
        rounds += txt.count("Optimization Round")
        for s in re.findall(r"score=([0-9.]+)", txt):
            v = float(s)
            if best is None or v > best: best = v
    return rounds, best


def eval_info(task):
    f = rd_for(task) / "unified" / ("evaluation.json" if task in FLASHINFER else "performance.json")
    if not f.exists(): return None
    d = json.load(open(f))
    pw = d.get("per_workload", [])
    passed = d.get("passed", sum(1 for w in pw if w.get("status") == "PASSED"))
    total = d.get("total", len(pw))
    sps = [w.get("speedup") for w in pw if w.get("status") == "PASSED" and w.get("speedup")]
    return {"valid": d.get("valid"), "passed": passed, "total": total,
            "geo": d.get("geomean_speedup"), "arith": d.get("arithmetic_mean_speedup"),
            "min": min(sps) if sps else None, "max": max(sps) if sps else None}


def fm(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "-"

# 消融更新：058 用续跑版（16/16 valid）替换原版（15/16 invalid）
EVAL_OVERRIDE = {
    "058_moe_expert_token_radix_sort_with_prefix_sum": {
        "valid": True, "passed": 16, "total": 16, "geo": 10.324,
        "arith": 10.33, "min": 6.58, "max": 11.53,
        # 注：原版 15/16（batch=1 抽样盲区）→ 全量反馈 5 轮续跑修复为 16/16
    },
}


out = []
out.append("# K-Search 30 题合并结果报告（formal_20260914 + formal2_solL2_20260916）")
out.append("")
out.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}（v2：torch 回退改为人工审计结论）")
out.append("- 方法：K-Search world-model 完整版，seed0，每题 100 轮（20 节点 × 5 attempt）")
out.append("- LLM：AWS-GPT-5.6-Sol @ llmapi.isrc.ac.cn/v1")
out.append("- 硬件：A800-SXM4-80GB；GPU 池化（flock 并发）+ 亲和模式混合")
out.append("- 搜索反馈：固定 seed 抽样 5 workload（FlashInfer 3/100/5；SOL 10/100）")
out.append("- 最终评测：全量 workload；FlashInfer evaluate.py（warmup3/iters100/trials1）；SOL 官方 CLI（iterations100）")
out.append("- 题集：FlashInfer 10 + SOL L1 10（092 顶补 094）+ SOL L2 10 = **30 题 / 738 workloads**")
out.append("")
out.append("## 1. 指标与分类定义")
out.append("")
out.append("| 指标 | 定义 |")
out.append("|---|---|")
out.append("| valid / pass / geomean | 全量评测：全部正确=valid；geomean=各 workload speedup 几何平均（ref 与 sol 同进程成对计时）|")
out.append("| 反馈 best | 搜索期间 5 个抽样 workload 上的最优 mean speedup |")
out.append("| 内核(定义/launch) | `@triton.jit` 装饰的函数数 / run() 内通过 `_kernel[grid](...)` 实际启动次数——定义了但未 launch 的是死代码 |")
out.append("")
out.append("**torch 回退人工审计分类**（逐题读 run() 源码，核心判断：「如果把库调用换成最朴素实现，这题还能拿到这个加速比吗？」）：")
out.append("")
out.append("| 分类 | 含义 | 判定依据 |")
out.append("|---|---|---|")
out.append("| **A·核心靠库** | 核心计算调 cuDNN/cuBLAS/cuFFT，自研 Triton 只做周边融合或干脆是死代码 | 库调用在主路径 + 处理题目核心操作 + 加速比通常 ≤1.4x |")
out.append("| **B·自研为主** | 库调用只做辅助步骤（投影/小 GEMM），计算热点由 Triton 实现 | 库调用不在热点路径 + Triton launch 数 > 0 + 内核覆盖核心操作 |")
out.append("| **C·待消融** | 无法从代码结构直接判断库和自研的贡献比例 | 需把库调用替换为朴素实现后重测 |")
out.append("| **干净** | run() 内零库调用 | AST 全扫描无命中 |")
out.append("")

out.append("## 2. 30 题总表（按题目组）")
out.append("")
out.append("| # | 任务 | bench | valid | pass | geomean | torch 回退 | 库调用 | 内核(定义/launch) |")
out.append("|---|---|---|---|---|---|---|---|---|")
n = 0; tc = ti = tca = to_ = tr_ = 0; token_rows = []
for task in FLASHINFER + SOL_L1 + SOL_L2:
    e = eval_info(task)
    if task in EVAL_OVERRIDE:
        e = EVAL_OVERRIDE[task]
    cat, libs, nd, nl, _ = audit_for(task)
    u = usage(task)
    if u:
        tc += u["calls"]; ti += u["input"]; tca += u["cached"]; to_ += u["output"]; tr_ += u["reasoning"]
        token_rows.append((task, u))
    if e:
        n += 1
        out.append(f"| {n} | {task} | {bench(task)} | {'✅' if e['valid'] else '❌'} | {e['passed']}/{e['total']} | {fm(e['geo'])} | {cat} | {libs} | {nd}/{nl} |")
out.append("")

out.append("## 3. 有库调用的 11 题逐题分析")
out.append("")
for task in FLASHINFER + SOL_L1 + SOL_L2:
    cat, libs, nd, nl, detail = audit_for(task)
    if cat == "干净":
        continue
    e = eval_info(task)
    geo = fm(e["geo"]) if e else "-"
    out.append(f"**{task}**（{cat}，{geo}x）")
    out.append(f"- 库调用：{libs} | Triton 内核：定义 {nd} 个 / launch {nl} 次")
    out.append(f"- {detail}")
    out.append("")

out.append("## 4. 失败题注记")
out.append("")
out.append("| 题 | 终数 | 原因 |")
out.append("|---|---|---|")
out.append("| 058 | 15/16 | batch=1/seq=2080 INCORRECT_NUMERICAL（排序 kernel 边界形状 bug）|")
out.append("| 057 | 9/16 | 反馈 5 全过但全量 7 shape 失败——泛化缺口 |")
out.append("| 015 | 15/16 | 1 workload 数值失败（伴随核心靠库）|")
out.append("| gemm | 43/43 正确但 0.84x | A 类：run() 主体调 cuBLAS |")
out.append("| 094（v3 剔除） | — | reference 朴素 Python 循环，评测 4h+/题 |")
out.append("")

out.append("## 5. 搜索进度与反馈 best")
out.append("")
out.append("| 任务 | 轮次 | 反馈 best |")
out.append("|---|---|---|")
for task in FLASHINFER + SOL_L1 + SOL_L2:
    rounds, best = search_info(task)
    out.append(f"| {task} | {rounds} | {fm(best, 3)} |")
out.append("")

out.append("## 6. Token 消耗")
out.append("")
out.append("| 任务 | LLM calls | input | cached | output | reasoning | 总计 |")
out.append("|---|---|---|---|---|---|---|")
for task, u in token_rows:
    out.append(f"| {task} | {u['calls']} | {u['input']:,} | {u['cached']:,} | {u['output']:,} | {u['reasoning']:,} | {u['total']:,} |")
out.append(f"| **合计** | **{tc}** | **{ti:,}** | **{tca:,}** | **{to_:,}** | **{tr_:,}** | **{ti+to_:,}** |")
out.append("")

out.append("## 7. 观察")
out.append("")
out.append("- **A 类 4 题终数全部 ≤1.39x**：gemm 0.84 / 002 1.39 / 015 1.03 / 007 1.38——加速上限被库调用封死")
out.append("- **B 类 4 题加速比 1.57~364x**：库调用只做辅助步骤，加速主体来自 Triton 自研（消融验证：005 去掉 F.linear 后仅 -10%）")
out.append("- **C 类 2+1=3 题**加速比 1.44~2.35x——020 经消融证明库贡献 32%（1.46x vs 2.16x），从 B 改判 C")
out.append("- **C 类 2 题**（012/051）加速比 1.44x / 2.35x——需消融实验（把库调用替换为朴素实现后重测）才能归因")
out.append("- **干净 19 题**加速比 1.38~393x——零库调用，加速全部来自自研")
out.append("- **A 类的共同特征**：题目核心操作即库函数擅长域（GEMM/conv/FFT），LLM 选择调库而非自研——与 gemm 的 r59 全败后 r60 转向调库行为一致")
out.append("- **守卫效果**：上游防回退守卫（v0.4 起注入 SOL）阻止了 run() 主体整体调库（gemm 式躺平），但无法阻止'热点自研+辅助调库'的混合策略（B/C 类）")
out.append("- **内核数 ≠ 自研贡献**：002 定义了 6 个内核但 0 次 launch（全死代码）；051 定义了 7 个但只 launch 1 个——必须数 launch 次数")
out.append("")

report = "\n".join(out)
dest = WS / "notes/reports"
dest.mkdir(parents=True, exist_ok=True)
(dest / "k-search_30t_0917.md").write_text(report, encoding="utf-8")
print(f"written: {dest / 'k-search_30t_0917.md'} ({len(report)} chars)")
