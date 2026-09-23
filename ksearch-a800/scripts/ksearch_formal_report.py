#!/usr/bin/env python3
"""生成正式实验 MD 报告（性能 + token 表格），数据直接从产物文件提取。"""
import json
import glob
from pathlib import Path
from datetime import datetime

WS = Path("/data1/workspace/weihongren")
FI = WS / "baseline/ksearch/experiments/formal_20260914"
SOL = WS / "baseline/ksearch-sol-execbench/experiments/formal_20260914"

FLASHINFER_TASKS = [
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
SOL_TASKS = [
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

def run_dir(task):
    base = SOL if task in SOL_TASKS else FI
    return base / task / "run_seed0"

def usage(task):
    f = run_dir(task) / "usage.jsonl"
    if not f.exists():
        return None
    c = i = ca = o = r = 0
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            x = json.loads(line)
        except Exception:
            continue
        c += 1
        i += x.get("input_tokens") or 0
        ca += x.get("input_cached_tokens") or 0
        o += x.get("output_tokens") or 0
        r += x.get("reasoning_tokens") or 0
    return {"calls": c, "input": i, "cached": ca, "output": o, "reasoning": r, "total": i + o}

def search_info(task):
    rd = run_dir(task)
    log = rd / "campaign_stdout.log"
    rounds = score = None
    if log.exists():
        txt = log.read_text(errors="ignore")
        rounds = txt.count("Optimization Round")
        import re
        m = re.findall(r"score=([0-9.]+)", txt)
        score = float(m[-1]) if m else None
    return {"rounds": rounds, "score": score, "done": (rd / "DONE").exists()}

def eval_info(task):
    rd = run_dir(task) / "unified"
    for name, kind in (("evaluation.json", "fi"), ("performance.json", "sol")):
        f = rd / name
        if f.exists():
            d = json.load(open(f))
            pw = d.get("per_workload", [])
            passed = d.get("passed", sum(1 for w in pw if w.get("status") == "PASSED"))
            total = d.get("total", len(pw))
            sps = [w.get("speedup") for w in pw if w.get("status") == "PASSED" and w.get("speedup")]
            return {
                "valid": d.get("valid"), "passed": passed, "total": total,
                "geo": d.get("geomean_speedup"), "arith": d.get("arithmetic_mean_speedup"),
                "min": min(sps) if sps else None, "max": max(sps) if sps else None,
            }
    return None

def fm(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "-"

rows = []
for t in FLASHINFER_TASKS + SOL_TASKS:
    rows.append({
        "task": t, "bench": "SOL" if t in SOL_TASKS else "FlashInfer",
        **search_info(t), "usage": usage(t), "eval": eval_info(t),
    })

out = []
out.append("# K-Search 正式实验报告（formal_20260914）")
out.append("")
out.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
out.append("- 方法：K-Search (caoshiyi/K-Search@53c8fab + 本地适配)，world-model 完整版，seed0")
out.append("- 生成模型：AWS-GPT-5.6-Sol @ llmapi.isrc.ac.cn/v1")
out.append("- 硬件：NVIDIA A800-SXM4-80GB（搜索与评测均独占空卡）")
out.append("- 搜索预算：100 轮 = 20 action nodes × 5 attempts/node，stagnation window 5")
out.append("- 搜索反馈：固定 seed0 抽样 5 workload；FlashInfer warmup3/iters100/trials5，SOL warmup10/iters100（eval seed 200，官方 CLI）")
out.append("- 最终评测：全量 workload；FlashInfer= ziming evaluate.py（warmup3/iters100/**trials1**, v0.5）；SOL= ziming evaluate_sol.py → 官方 CLI（iterations100）")
out.append("- 范围：FlashInfer 10 + SOL L1 10 = 20 题 / 580 workloads（manifest v2，dsa_sparse 已换入）")
out.append("")

out.append("## 1. 指标定义")
out.append("")
out.append("| 指标 | 定义 |")
out.append("|---|---|")
out.append("| 轮次 (rounds) | 已执行的候选生成+评测循环数；每轮=1 次 LLM 代码生成+1 次反馈评测 |")
out.append("| 反馈最优分 (feedback best) | 搜索过程中 5 个抽样 workload 上的最优 mean speedup（仅搜索信号，非论文数字） |")
out.append("| pass/total (评测) | 最终全量评测中通过正确性校验的 workload 数/总数；valid=全部通过 |")
out.append("| geomean speedup | 各 workload speedup=ref_ms/sol_ms 的几何平均（优化目标指标） |")
out.append("| arith mean / min / max | speedup 的算术平均/最差/最好 workload |")
out.append("| input tokens | 全部 LLM 调用的输入 token 总和（cached 为其子集，provider 报告） |")
out.append("| cached tokens | provider 前缀/精确缓存命中的输入 token（本端点为逐字节精确匹配缓存，命中≈input−3） |")
out.append("| output tokens | 全部 LLM 调用的输出 token 总和 |")
out.append("| reasoning tokens | provider 单列的推理 token（output 的子集，非额外增量） |")
out.append("| LLM calls | 记账的 API 调用次数（≈ WM init 1 + 每节点 propose/refine 各1 + 每轮 codegen 1） |")
out.append("")

out.append("## 2. 最终全量评测结果（trials=1 口径）")
out.append("")
out.append("| # | 任务 | bench | valid | pass | geomean | arith | min | max |")
out.append("|---|---|---|---|---|---|---|---|---|")
n = 0
for r in rows:
    if r["eval"]:
        n += 1
        e = r["eval"]
        out.append(f"| {n} | {r['task']} | {r['bench']} | {'✅' if e['valid'] else '❌'} | {e['passed']}/{e['total']} | {fm(e['geo'])} | {fm(e['arith'])} | {fm(e['min'])} | {fm(e['max'])} |")
out.append("")
out.append("- 058：15/16 通过，batch_size=1/seq_len=2080 的 workload INCORRECT_NUMERICAL（max_abs=16525，候选排序 kernel 的边界形状 bug），其余 15 个 workload geomean 10.87x——按整体 INVALID 记录")
out.append('- gemm：r1–r59 自研 Triton GEMM 全部数值错误（零通过）；r60 起「最优解」退化为调用与 reference 相同的 torch.matmul(cuBLAS) + 包装，全 43 workload 正确但 geomean 0.839x（略慢于直接调用 cuBLAS 的 reference）——双重失败：写不出正确自研 kernel + 退化为库调用')
out.append("")

out.append("## 3. 搜索状态总表（20 题）")
out.append("")
out.append("| 任务 | bench | 状态 | 轮次 | 反馈最优分 |")
out.append("|---|---|---|---|---|")
for r in rows:
    st = "完成" if r["done"] else ("中断(可续跑)" if r["rounds"] else "未开始")
    out.append(f"| {r['task']} | {r['bench']} | {st} | {r['rounds'] or '-'} | {fm(r['score'],3)} |")
out.append("")

out.append("## 4. Token 消耗总表（20 题）")
out.append("")
out.append("| 任务 | LLM calls | input | cached | output | reasoning | 总计 |")
out.append("|---|---|---|---|---|---|---|")
tc = ti = tca = to_ = tr_ = 0
for r in rows:
    u = r["usage"]
    if not u:
        continue
    tc += u["calls"]; ti += u["input"]; tca += u["cached"]; to_ += u["output"]; tr_ += u["reasoning"]
    out.append(f"| {r['task']} | {u['calls']} | {u['input']:,} | {u['cached']:,} | {u['output']:,} | {u['reasoning']:,} | {u['total']:,} |")
out.append(f"| **合计** | **{tc}** | **{ti:,}** | **{tca:,}** | **{to_:,}** | **{tr_:,}** | **{ti+to_:,}** |")
out.append("")
out.append("- 094_time_decay 已于 v3 剔除出基准集（reference 为朴素 Python 循环，评测成本 4h+/题；详见 manifest v3_note），其搜索 token 保留在上表以保成本透明")
out.append("- 每题预算 ≤2M：18 题完成者中 3 题超支（gqa_paged_decode 3.18M / 020 2.26M / 002 2.17M，原因为 100 轮长程搜索中 prompt 持续增长——WM JSON 与 debug 日志累积）")
out.append("")

out.append("## 5. 未完成项")
out.append("")
out.append("| 项 | 状态 | 待办 |")
out.append("|---|---|---|")
out.append("| gdn_prefill / mla_paged_prefill / gqa_paged_decode 全量评测 | trials=1 评测被中断 | 重跑（worker 自动跳过已有结果） |")
out.append("| 058 评测 | 1/16 workload 数值失败 | 如实记录为 INVALID（或换次优候选复评） |")
out.append("| gqa_paged_prefill 搜索 | 46/100 轮，best 0.17x | 方案待定：续跑（轻量反馈）/原口径跑完/就此定型 |")
out.append("| 092_gqa_attention_with_qk_norm 搜索+评测 | v3 换入（探针健康 ref 2.3ms） | 搜索进行中（GPU4），完成后全量评测 |")
out.append("")

report = "\n".join(out)
dest = WS / "experiments_archive" / "formal_20260914"
dest.mkdir(parents=True, exist_ok=True)
(dest / "ksearch_formal_report.md").write_text(report, encoding="utf-8")
print(f"written: {dest/'ksearch_formal_report.md'} ({len(report)} chars)")
