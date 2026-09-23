#!/usr/bin/env python3
"""L2 批次（formal2_solL2_20260916）报告生成"""
import json, glob, re
from pathlib import Path
from datetime import datetime

WS = Path("/data1/workspace/weihongren")
TAG = "formal2_solL2_20260916"
BASE = WS / "baseline/ksearch-sol-execbench/experiments" / TAG

L2_TASKS = [
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

# torch 退化扫描结果（AST 分析 run() 内库调用）
TORCH_DEGEN = {
    "012": "部分退化(bmm×3)",
    "015": "部分退化(conv2d×3,linear)",
    "030": "干净",
    "036": "合理分步(mm×4,小GEMM)",
    "040": "干净",
    "043": "干净",
    "049": "干净",
    "051": "部分退化(addmm×2,rfft/irfft)",
    "057": "干净",
    "080": "部分退化(matmul×7,addmm)",
}

def usage(task):
    f = BASE / task / "run_seed0" / "usage.jsonl"
    if not f.exists(): return None
    c=i=ca=o=r=0
    for line in f.read_text().splitlines():
        line=line.strip()
        if not line: continue
        try: x=json.loads(line)
        except: continue
        c+=1; i+=x.get("input_tokens") or 0; ca+=x.get("input_cached_tokens") or 0
        o+=x.get("output_tokens") or 0; r+=x.get("reasoning_tokens") or 0
    return {"calls":c,"input":i,"cached":ca,"output":o,"reasoning":r,"total":i+o}

def eval_info(task):
    f = BASE / task / "run_seed0" / "unified" / "performance.json"
    if not f.exists(): return None
    d = json.load(open(f))
    pw = d.get("per_workload", [])
    passed = d.get("passed", sum(1 for w in pw if w.get("status")=="PASSED"))
    total = d.get("total", len(pw))
    sps = [w.get("speedup") for w in pw if w.get("status")=="PASSED" and w.get("speedup")]
    return {"valid":d.get("valid"),"passed":passed,"total":total,
            "geo":d.get("geomean_speedup"),"arith":d.get("arithmetic_mean_speedup"),
            "min":min(sps) if sps else None,"max":max(sps) if sps else None}

def search_info(task):
    rd = BASE / task / "run_seed0"
    log = rd / "campaign_stdout.log"
    rounds = 0; best = None
    if log.exists():
        txt = log.read_text(errors="ignore")
        rounds = txt.count("Optimization Round")
        scores = re.findall(r"score=([0-9.]+)", txt)
        best = max(float(s) for s in scores) if scores else None
    return rounds, best

def fm(x,nd=2): return f"{x:.{nd}f}" if isinstance(x,(int,float)) else "-"

rows=[]
for t in L2_TASKS:
    r,b = search_info(t)
    rows.append({"task":t,"rounds":r,"best":b,"usage":usage(t),"eval":eval_info(t),
                 "degen":TORCH_DEGEN.get(t[:3],"?")})

out=[]
out.append("# K-Search L2 批次报告（formal2_solL2_20260916）")
out.append("")
out.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
out.append("- 方法：K-Search world-model 完整版，seed0，100 轮 = 20 节点 × 5 attempt")
out.append("- LLM：AWS-GPT-5.6-Sol @ llmapi.isrc.ac.cn/v1（key: K_SEARCH_KEY sk-4byTPK...）")
out.append("- 硬件：A800-SXM4-80GB；GPU 池化（{5,6} flock 并发，部分题亲和模式）")
out.append("- 防 torch 回退守卫：已注入（上游 KernelBench 同款文本，本地新增）")
out.append("- 最终评测：evaluate_sol.py（官方 CLI），iterations=100，全量 workloads")
out.append("")

out.append("## 1. 最终全量评测结果")
out.append("")
out.append("| # | 任务 | valid | pass | geomean | arith | min | max | torch 退化 |")
out.append("|---|---|---|---|---|---|---|---|---|")
n=0
tc=ti=to_=tr_=0
for r in rows:
    if r["eval"]:
        n+=1; e=r["eval"]
        out.append(f"| {n} | {r['task']} | {'✅' if e['valid'] else '❌'} | {e['passed']}/{e['total']} | {fm(e['geo'])} | {fm(e['arith'])} | {fm(e['min'])} | {fm(e['max'])} | {r['degen']} |")
out.append("")
out.append("- 057_residual_coupling：9/16 通过（7 个 workload 反馈阶段未覆盖到的 shape 失败），反馈抽样 5 全过但全量出现泛化缺口——记 INVALID")
out.append("- torch 退化分类：「干净」= run() 零库调用；「合理分步」= 库调用仅做辅助小步骤（如 pwconv 梯度的小 GEMM）；「部分退化」= 核心计算部分调库（conv2d/linear 主体），自研 Triton 只融合周边")
out.append("")

out.append("## 2. 搜索状态与反馈最优分")
out.append("")
out.append("| 任务 | 轮次 | 反馈 best |")
out.append("|---|---|---|")
for r in rows:
    out.append(f"| {r['task']} | {r['rounds']} | {fm(r['best'],3)} |")
out.append("")

out.append("## 3. Token 消耗")
out.append("")
out.append("| 任务 | LLM calls | input | cached | output | reasoning | 总计 |")
out.append("|---|---|---|---|---|---|---|")
for r in rows:
    u=r["usage"]
    if not u: continue
    tc+=u["calls"];ti+=u["input"];to_+=u["output"];tr_+=u["reasoning"]
    out.append(f"| {r['task']} | {u['calls']} | {u['input']:,} | {u['cached']:,} | {u['output']:,} | {u['reasoning']:,} | {u['total']:,} |")
out.append(f"| **合计** | **{tc}** | **{ti:,}** | | **{to_:,}** | **{tr_:,}** | **{ti+to_:,}** |")
out.append("")

report="\n".join(out)
dest=WS/"notes/reports"
dest.mkdir(parents=True,exist_ok=True)
(dest/"k-search-L2_0917.md").write_text(report,encoding="utf-8")
print(f"written: {dest/'k-search-L2_0917.md'} ({len(report)} chars)")
