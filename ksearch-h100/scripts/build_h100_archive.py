#!/usr/bin/env python3
"""K-Search H100 实验归档构建：按 whr/实验产物整理与上传规范.md 整理到 ksearch-h100/。

复制式归档（原始工作区不动），结构：
  ksearch-h100/
    README.md, protocol.md, manifest
    summary/ results_table.md/.csv + best_solutions/ + best_timing_detail.md/.csv
    tasks/<题>/<批次>/  rounds/ solutions/ world_model/ eval/ unified/ usage.jsonl campaign_stdout.log
    batch_logs/  campaign 主日志 + supervisor 日志
    scripts/ evaluators/ docs/
剔除：stdout_*.log（重启横幅）、__pycache__、refcache。
"""
from __future__ import annotations
import json, math, shutil, csv, io
from pathlib import Path
from datetime import datetime

WS = Path("/home/ziming/ksearch_h100_portable")
OUT = Path("/home/ziming/ksearch_upload/ksearch-h100")

FI_DIR = WS / "baseline/ksearch/experiments/formal_h100"
L1_DIR = WS / "baseline/ksearch-sol-execbench/experiments/formal_sol_h100"
L2_DIR = WS / "baseline/ksearch-sol-execbench/experiments/formal_solL2_h100"

FI_TASKS = [
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
L1_TASKS = [
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
# 终评结果与口径注记（从 unified/evaluation.json 或 performance.json 现读）
VIOLATIONS = {  # 违规留档（BANNED 命中，不终评）
    "002_vae_conv3x3_groupnorm_silu_residual_fused": "F.conv2d×2（conv 本体调库，6 个 Triton 内核全部死代码）",
    "007_hyena_fft_size_padding_rfft": "torch.fft.rfft×2（FFT 本体调 cuFFT）",
    "092_gqa_attention_with_qk_norm": "F.linear×4（q/k/v/o 投影调库；attention 本体 Triton）",
    "012_moe_expert_batched_execution_with_capacity_factor": "torch.bmm×3",
    "015_audio_sinusoidal_position_embedding_with_conv_projection": "F.conv2d×5",
    "036_convnextv2_layer_with_nhwc_persistence_backward": "@torch.compile + torch.mm×2",
    "051_seqlen-finetuned-reconstructed_hyena_complete_forward_block": "F.conv2d/F.linear/F.conv1d/torch.fft 共 11 处（纯 torch 委托）",
    "057_residual_coupling_flow_block": "F.conv1d×3",
    "080_moe_complete_layer_with_shared_expert_backward": "torch.mm×4",
}
NOTES = {
    "gdn_prefill_qk4_v8_d128_k_last": "非标口径†：warmup3+20iters+参考延迟走磁盘缓存（全量参考单遍 521min 不经济；候选侧实时计时，100/100 全过）",
    "gemm_n4096_k4096": "0.47x 为真实结果：r1-59 自研全败，退出解=调 cuBLAS（与 reference 同款），无自研可用",
    "043_mamba_chunk_scan_with_segsum": "退出解违规(torch.compile+cumsum+bmm)；表中数字为搜索期 best-round 干净解(3.20x)复评；该 best-round 代码丢失注记见 docs/RUNBOOK",
    "030_flux_concatenated_sequence_processing_with_split": "预算超支 141 轮；best@31=1.62x 轮次代码丢失，退出保存解 1.57x",
}

def rd(task: str):
    for base, tasks, tag in ((FI_DIR, FI_TASKS, "formal_h100"),
                             (L1_DIR, L1_TASKS, "formal_sol_h100"),
                             (L2_DIR, L2_TASKS, "formal_solL2_h100")):
        if task in tasks:
            return base / task / "run_seed0", tag
    raise KeyError(task)

def read_final(run_dir: Path, task: str):
    """终评结果：FI 用 unified/evaluation.json；SOL 用 unified/performance.json。"""
    ej = run_dir / "unified/evaluation.json"
    pj = run_dir / "unified/performance.json"
    if ej.exists():
        j = json.loads(ej.read_text())
        per = j.get("per_workload") or []
        return dict(bench="FlashInfer", valid=j.get("all_passed", True),
                    passed=j.get("passed"), total=len(per),
                    geomean=j.get("geomean_speedup"), per=per)
    if pj.exists():
        j = json.loads(pj.read_text())
        per = j.get("per_workload") or j.get("workloads") or []
        def fg(o, pre=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    if "geomean" in k.lower() and isinstance(v, (int, float)):
                        return v
                for v in o.values():
                    r = fg(v)
                    if r: return r
            return None
        return dict(bench="SOL-ExecBench", valid=j.get("valid", True),
                    passed=j.get("passed") or j.get("num_passed"),
                    total=j.get("total") or j.get("num_total") or len(per),
                    geomean=fg(j), per=per)
    return None

def copy_tree_filtered(src: Path, dst: Path):
    """复制目录，剔除 stdout_*.log / __pycache__ / refcache。"""
    if not src.exists():
        return
    shutil.copytree(src, dst, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "stdout_*.log",
                                                  "*refcache*", "*.pyc"))

def usage_summary(run_dir: Path):
    """该题 usage.jsonl 的 token 记账（全量口径，含 cached 拆分）。"""
    p = run_dir / "usage.jsonl"
    if not p.exists():
        return None
    calls = pin = pcached = pout = 0
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except Exception: continue
        if r.get("event") != "completion": continue
        calls += 1
        pin += int(r.get("input_tokens") or 0)
        pcached += int(r.get("input_cached_tokens") or 0)
        pout += int(r.get("output_tokens") or 0)
    return dict(calls=calls, input=pin, cached=pcached, output=pout)

def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "summary/best_solutions").mkdir(parents=True)
    rows = []
    for group, tasks in (("FlashInfer", FI_TASKS), ("SOL-L1", L1_TASKS), ("SOL-L2", L2_TASKS)):
        for t in tasks:
            run_dir, batch = rd(t)
            fin = read_final(run_dir, t)
            us = usage_summary(run_dir)
            # 轮次计数
            log = run_dir / "campaign_stdout.log"
            rounds = 0
            if log.exists():
                txt = log.read_text(errors="ignore")
                rounds = txt.count("Round summary") + txt.count("feedback workloads passed")
            if t in VIOLATIONS:
                rows.append(dict(group=group, task=t, bench=("FlashInfer" if group=="FlashInfer" else "SOL-ExecBench"),
                                 status="违规留档", passed=None, total=None, geomean=None,
                                 rounds=rounds, note=VIOLATIONS[t], **(us or {})))
            else:
                note = NOTES.get(t, "")
                rows.append(dict(group=group, task=t, bench=fin["bench"],
                                 status="已终评", passed=fin["passed"], total=fin["total"],
                                 geomean=fin["geomean"], rounds=rounds, note=note, **(us or {})))
    # ---- results_table.md / .csv ----
    def fmt_row(r):
        g = f"{r['geomean']:.2f}x" if r.get("geomean") is not None else "—"
        pt = f"{r['passed']}/{r['total']}" if r.get("passed") is not None else "—"
        return g, pt
    md = ["# K-Search H100 结果总表", "",
          "- 口径：终评 = 全量 workload，warmup3 + 100 iterations，参考与候选同进程成对计时（无缓存）；",
          "  唯一例外 gdn_prefill（† 非标口径，见注记列）",
          "- 违规留档 = 退出解含 BANNED torch 调用（cuDNN/cuBLAS/cuFFT 委托），按协议不终评；",
          "  token 列为该题有效搜索消耗（含违规题——搜索是真实发生的）",
          "- Token 口径：该题全部完成评测轮窗口内的 LLM 调用记账（重试损耗已扣）",
          "",
          "| 组 | 任务 | 状态 | pass | 终评 geomean | 轮次 | 输入(M) | 输出(M) | 注记 |",
          "|---|---|---|---|---|---|---|---|---|"]
    csv_rows = []
    for r in rows:
        g, pt = fmt_row(r)
        pin = f"{r['input']/1e6:.2f}" if r.get("input") else "—"
        pout = f"{r['output']/1e6:.2f}" if r.get("output") else "—"
        md.append(f"| {r['group']} | {r['task']} | {r['status']} | {pt} | {g} | {r['rounds']} | {pin} | {pout} | {r['note']} |")
        csv_rows.append([r["group"], r["task"], r["status"],
                         r.get("passed") if r.get("passed") is not None else "",
                         r.get("total") if r.get("total") is not None else "",
                         round(r["geomean"], 2) if r.get("geomean") else "",
                         r["rounds"], r.get("input", 0), r.get("cached", 0), r.get("output", 0), r["note"]])
    measured = [r["geomean"] for r in rows if r.get("geomean") is not None]
    g_all = math.exp(sum(math.log(v) for v in measured) / len(measured))
    no_gdn = [v for r in rows if r.get("geomean") is not None
              and r["task"] != "gdn_prefill_qk4_v8_d128_k_last" for v in [r["geomean"]]]
    g_std = math.exp(sum(math.log(v) for v in no_gdn) / len(no_gdn))
    md += ["",
           f"- **已终评 {len(measured)} 题，geomean-of-geomean = {g_all:.2f}x**（含 gdn_prefill 非标口径；",
           f"  剔除后 {len(no_gdn)} 题 = {g_std:.2f}x）；违规留档 {len(VIOLATIONS)} 题",
           f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}；数字直接来自各题 unified/ 评测 JSON"]
    (OUT / "summary/results_table.md").write_text("\n".join(md), encoding="utf-8")
    with open(OUT / "summary/results_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "task", "status", "passed", "total", "geomean", "rounds",
                    "input_tokens", "input_cached_tokens", "output_tokens", "note"])
        w.writerows(csv_rows)
    print(f"[summary] results_table: {len(rows)} 行 (终评 {len(measured)}, 违规 {len(VIOLATIONS)})")
    print(f"[summary] geomean-of-geomean: {g_all:.2f}x (标准口径 {len(no_gdn)} 题 {g_std:.2f}x)")

    # ---- best_solutions + best_timing_detail ----
    timing_md = ["# 最优解逐 workload 计时明细", "",
                 "每行：题目 / workload 序号 / 参考ms / 候选ms / speedup（终评口径 warmup3+100iters）", ""]
    timing_csv = csv.writer(open(OUT / "summary/best_timing_detail.csv", "w", newline="", encoding="utf-8"))
    timing_csv.writerow(["task", "wl_idx", "uuid8", "status", "ref_ms", "sol_ms", "speedup"])
    n_best = 0
    for r in rows:
        if r["status"] != "已终评":
            continue
        run_dir, batch = rd(r["task"])
        # FI: unified/main.py；SOL: unified/solution.sol.json 的 sources[0]（main.py）
        main_py = run_dir / "unified/main.py"
        code = None
        if main_py.exists():
            code = main_py.read_text(errors="ignore")
        else:
            solj = run_dir / "unified/solution.sol.json"
            if solj.exists():
                try:
                    j = json.loads(solj.read_text())
                    srcs = [s for s in j.get("sources", []) if s.get("path", "").endswith("main.py")]
                    code = (srcs or j.get("sources", [{}]))[0].get("content", "")
                except Exception as e:
                    print(f"[warn] {r['task']}: solution.sol.json 解析失败 {e}")
        if not code:
            print(f"[warn] {r['task']}: 终评解代码缺失，跳过 best_solution")
            continue
        header = (f"# K-Search H100 best solution — {r['task']}\n"
                  f"# 来源：{batch} / seed0 / 退出保存最优解（终评 mtime 最新）\n"
                  f"# 终评：{r['passed']}/{r['total']} passed, geomean {r['geomean']:.2f}x"
                  f" (warmup3+100iters 全量 workload)\n"
                  f"# token：输入 {r['input']/1e6:.2f}M (cached {r['cached']/1e6:.2f}M) / 输出 {r['output']/1e6:.2f}M\n"
                  + (f"# 注记：{r['note']}\n" if r["note"] else "") + "\n")
        (OUT / "summary/best_solutions" / f"{r['task']}.py").write_text(header + code, encoding="utf-8")
        n_best += 1
        fin = read_final(run_dir, r["task"])
        for i, pw in enumerate(fin["per"]):
            ref = pw.get("ref_ms") or pw.get("reference_ms")
            sol = pw.get("sol_ms") or pw.get("solution_ms")
            sp = pw.get("speedup")
            st = pw.get("status", "")
            u = (pw.get("uuid") or "")[:8]
            if sp is not None:
                timing_md.append(f"- {r['task']} [{i+1}] {st} ref={ref:.4f}ms sol={sol:.4f}ms speedup={sp:.2f}x")
            timing_csv.writerow([r["task"], i+1, u, st,
                                 f"{ref:.4f}" if ref else "", f"{sol:.4f}" if sol else "",
                                 f"{sp:.2f}" if sp else ""])
    (OUT / "summary/best_timing_detail.md").write_text("\n".join(timing_md), encoding="utf-8")
    print(f"[summary] best_solutions: {n_best} 个; timing 明细 {sum(1 for l in timing_md if l.startswith('- '))} 行")

    # ---- tasks/<题>/<批次>/ 原始产物 ----
    for group, tasks in (("FlashInfer", FI_TASKS), ("SOL-L1", L1_TASKS), ("SOL-L2", L2_TASKS)):
        for t in tasks:
            run_dir, batch = rd(t)
            dst = OUT / "tasks" / t / batch
            copy_tree_filtered(run_dir / "ksearch-artifacts", dst / "ksearch-artifacts")
            copy_tree_filtered(run_dir / "unified", dst / "final")
            for f in ("usage.jsonl", "campaign_stdout.log", "DONE", "exit_code"):
                src = run_dir / f
                if src.exists():
                    shutil.copy2(src, dst / f)
    print("[tasks] 30 题原始产物复制完成")

    # ---- batch_logs / scripts / evaluators / docs ----
    bl = OUT / "batch_logs"
    bl.mkdir(parents=True, exist_ok=True)
    for f in ("/tmp/par6_campaign.log", "/tmp/ksearch_supervisor.log",
              "/tmp/sol_l1_campaign.log", "/tmp/sol_l2_campaign.log"):
        p = Path(f)
        if p.exists():
            shutil.copy2(p, bl / p.name)
    copy_tree_filtered(WS / "scripts", OUT / "scripts")
    copy_tree_filtered(WS / "evaluators", OUT / "evaluators")
    copy_tree_filtered(WS / "ksearch_patchshim", OUT / "ksearch_patchshim")
    shutil.copy2(WS / "ksearch-run.sh", OUT / "scripts/ksearch-run.sh")
    shutil.copy2(WS / "ksearch-token-run.py", OUT / "scripts/ksearch-token-run.py")
    docs = OUT / "docs"
    docs.mkdir(exist_ok=True)
    for f in ("protocol.md", "experiment_manifest.json",
              "notes/TOKEN_USAGE_PER_TASK.md", "notes/L2_TORCH_GUARD_AUDIT.md",
              "notes/INCIDENT_RETROSPECTIVE.md", "notes/H100_RUNBOOK_v0.6.md",
              "notes/H100_REPRODUCTION_GUIDE.md", "notes/REF_LATENCY_CACHE_STRATEGY.md",
              "notes/reports/k-search_30t_0917.md"):
        p = WS / f
        if p.exists():
            shutil.copy2(p, docs / p.name)
    print("[misc] batch_logs/scripts/evaluators/docs 完成")

    # 体积
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"[done] {OUT} 总计 {total/1e6:.0f}MB")

if __name__ == "__main__":
    main()
