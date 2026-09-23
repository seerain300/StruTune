#!/usr/bin/env python3
"""Generate summary tables and best_solutions/ for the dr-kernel-a800 archive.

Reads ONLY原始 JSON（tasks/**/summary.json + final/evaluation.json + state.json）,
never hand-typed numbers. Two batches:
  - stts: tasks/            (kbstyle_stts_full8: 8 samples x 10 iters)
  - stts3turn: tasks_v2/    (kbstyle_campaign_v2: 8 samples x 3 turns)
"""
import json
import glob
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BATCHES = {'stts': ROOT / 'tasks', 'stts3turn': ROOT / 'tasks_v2'}

rows = []
solved = []  # (task, batch, pass, geo, best_src, src_path)
for batch, troot in BATCHES.items():
    for f in sorted(glob.glob(str(troot / '**' / 'summary.json'), recursive=True)):
        s = json.load(open(f))
        task_dir = Path(f).parent
        key = s['task']
        geo = s.get('final_geomean_speedup')
        # best solution provenance: state.json samples[*].best = [reward, geomean, turn, source]
        best_meta = None
        state_f = task_dir / 'state.json'
        if state_f.exists() and batch == 'stts':
            st = json.load(open(state_f))
            for i, sm in enumerate(st['samples']):
                b = sm.get('best')
                if b:
                    if best_meta is None or b[0] > best_meta[1]:
                        best_meta = (i, b[0], b[1], b[2])
        # timing detail from final perf
        perf_f = task_dir / 'final' / 'evaluation.json'
        final_cfg = None
        if perf_f.exists():
            try:
                final_cfg = (json.load(open(perf_f)).get('evaluation_config')) or {}
            except json.JSONDecodeError:
                pass
        rows.append({
            'task': key, 'batch': batch, 'pass_at_1': s['pass_at_1'],
            'best_of_history_geomean_feedback': s.get('best_of_history_geomean'),
            'final_geomean_speedup': geo,
            'final_valid': (s.get('final_correctness') or {}).get('valid'),
            'final_config': final_cfg,
            'dir': str(task_dir.relative_to(ROOT)),
        })
        if geo and geo > 0 and (s.get('final_correctness') or {}).get('valid') is True:
            src = task_dir / 'best_solution.py'
            if src.exists():
                solved.append((key, batch, s['pass_at_1'], geo, best_meta, src))

# ---- CSV
with open(ROOT / 'summary' / 'results_table.csv', 'w') as fh:
    fh.write('task,batch,pass_at_1,best_feedback_geomean,final_geomean_speedup(A800),final_valid\n')
    for r in rows:
        fh.write(f"{r['task']},{r['batch']},{r['pass_at_1']:.3f},"
                 f"{r['best_of_history_geomean_feedback'] or ''},"
                 f"{r['final_geomean_speedup'] if r['final_geomean_speedup'] is not None else ''},"
                 f"{r['final_valid']}\n")

# ---- MD
def fmt(x, nd=2):
    return f'{x:.{nd}f}' if isinstance(x, (int, float)) else '—'

lines = [
    '# dr-kernel-a800 结果总表',
    '',
    '- 模型：hkust-nlp/drkernel-8b（vLLM 推理，权重与官方 main @990fe42 一致）',
    '- 硬件：NVIDIA A800-SXM4-80GB（生成 2×vLLM；评测走 GPU 池）',
    '- **final_geomean_speedup 口径：官方评测器全新复评（全 workload、参考重计时、warmup 3 + 100 iters），A800 硬件绑定**',
    '- best_feedback_geomean：轮级反馈口径（参考延迟缓存 + warmup 3 + 10 iters），仅用于轨迹内选择，不可与 final 混用',
    '- pass_at_1：该批次 8 条采样中"任一轮全部 workload 正确"的采样占比',
    '',
    '## 批次一 stts（STTS：8 采样 × 最多 10 迭代 × 每迭代 5 轮，2026-09-19~20）',
    '',
    '| 任务 | pass@1 | best反馈geomean | **终局speedup(A800)** | 解出 |',
    '|---|---|---|---|---|',
]
stts_rows = [r for r in rows if r['batch'] == 'stts']
for r in sorted(stts_rows, key=lambda x: (-(x['final_geomean_speedup'] or 0), x['task'])):
    ok = '✅' if (r['final_geomean_speedup'] or 0) > 1 else ('·' if r['final_geomean_speedup'] else '✗')
    lines.append(f"| {r['task']} | {r['pass_at_1']:.0%} | "
                 f"{fmt(r['best_of_history_geomean_feedback'])} | "
                 f"{fmt(r['final_geomean_speedup'])} | {ok} |")
lines += ['', '## 批次二 stts3turn（基线：8 采样 × 3 轮反馈，2026-09-18）', '',
          '| 任务 | pass@1 | 终局speedup(A800) | 解出 |', '|---|---|---|---|']
for r in sorted([r for r in rows if r['batch'] == 'stts3turn'],
                key=lambda x: (-(x['final_geomean_speedup'] or 0), x['task'])):
    ok = '✅' if (r['final_geomean_speedup'] or 0) > 1 else ('·' if r['final_geomean_speedup'] else '✗')
    lines.append(f"| {r['task']} | {r['pass_at_1']:.0%} | {fmt(r['final_geomean_speedup'])} | {ok} |")
(ROOT / 'summary' / 'results_table.md').write_text('\n'.join(lines) + '\n')

# ---- best_solutions（只收终局权威口径有效题，按规范带来源头注释）
bs = ROOT / 'summary' / 'best_solutions'
bs.mkdir(exist_ok=True)
for key, batch, p, geo, best_meta, src in solved:
    dst = bs / (key.replace('/', '__') + '.py')
    header = [f'# task: {key}', f'# batch: {batch}',
              f'# pass_at_1: {p:.2f}', f'# final_geomean_speedup(A800, official re-eval): {geo:.3f}']
    if best_meta:
        header += [f'# provenance: sample s{best_meta[0]} reward={best_meta[1]:.3f} '
                   f'feedback_geomean={best_meta[2]:.3f} turn={best_meta[3]}']
    body = src.read_text()
    if not body.startswith('# task:'):
        body = '\n'.join(header) + '\n' + body
    dst.write_text(body)
print(f'rows={len(rows)} solved={len(solved)}')
for key, batch, p, geo, m, _ in solved:
    print(f'  best: {key} [{batch}] {geo:.2f}x')
