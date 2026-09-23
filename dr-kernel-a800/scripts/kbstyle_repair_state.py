#!/usr/bin/env python3
"""Repair flashinfer task states corrupted by the '--mode full' argparse bug.

For every fib task with a state.json: re-run the (now fixed) full evaluation for
each turn>=1 sample whose solution.py exists but whose evaluation.json is
missing, then rebuild records/rewards/histories/best from disk artifacts
(prompt.txt, s*t*/response.txt, re-evaluated results).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, '/data1/workspace/weihongren/scripts')
import kbstyle_campaign as C  # noqa: E402


def repair_task(pool, args, task, task_dir):
    state = json.loads((task_dir / 'state.json').read_text())
    records = state['records']
    prompt = (task_dir / 'prompt.txt').read_text()
    changed = 0
    for s in range(C.N_SAMPLES):
        for rec in records[s]['samples']:
            t = rec['turn']
            if t < 1 or not rec.get('code_ok') or not rec.get('compliant'):
                continue
            sd = task_dir / f's{s}t{t}'
            if not (sd / 'solution.py').exists():
                continue
            already = sd / 'evaluation.json'
            if already.exists():
                try:
                    rec['status'] = json.loads(already.read_text())
                    continue
                except json.JSONDecodeError:
                    pass
            print(f'  re-eval {task_dir.name} s{s}t{t}', flush=True)
            status = C.eval_candidate(pool, args, task, (sd / 'solution.py').read_text(), sd, 'full')
            rec['status'] = status.get('result')
            changed += 1

    # rebuild derived fields + histories
    for s in range(C.N_SAMPLES):
        for rec in records[s]['samples']:
            if rec.get('code_ok') and rec.get('compliant') and rec.get('status'):
                reward, info = C.reward_of({'result': rec['status']})
                rec['reward'], rec['info'] = reward, info
            else:
                rec['reward'], rec['info'] = None, None
        hist = [{'role': 'user', 'content': prompt}]
        for rec in records[s]['samples']:
            sd = task_dir / f"s{rec['turn']}"  # placeholder, replaced below
            resp = ''
            resp_path = task_dir / f"s{s}t{rec['turn']}" / 'response.txt'
            if resp_path.exists():
                resp = C.compact(resp_path.read_text())
            if not rec.get('code_ok'):
                fb = C.feedback_text(False, None)
            elif not rec.get('compliant'):
                fb = ('Your submission violates the TRITON-ONLY requirement and was not '
                      'evaluated:\n- ' + '\n- '.join((rec.get('violations') or [])[:6]) +
                      '\nRewrite ModelNew so that all computation happens in @triton.jit '
                      'kernels launched by forward.')
            else:
                fb = C.feedback_text(True, {'result': rec.get('status') or {}})
            hist += [{'role': 'assistant', 'content': resp}, {'role': 'user', 'content': fb}]
        state['histories'][s] = hist

    best = None
    for s in range(C.N_SAMPLES):
        for rec in records[s]['samples']:
            info = rec.get('info') or {}
            if rec.get('compliant') and info.get('correct_ratio', 0) >= 1.0:
                sd = task_dir / f"s{s}t{rec['turn']}"
                if (sd / 'solution.py').exists():
                    cand = (rec.get('reward') or 0, info.get('geomean'), s, rec['turn'],
                            (sd / 'solution.py').read_text())
                    if best is None or cand[0] > best[0]:
                        best = cand
    state['best'] = list(best) if best else None
    state_path = task_dir / 'state.json'
    state_path.write_text(json.dumps(state, default=str))
    return changed


def main():
    args = C.parse_args()
    args.max_parallel = 4
    root = C.RUN_ROOT / 'tasks' / 'flashinfer'
    tasks = {t['key']: t for t in C.load_tasks([])}
    pool = C.GPUPool([int(g) for g in args.gpus.split(',')], 4, 10)
    total = 0
    for task_dir in sorted(root.iterdir()):
        if not (task_dir / 'state.json').exists():
            continue
        task = tasks.get(f'flashinfer/{task_dir.name}')
        if task is None:
            continue
        st = json.loads((task_dir / 'state.json').read_text())
        if st.get('next_turn', 0) < 2:
            continue  # turn-1 not run yet, nothing corrupted
        n = repair_task(pool, args, task, task_dir)
        total += n
        print(f'repaired {task_dir.name}: {n} re-evals', flush=True)
    print(f'total re-evals: {total}')


if __name__ == '__main__':
    main()
