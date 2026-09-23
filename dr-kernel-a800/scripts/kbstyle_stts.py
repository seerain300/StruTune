#!/usr/bin/env python3
"""STTS campaign for drkernel-8b over all 30 tasks (KernelGYM maxturns5-maxiter10 style).

Per task, 8 independent sample trajectories. Each trajectory runs segments of up
to MAX_TURNS(5) user turns; between segments the context is compressed to the
best consecutive REMAIN_TURNS(4) window by summed reward (iteration_method=best,
metric=reward). Up to ITERATIONS(10) segments, with per-sample patience early
stop. Existing 3-turn v2 campaign state is imported as the first (partial)
segment. Final reporting = best-of-history per sample + official fresh eval of
the task best. Token usage is persisted per request.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

import kbstyle_campaign as C  # reuse pool/eval/rewards/compliance/feedback/prompts

WS = Path('/data1/workspace/weihongren')
V2_ROOT = WS / 'baseline/drtriton/kbstyle_campaign_v2_20260918'
RUN_ROOT = WS / 'baseline/drtriton/kbstyle_stts_full8_20260919'
RESULTS_LOCK = threading.Lock()
TOKENS_LOCK = threading.Lock()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpus', default='0,1,2,3,4,5')
    p.add_argument('--max-parallel', type=int, default=6)
    p.add_argument('--task', action='append', default=[])
    p.add_argument('--servers', default='http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1')
    p.add_argument('--samples', type=int, default=1,
                   help='independent trajectories per task (1 = ceiling probe, 8 = paper default)')
    p.add_argument('--iterations', type=int, default=10)
    p.add_argument('--max-turns', type=int, default=5)
    p.add_argument('--remain-turns', type=int, default=4)
    p.add_argument('--patience', type=int, default=2)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--max-tokens', type=int, default=8192)
    p.add_argument('--seed', type=int, default=20260918)
    p.add_argument('--concurrent-tasks', type=int, default=3)
    args = p.parse_args()
    args.full_timeout = 1200
    args.screen_timeout = 420
    args.precompute_timeout = 2700
    args.final_timeout = 2700
    args.poll_seconds = 10
    args.model = 'drkernel-stop-8b'
    return args


def log(msg):
    print(f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}', flush=True)


class Router:
    """Round-robin across vLLM servers; identical model+seed -> identical output."""

    def __init__(self, urls, args):
        self.clients = [OpenAI(api_key='EMPTY', base_url=u, timeout=900, max_retries=2)
                        for u in urls.split(',')]
        self.args = args
        self.rr = 0
        self.lock = threading.Lock()

    def chat(self, messages, seed, n=1):
        for attempt in range(2):
            try:
                return self._chat_once(messages, seed, n)
            except Exception as e:
                if attempt == 0 and 'context length' in str(e):
                    trimmed = [messages[0]] + messages[-4:]
                    messages = trimmed
                    continue
                raise

    def _chat_once(self, messages, seed, n):
        with self.lock:
            client = self.clients[self.rr % len(self.clients)]
            self.rr += 1
        resp = client.chat.completions.create(
            model=self.args.model, messages=messages, n=n,
            temperature=self.args.temperature, top_p=self.args.top_p,
            max_tokens=self.args.max_tokens, stop=['<|im_end|>', '<|endoftext|>'], seed=seed)
        out = [{'text': c.message.content or '', 'finish_reason': c.finish_reason}
               for c in resp.choices]
        usage = {'prompt_tokens': getattr(resp.usage, 'prompt_tokens', 0) or 0,
                 'completion_tokens': getattr(resp.usage, 'completion_tokens', 0) or 0}
        with TOKENS_LOCK:
            with open(RUN_ROOT / 'tokens.jsonl', 'a') as fh:
                fh.write(json.dumps({'ts': time.time(), 'n': n, **usage}) + '\n')
        return out, usage


# ---------------------------------------------------------------- trajectory
def tc(text, chars=4000):
    """Tight compaction for history messages (chars ~ 0.27 tokens)."""
    if len(text) <= chars:
        return text
    half = chars // 2
    return text[:half] + '\n\n# ... (middle omitted) ...\n\n' + text[-half:]


def best_window(traj, keep, best_key='reward'):
    """traj: list of turn dicts with reward; return best consecutive window of len keep."""
    if len(traj) <= keep:
        return list(range(len(traj)))
    scores, windows = [], []
    for i in range(len(traj) - keep + 1):
        w = traj[i:i + keep]
        scores.append(sum((t.get('reward') or 0.0) for t in w))
        windows.append(list(range(i, i + keep)))
    return windows[scores.index(max(scores))]


def import_v2_state(task, task_dir, prompt, n_samples):
    """Convert the finished 3-turn v2 campaign state into an STTS trajectory state.
    With n_samples < 8 the BEST v2 samples (by best reward seen) are imported."""
    src = V2_ROOT / 'tasks' / task['key'] / 'state.json'
    if not src.exists():
        return None
    st = json.loads(src.read_text())
    per_sample = []
    for s in range(C.N_SAMPLES):
        traj = []
        for rec in st['records'][s]['samples']:
            info = rec.get('info') or {}
            t = rec['turn']
            resp_path = V2_ROOT / 'tasks' / task['key'] / f's{s}t{t}' / 'response.txt'
            sol_path = V2_ROOT / 'tasks' / task['key'] / f's{s}t{t}' / 'solution.py'
            traj.append({'turn': t, 'code_ok': rec.get('code_ok', False),
                         'compliant': rec.get('compliant', False),
                         'violations': rec.get('violations') or [],
                         'reward': rec.get('reward'),
                         'info': info, 'status': rec.get('status'),
                         'response': tc(resp_path.read_text()) if resp_path.exists() else '',
                         'feedback': (st['histories'][s][2 * t + 2]['content']
                                      if st['histories'][s] and len(st['histories'][s]) > 2 * t + 2
                                      else ''),
                         'source': sol_path.read_text() if sol_path.exists() else None})
        best = None
        for tr in traj:
            if tr['compliant'] and (tr.get('info') or {}).get('correct_ratio', 0) >= 1.0:
                cand = (tr.get('reward') or 0, (tr['info']).get('geomean'), tr['turn'], tr['source'])
                if best is None or cand[0] > best[0]:
                    best = cand
        per_sample.append({'traj': traj, 'best': list(best) if best else None,
                            'best_reward_seen': (best[0] if best else 0.0),
                            'patience': 0, 'done': False})
    order = sorted(range(C.N_SAMPLES),
                   key=lambda s: per_sample[s]['best_reward_seen'], reverse=True)
    state = {'prompt': prompt, 'seg_new_turns': 3, 'iteration': 1,
             'samples': [per_sample[s] for s in sorted(order[:n_samples])]}
    return state


def history_from_window(prompt, traj, idxs):
    msgs = [{'role': 'user', 'content': prompt}]
    for i in idxs:
        t = traj[i]
        msgs += [{'role': 'assistant', 'content': t['response']},
                 {'role': 'user', 'content': t['feedback']}]
    return msgs


def run_segment(pool, router, args, task, task_dir, state, task_index, seg_no):
    """Run up to max_turns - seg_new_turns new turns for every active sample."""
    prompt = state['prompt']
    budget = args.max_turns - state['seg_new_turns'] - state.get('seg_done_turns', 0)
    if budget <= 0:
        return 0
    n = len(state['samples'])
    active = [s for s in range(n) if not state['samples'][s]['done']]
    if not active:
        return 0
    turn_base = max((t['turn'] for st_ in state['samples'] for t in st_['traj']), default=-1) + 1

    # history: at segment start compress to best window; new turns append within segment
    msgs = {}
    for s in active:
        traj = state['samples'][s]['traj']
        idxs = best_window(traj, args.remain_turns)
        msgs[s] = history_from_window(prompt, traj, idxs)

    # generation: turn numbering global per task; first new turn batched when all
    # samples share identical history only happens post-import (windows differ),
    # so issue per-sample parallel requests routed across servers.
    new_turns = 0
    for k in range(budget):
        turn = turn_base + k
        samples = {}
        with ThreadPoolExecutor(max_workers=max(1, n)) as ex:
            futs = {s: ex.submit(router.chat, msgs[s],
                                 args.seed + task_index * 1000 + seg_no * 50 + turn * 8 + s, 1)
                    for s in active}
            for s, f in futs.items():
                try:
                    outs, _ = f.result()
                    samples[s] = outs[0]
                except Exception as e:
                    samples[s] = {'text': '', 'finish_reason': f'error:{e}'}
        # evaluate
        evals = {}
        with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
            futs = {}
            for s in active:
                code = C.extract_modelnew(samples[s]['text'])
                sd = task_dir / f's{s}t{turn}'
                sd.mkdir(parents=True, exist_ok=True)
                (sd / 'response.txt').write_text(samples[s]['text'])
                ok, viol = (False, ['no complete codeblock extracted'])
                if code:
                    (sd / 'modelnew.py').write_text(code)
                    source = C.solution_source(code)
                    (sd / 'solution.py').write_text(source)
                    ok, viol = C.compliance_check(code)
                evals[s] = {'code_ok': code is not None, 'compliant': ok, 'violations': viol,
                            'dir': sd}
                if code and ok:
                    futs[s] = ex.submit(C.eval_candidate, pool, args, task, source, sd, 'full')
            for s in active:
                if s in futs:
                    evals[s]['status'] = futs[s].result()

        for s in active:
            ev = evals[s]
            if not ev['code_ok']:
                fb = C.feedback_text(False, None)
                reward = 0.0
                info, status = {}, None
            elif not ev['compliant']:
                fb = ('Your submission violates the TRITON-ONLY requirement and was not '
                      'evaluated:\n- ' + '\n- '.join(ev['violations'][:6]) +
                      '\nRewrite ModelNew so that all computation happens in @triton.jit '
                      'kernels launched by forward.')
                reward, info, status = 0.0, {}, None
            else:
                reward, info = C.reward_of(ev['status'])
                status = ev['status'].get('result')
                fb = C.feedback_text(True, ev['status'])
                sm = state['samples'][s]
                if info.get('correct_ratio', 0) >= 1.0 and reward > (sm['best_reward_seen'] or 0):
                    sm['best'] = [reward, info.get('geomean'), turn,
                                  (ev['dir'] / 'solution.py').read_text()]
                    sm['best_reward_seen'] = reward
            state['samples'][s]['traj'].append({
                'turn': turn, 'code_ok': ev['code_ok'], 'compliant': ev['compliant'],
                'violations': ev['violations'][:6], 'reward': reward, 'info': info,
                'status': status, 'response': tc(samples[s]['text']),
                'feedback': fb[:1500],
                'source': (ev['dir'] / 'solution.py').read_text() if ev['code_ok'] else None})
            # append this turn to the segment history (window fixed at segment start)
            msgs[s] += [{'role': 'assistant', 'content': state['samples'][s]['traj'][-1]['response']},
                        {'role': 'user', 'content': fb}]
        new_turns += 1
        state['seg_done_turns'] = k + 1
        correct = sum(1 for s in active if (evals[s].get('status') or {}).get('result', {}).get('valid'))
        log(f'  seg{seg_no} turn+{k + 1}: {correct}/{len(active)} valid')
        (task_dir / 'state.json').write_text(json.dumps(state, default=str))
    return new_turns


def run_task(pool, router, args, task, task_index):
    key = task['key']
    task_dir = RUN_ROOT / 'tasks' / key
    task_dir.mkdir(parents=True, exist_ok=True)
    prompt, n_wl = C.build_prompt(task)
    (task_dir / 'prompt.txt').write_text(prompt)
    log(f'=== STTS task {task_index + 1}: {key} ({n_wl} wl) ===')
    C.precompute_refs(pool, args, task, task_dir)

    state_path = task_dir / 'state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = import_v2_state(task, task_dir, prompt, args.samples)
        if state is None:
            state = {'prompt': prompt, 'seg_new_turns': 0, 'iteration': 1, 'samples': [
                {'traj': [], 'best': None, 'best_reward_seen': 0.0, 'patience': 0, 'done': False}
                for _ in range(args.samples)]}
        state_path.write_text(json.dumps(state, default=str))

    for seg in range(state['iteration'], args.iterations + 1):
        state['iteration'] = seg
        active_before = [s for s in range(len(state['samples'])) if not state['samples'][s]['done']]
        if not active_before:
            break
        best_before = [sm['best_reward_seen'] for sm in state['samples']]
        new_turns = run_segment(pool, router, args, task, task_dir, state, task_index, seg)
        state['seg_new_turns'] = 0  # next segment gets full budget
        state['seg_done_turns'] = 0
        # patience update + window compression is implicit (history built from window)
        for s in range(len(state['samples'])):
            sm = state['samples'][s]
            if sm['done'] or s not in active_before:
                continue
            if (sm['best_reward_seen'] or 0) > (best_before[s] or 0) + 1e-9:
                sm['patience'] = 0
            else:
                sm['patience'] += 1
                if args.patience > 0 and sm['patience'] >= args.patience:
                    sm['done'] = True
        done_n = sum(1 for sm in state['samples'] if sm['done'])
        log(f'  iter {seg}/{args.iterations}: +{new_turns} turns, done {done_n}/{len(state["samples"])}')
        state_path.write_text(json.dumps(state, default=str))

    # summarize: pass@1 over best-of-history; task best -> official fresh eval
    n = len(state['samples'])
    pass1 = sum(1 for sm in state['samples'] if sm['best'])
    geos = [(sm['best'][1] if sm['best'] else None) for sm in state['samples']]
    summary = {'task': key, 'workloads': n_wl, 'pass_at_1': pass1 / n,
               'best_of_history_geomean': max((g for g in geos if g), default=None),
               'samples': [{'best': sm['best'], 'turns': len(sm['traj']),
                            'final_patience': sm['patience']} for sm in state['samples']]}
    task_best = None
    for sm in state['samples']:
        if sm['best'] and (task_best is None or sm['best'][0] > task_best[0]):
            task_best = sm['best']
    if task_best:
        corr, perf = C.final_official(pool, args, task, task_best[3], task_dir / 'final')
        summary['final_correctness'] = {'valid': (corr.get('result') or {}).get('valid')}
        if perf is not None:
            summary['final_geomean_speedup'] = (perf.get('result') or {}).get('geomean_speedup')
        (task_dir / 'best_solution.py').write_text(task_best[3])
    (task_dir / 'summary.json').write_text(json.dumps(summary, default=str))
    with RESULTS_LOCK, open(RUN_ROOT / 'results.jsonl', 'a') as fh:
        fh.write(json.dumps({k: v for k, v in summary.items() if k != 'samples'},
                            default=str) + '\n')
    log(f'=== STTS done: {key} pass@1={summary["pass_at_1"]:.2f} '
        f'final={summary.get("final_geomean_speedup")}')


def main():
    args = parse_args()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    tasks = C.load_tasks(args.task)
    log(f'STTS start: {len(tasks)} tasks, iters={args.iterations} turns/seg={args.max_turns} '
        f'keep={args.remain_turns} patience={args.patience} servers={args.servers}')
    pool = C.GPUPool([int(g) for g in args.gpus.split(',')], args.max_parallel, args.poll_seconds)
    router = Router(args.servers, args)

    def safe(item):
        i, task = item
        try:
            run_task(pool, router, args, task, i)
        except Exception as e:
            log(f'TASK FAILED {task["key"]}: {type(e).__name__}: {e}')

    todo = [(i, t) for i, t in enumerate(tasks)
            if not (RUN_ROOT / 'tasks' / t['key'] / 'summary.json').exists()]
    with ThreadPoolExecutor(max_workers=args.concurrent_tasks) as ex:
        list(ex.map(safe, todo))
    log('STTS complete -> ' + str(RUN_ROOT / 'results.jsonl'))


if __name__ == '__main__':
    main()
