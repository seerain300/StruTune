#!/usr/bin/env python3
"""逐题人工审计：提取 run() 内库调用的源码上下文，判断哪些是'用了库才有主要提升'。"""
import json, ast, glob, re
from pathlib import Path

LIB = {'matmul','mm','bmm','addmm','einsum','linear','conv1d','conv2d','conv3d',
       'conv_transpose1d','conv_transpose2d','conv_transpose3d',
       'rfft','fft','fftn','irfft','ifft','hfft','ihfft','rfft2','fft2',
       'cumsum','cumprod','sort','topk','argsort','unique','scatter_add','index_add'}

targets = []
for tag_dir in [
    '/data1/workspace/weihongren/baseline/ksearch/experiments/formal_20260914/*',
    '/data1/workspace/weihongren/baseline/ksearch-sol-execbench/experiments/formal_20260914/*',
    '/data1/workspace/weihongren/baseline/ksearch-sol-execbench/experiments/formal2_solL2_20260916/*',
]:
    for td in sorted(glob.glob(tag_dir)):
        rd = Path(td) / 'run_seed0'
        name = rd.parent.name
        sols = sorted((rd / 'ksearch-artifacts').glob('*/solutions/*/*.json'), key=lambda p: p.stat().st_mtime)
        if not sols:
            continue
        src = json.loads(sols[-1].read_text())
        code = ''.join(s['content'] for s in src.get('sources', []))
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        run_node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run'), None)
        if run_node is None:
            continue
        libs = set()
        for n in ast.walk(run_node):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in LIB:
                libs.add(n.func.attr)
        if libs:
            targets.append((name, code, run_node, libs, tree))

print(f'共 {len(targets)} 题有库调用\n')
for name, code, run_node, libs, tree in targets:
    kernels = [n.name for n in tree.body
               if isinstance(n, ast.FunctionDef)
               and any(isinstance(d, ast.Attribute) and d.attr == 'jit' for d in n.decorator_list)]
    lines = code.splitlines()
    start = run_node.lineno - 1
    end = run_node.end_lineno if hasattr(run_node, 'end_lineno') else min(start + 60, len(lines))
    run_lines = lines[start:end]
    n_launch = sum(1 for l in run_lines if re.search(r'_kernel\[', l))
    print(f'=== {name[:44]} ===')
    print(f'  库: {sorted(libs)} | 内核: {len(kernels)} | launch: {n_launch} | run: {len(run_lines)} 行')
    for i, l in enumerate(run_lines):
        if any(f'.{lib}(' in l for lib in libs):
            lo = max(0, i - 2)
            hi = min(len(run_lines), i + 3)
            for j in range(lo, hi):
                mark = '>>>' if j == i else '   '
                print(f'  {mark} {run_lines[j][:100]}')
            print()
    print()
