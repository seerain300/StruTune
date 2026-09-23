"""
Solution for L1/058 MoE Expert Token Radix Sort with Prefix Sum.

Target: NVIDIA A800 (sm_80, Ampere). Implementation is Triton; PyTorch is used
only for tensor allocation / metadata / launch plumbing (no computational fallback).

Reference semantics reproduced EXACTLY (integer, exact-match):
    flat = topk_idx.reshape(-1)                      # C-order, length N, values in [0,255]
    _, sorted_token_indices = flat.sort(stable=True) # stable argsort by expert id
    expert_offsets[0]   = 0
    expert_offsets[e+1] = sum_{j<=e} count[j]        # inclusive prefix of the histogram

Pipeline (candidate c001):
    K1  hist_kernel    : atomic 256-bin histogram of flat            (grid = ceil(N/BLOCK))
    K2  prefix_kernel  : inclusive prefix sum -> expert_offsets[1:]  (grid = 1)
    K3  scatter_kernel : expert-parallel stable scatter (Scheme S1)  (grid = 256)

Stability (decisive): in K3 each program owns one expert e and scans `flat` in
ascending tile order; within a tile the exclusive cumsum of the (v==e) mask gives
each matched lane its rank, and lanes carry ascending flat indices. So matched
indices are written in strictly ascending order -> matches torch.sort(stable=True).
Each program writes only into its disjoint slice [base, base+count) -> no output
atomics, no write races, fully deterministic.
"""

import torch
import triton
import triton.language as tl

NUM_EXPERTS = 256


@triton.jit
def hist_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < N
    v = tl.load(flat_ptr + off, mask=mask, other=0)
    # Commutative counting -> order-independent -> stability-safe.
    tl.atomic_add(counts_ptr + v, 1, mask=mask)


@triton.jit
def prefix_kernel(counts_ptr, offsets_ptr, E: tl.constexpr):
    e = tl.arange(0, E)
    c = tl.load(counts_ptr + e)
    inc = tl.cumsum(c, axis=0)            # inclusive prefix, inc[e] = sum_{j<=e} c[j]
    tl.store(offsets_ptr + 1 + e, inc)    # offsets[0] left as 0 (pre-zeroed alloc)


@triton.jit
def scatter_kernel(flat_ptr, offsets_ptr, out_ptr, N, BLOCK: tl.constexpr):
    e = tl.program_id(0)                  # this program owns expert e
    base = tl.load(offsets_ptr + e)       # exclusive prefix = write base for expert e
    running = tl.zeros((), dtype=tl.int32)
    num_tiles = tl.cdiv(N, BLOCK)
    for t in range(0, num_tiles):
        off = t * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        v = tl.load(flat_ptr + off, mask=mask, other=-1)
        m = (v == e) & mask               # masked/tail lanes never match (other=-1)
        mi = m.to(tl.int32)
        excl = tl.cumsum(mi, axis=0) - mi  # exclusive within-tile rank of matches
        pos = base + running + excl
        tl.store(out_ptr + pos, off.to(tl.int32), mask=m)
        running += tl.sum(mi)


@torch.no_grad()
def run(topk_idx: torch.Tensor):
    # Plumbing only: C-contiguous flatten to match reshape(-1).
    flat = topk_idx.reshape(-1).contiguous()
    if flat.dtype != torch.int32:
        flat = flat.to(torch.int32)
    N = flat.numel()
    device = flat.device

    counts = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=device)
    expert_offsets = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32, device=device)
    sorted_token_indices = torch.empty(N, dtype=torch.int32, device=device)

    BLOCK_H = 1024
    BLOCK_S = 1024

    hist_kernel[(triton.cdiv(N, BLOCK_H),)](
        flat, counts, N, BLOCK=BLOCK_H, num_warps=4
    )
    prefix_kernel[(1,)](
        counts, expert_offsets, E=NUM_EXPERTS, num_warps=4
    )
    scatter_kernel[(NUM_EXPERTS,)](
        flat, expert_offsets, sorted_token_indices, N, BLOCK=BLOCK_S, num_warps=4
    )

    return sorted_token_indices, expert_offsets
