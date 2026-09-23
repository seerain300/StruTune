"""
Solution c001 — L1/058 MoE Expert Token Radix Sort with Prefix Sum.

Design: Option C from docs/plan.md — expert-parallel ordered gather, 2 Triton launches.

Semantics reproduced EXACTLY (stable counting sort on bounded keys 0..255 + prefix sum):
  flat = topk_idx.reshape(-1)                    # int32, values 0..255, length N
  expert_offsets[0]        = 0
  expert_offsets[e]        = #{ i : flat[i] < e }        (exclusive prefix of histogram)
  expert_offsets[256]      = N
  sorted_token_indices[expert_offsets[flat[i]] + rank_i] = i
    where rank_i = #{ j < i : flat[j] == flat[i] }       (stable, ascending original order)

K1 (grid=(1,)): single program, tile-loop histogram (tl.histogram) over 256 bins,
                inclusive cumsum -> expert_offsets. Masked tail lanes loaded as 0 and
                corrected out of bin 0 via PAD.
K2 (grid=(256,)): program e streams all N in tiles, compacts matching source indices
                consecutively from expert_offsets[e] using a carried exclusive cumsum.
                Ascending tiles + ascending within-tile scan => stable by construction.

Triton-only compute. Torch used solely for reshape, output allocation, and grid math.
"""

import torch
import triton
import triton.language as tl

NUM_EXPERTS = 256


@triton.jit
def _hist_prefix_kernel(flat_ptr, offsets_ptr, N, NUM_TILES, PAD, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    counts = tl.zeros([NUM_EXPERTS], dtype=tl.int32)
    for i in range(NUM_TILES):
        idx = i * BLOCK + offs
        mask = idx < N
        x = tl.load(flat_ptr + idx, mask=mask, other=0)
        counts += tl.histogram(x, NUM_EXPERTS)
    # masked (padding) lanes were loaded as 0 -> subtract them out of bin 0
    ebins = tl.arange(0, NUM_EXPERTS)
    counts = counts - (ebins == 0).to(tl.int32) * PAD
    # inclusive cumsum -> expert_offsets[1:257]; expert_offsets[0] = 0
    inclusive = tl.cumsum(counts, axis=0)
    tl.store(offsets_ptr + 1 + ebins, inclusive)
    tl.store(offsets_ptr, 0)


@triton.jit
def _scatter_kernel(flat_ptr, out_ptr, offsets_ptr, N, NUM_TILES, BLOCK: tl.constexpr):
    e = tl.program_id(0)
    base = tl.load(offsets_ptr + e)
    offs = tl.arange(0, BLOCK)
    running = 0
    for i in range(NUM_TILES):
        idx = i * BLOCK + offs
        mask = idx < N
        x = tl.load(flat_ptr + idx, mask=mask, other=-1)
        is_e = (x == e).to(tl.int32)
        incl = tl.cumsum(is_e, axis=0)          # inclusive rank within tile
        excl = incl - is_e                       # exclusive rank within tile
        pos = base + running + excl
        write_mask = mask & (is_e == 1)
        tl.store(out_ptr + pos, idx.to(tl.int32), mask=write_mask)
        running += tl.sum(is_e, axis=0)


@torch.no_grad()
def run(topk_idx: torch.Tensor):
    flat = topk_idx.reshape(-1)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    N = flat.numel()

    sorted_token_indices = torch.empty(N, dtype=torch.int32, device=flat.device)
    expert_offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=flat.device)

    BLOCK = 1024
    num_tiles = triton.cdiv(N, BLOCK)
    pad = num_tiles * BLOCK - N

    _hist_prefix_kernel[(1,)](flat, expert_offsets, N, num_tiles, pad, BLOCK=BLOCK, num_warps=4)
    _scatter_kernel[(NUM_EXPERTS,)](flat, sorted_token_indices, expert_offsets, N, num_tiles,
                                    BLOCK=BLOCK, num_warps=4)

    return sorted_token_indices, expert_offsets
