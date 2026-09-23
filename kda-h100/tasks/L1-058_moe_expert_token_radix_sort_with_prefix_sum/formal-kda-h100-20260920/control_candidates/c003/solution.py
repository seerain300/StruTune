"""
Solution c003 (staged) — L1/058 MoE Expert Token Radix Sort with Prefix Sum.

Design: hardened Option C (expert-parallel ordered gather), 2 Triton launches.
Fixes the confirmed c001/c002 compile failure: Triton 3.5.0 forbids reading a plain
python module global (NUM_EXPERTS) inside a @jit kernel. NUM_EXPERTS is now passed as a
tl.constexpr kernel argument; no @jit kernel reads any bare python global.

Semantics reproduced EXACTLY (stable counting sort on bounded keys 0..255 + prefix sum):
  flat = topk_idx.reshape(-1)                    # int32, values 0..255, length N
  expert_offsets[e] = #{ i : flat[i] < e }       (exclusive prefix of histogram)
    -> expert_offsets[0]=0, expert_offsets[256]=N
  sorted_token_indices[expert_offsets[flat[i]] + rank_i] = i
    where rank_i = #{ j < i : flat[j] == flat[i] } (stable, ascending original order)

K1 _count_kernel (grid=cdiv(N,BLOCK)): atomic histogram of flat into counts[256].
    Counting is order-independent, so atomics are correct here (atomics are NOT used to
    place indices, which must stay ordered).
K2 _prefix_scatter_kernel (grid=257): program e loads counts[256], computes exclusive
    prefix base_e = sum_{j<e} counts[j], stores expert_offsets[e]=base_e, then (for
    e<256; a no-op for e==256) streams all N in constexpr tiles, compacting matching
    source indices consecutively from base_e via a carried exclusive cumsum. Ascending
    tiles + ascending within-tile scan => stable.

Triton-only compute. Torch used solely for reshape, output/scratch allocation, grid math.
"""

import torch
import triton
import triton.language as tl

NUM_EXPERTS = 256


@triton.jit
def _count_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N
    x = tl.load(flat_ptr + idx, mask=mask, other=0)
    tl.atomic_add(counts_ptr + x, 1, mask=mask)


@triton.jit
def _prefix_scatter_kernel(flat_ptr, counts_ptr, out_ptr, offsets_ptr, N,
                           NE: tl.constexpr, NUM_TILES: tl.constexpr, BLOCK: tl.constexpr):
    e = tl.program_id(0)  # 0 .. NE (inclusive) -> NE+1 programs
    ebins = tl.arange(0, NE)
    counts = tl.load(counts_ptr + ebins)                       # [NE] int32
    base = tl.sum(tl.where(ebins < e, counts, 0), axis=0)      # exclusive prefix (scalar)
    tl.store(offsets_ptr + e, base)                            # expert_offsets[e]

    offs = tl.arange(0, BLOCK)
    running = tl.zeros([1], tl.int32)                          # matches seen in prior tiles
    for i in range(NUM_TILES):
        idx = i * BLOCK + offs
        m = idx < N
        x = tl.load(flat_ptr + idx, mask=m, other=-1)
        is_e = (x == e).to(tl.int32)
        excl = tl.cumsum(is_e, axis=0) - is_e                  # within-tile exclusive rank
        pos = base + running + excl
        wmask = m & (is_e == 1)
        tl.store(out_ptr + pos, idx.to(tl.int32), mask=wmask)  # e==NE -> wmask all False
        running += tl.sum(is_e, axis=0)


@torch.no_grad()
def run(topk_idx: torch.Tensor):
    flat = topk_idx.reshape(-1)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    N = flat.numel()
    device = flat.device

    sorted_token_indices = torch.empty(N, dtype=torch.int32, device=device)
    expert_offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=device)
    counts = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=device)

    BLOCK = 1024
    count_grid = triton.cdiv(N, BLOCK)
    num_tiles = triton.cdiv(N, BLOCK)

    _count_kernel[(count_grid,)](flat, counts, N, BLOCK=BLOCK, num_warps=4)
    _prefix_scatter_kernel[(NUM_EXPERTS + 1,)](
        flat, counts, sorted_token_indices, expert_offsets, N,
        NE=NUM_EXPERTS, NUM_TILES=num_tiles, BLOCK=BLOCK, num_warps=4,
    )

    return sorted_token_indices, expert_offsets
