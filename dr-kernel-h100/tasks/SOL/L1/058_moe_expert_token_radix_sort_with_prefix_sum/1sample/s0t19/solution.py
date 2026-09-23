import torch
import triton
import triton.language as tl


@triton.jit
def odd_even_sort_values_and_indices_kernel(values_ptr, indices_ptr, N: tl.int32, BLOCK: tl.constexpr):
    # We use a single program instance with BLOCK lanes; each lane processes elements in its offset.
    # This odd-even transposition sort is simple and correct for any N.
    # We preserve stability by not swapping on ties.
    offs = tl.arange(0, BLOCK)
    # We will iterate for N phases; each phase performs either even or odd pair comparisons.
    # Note: For correctness across all N, we run N phases. For performance, we could mask to log2(N) phases,
    # but to avoid complexity and ensure correctness, we use N phases here.
    # Triton while loops need static bounds; we iterate t from 0 to N-1.
    for t in range(0, N):
        # Compute phase: even or odd
        is_even = (t % 2 == 0)
        # Stride for this phase
        stride = 1 if is_even else 0
        # Partner index
        j = offs + (1 - stride)
        # Mask for valid lanes
        mask_i = offs < N
        mask_j = j < N
        mask_pair = mask_i & mask_j

        # Load current values and indices
        vi = tl.load(values_ptr + offs, mask=mask_i, other=0)
        vj = tl.load(values_ptr + j, mask=mask_j, other=0)
        ii = tl.load(indices_ptr + offs, mask=mask_i, other=0)
        ij = tl.load(indices_ptr + j, mask=mask_j, other=0)

        # Decide swap: ascending order, stable (do not swap on equal)
        swap = mask_pair & (vi > vj)

        # Compute new values and new indices after potential swap
        new_vi = tl.where(swap, vj, vi)
        new_vj = tl.where(swap, vi, vj)
        new_ii = tl.where(swap, ij, ii)
        new_ij = tl.where(swap, ii, ij)

        # Store back
        tl.store(values_ptr + offs, new_vi, mask=mask_i)
        tl.store(values_ptr + j, new_vj, mask=mask_j)
        tl.store(indices_ptr + offs, new_ii, mask=mask_i)
        tl.store(indices_ptr + j, new_ij, mask=mask_j)


@triton.jit
def histogram_atomic_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    # Each program instance processes BLOCK elements; counts_ptr has length 256.
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # Load flat values (int32). If any masked, use 0 (they won't contribute).
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
    # Atomic add to counts for each value in [0..255]. Only add for valid offsets.
    # Triton's atomic_add expects tl.int32 pointer.
    # Note: counts_ptr is contiguous, and we index by vals directly (vals are in [0,255]).
    # We need to ensure we only atomic_add for lanes that loaded (mask).
    # Triton does not support vectorized pointer indexing like counts_ptr[vals], so we loop per bin.
    # A more idiomatic approach: per-element atomic add to counts[vals].
    for v in range(0, 256):
        # For each bin v, we find lanes where vals == v and atomic_add 1
        eq = (vals == v) & mask
        # eq is boolean; cast to int for accumulation
        inc = eq.to(tl.int32)
        # Atomic add to counts[v]
        tl.atomic_add(counts_ptr + v, inc.sum())


@triton.jit
def inclusive_scan_prefix_sum_kernel(counts_ptr, offsets_ptr, M: tl.constexpr):
    # Compute inclusive prefix sum over counts_ptr[0..M-1] and write to offsets_ptr[0..M].
    # We initialize offsets[0] = 0 on host, so here we compute offsets[1..M].
    # This is a simple sequential scan inside one program instance.
    acc = 0
    for i in range(0, M):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + i + 1, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # Ensure topk_idx is on CUDA for Triton kernels
        assert topk_idx.is_cuda, "Input topk_idx must be on CUDA device for Triton kernels."
        # Flatten to 1D
        flat = topk_idx.reshape(-1)  # int32 on GPU
        N = flat.numel()
        device = flat.device

        # 1) Triton sort: odd-even transposition sort
        # Prepare working copies: values (for sorting) and indices (permutation).
        values = flat.clone()  # int32 tensor
        indices = torch.arange(N, dtype=torch.int32, device=device)  # original positions 0..N-1

        # Choose a BLOCK size; 1024 works well for typical N. We use a single program instance with BLOCK lanes.
        BLOCK = 1024
        # The kernel will iterate N phases to ensure correctness for any N.
        odd_even_sort_values_and_indices_kernel[(1,)](values, indices, N=N, BLOCK=BLOCK, num_warps=8)

        # After sorting, indices hold the permutation (sorted positions). Return as int32.
        sorted_token_indices = indices

        # 2) Histogram via Triton (counts per value in [0..255])
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        grid_hist = (triton.cdiv(N, BLOCK),)
        histogram_atomic_kernel[grid_hist](values, counts, N=N, BLOCK=BLOCK, num_warps=8)

        # 3) Prefix sum (offsets) via Triton
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        offsets[0] = 0
        inclusive_scan_prefix_sum_kernel[(1,)](counts, offsets, M=256, num_warps=1)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
