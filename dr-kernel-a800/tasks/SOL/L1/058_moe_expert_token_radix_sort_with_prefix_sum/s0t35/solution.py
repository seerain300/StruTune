import math
import torch
import triton
import triton.language as tl


@triton.jit
def count_histogram_kernel(flat_ptr, counts_ptr, N, num_experts: tl.constexpr):
    """
    Triton kernel: compute histogram of flat values (int32) into counts_ptr (int32),
    length num_experts. Each program processes BLOCK elements and atomically adds 1
    for each occurrence.
    """
    pid = tl.program_id(axis=0)
    BLOCK = 1024
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32
    # Atomic add 1 for each valid occurrence
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Triton kernel: compute inclusive prefix sum of counts_ptr (int32) into offsets_ptr (int32).
    Assumes offsets_ptr[0] is already 0. We write offsets[1..].
    """
    pid = tl.program_id(axis=0)  # single program does the job
    acc = tl.zeros((), dtype=tl.int32)
    for i in range(0, num_experts):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + (i + 1), acc)


@triton.jit
def stable_bitonic_sort_kernel(flat_ptr, out_idx_ptr, N, LOGN: tl.constexpr):
    """
    Triton kernel: stable bitonic sort on flat_ptr (int32), length N.
    out_idx_ptr holds int64 original positions 0..N-1. We perform compare-and-swap
    at each bitonic stage j for pairs (i, partner = i ^ (1 << j)). We only process
    i < partner to avoid double updates. Ascending if (i & (1 << (k+1))) == 0,
    else descending. For ties, lower original index (i < partner) comes first.
    """
    i = tl.program_id(axis=0)
    if i >= N:
        return

    # Initialize indices: each position starts as its own index (int64)
    idx = tl.full((), i, tl.int64)
    tl.store(out_idx_ptr + i, idx)

    # Bitonic sort stages: j in range(k+1, LOGN) for k in 0..LOGN-1
    for k in range(0, LOGN):
        for j in range(k + 1, LOGN):
            partner = i ^ (1 << j)
            # Only process each pair once
            if i < partner:
                # Load values and current indices
                val_i = tl.load(flat_ptr + i)
                val_p = tl.load(flat_ptr + partner)
                idx_i = tl.load(out_idx_ptr + i)
                idx_p = tl.load(out_idx_ptr + partner)

                # Ascending/descending direction for this stage
                asc = ((i & (1 << (k + 1))) == 0)

                # Compare-and-swap for this pair
                cmp = val_i - val_p
                swap_asc = cmp > 0
                swap_desc = cmp < 0
                swap = tl.where(asc, swap_asc, swap_desc)

                # Tie-break for stability: if equal values, lower original index comes first
                equal = cmp == 0
                lower_prefers_asc = (i < partner)  # True when tie-break prefers lower index in ascending context

                if swap or (equal and lower_prefers_asc):
                    # Swap indices for position i and partner
                    tmp = idx_i
                    idx_i = idx_p
                    idx_p = tmp

                # Store back
                tl.store(out_idx_ptr + i, idx_i)
                tl.store(out_idx_ptr + partner, idx_p)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized forward that matches original behavior:
        - sorted_token_indices: int64, shape (N,)
        - expert_offsets: int32, shape (num_experts+1,)
        """
        # Ensure device is CUDA and data is contiguous
        assert topk_idx.is_cuda, "topk_idx must be on CUDA for Triton kernels."
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()

        # Output buffers
        sorted_token_indices = torch.empty(N, dtype=torch.int64, device=flat.device)
        # Initialize with identity mapping (will be overwritten by Triton sort)
        torch.arange(0, N, out=sorted_token_indices)

        # num_experts default: 256
        num_experts = 256
        # Compute LOGN = ceil(log2(N)) as constexpr for Triton
        LOGN = max(1, int(math.ceil(math.log2(max(1, N)))))

        # Launch stable bitonic sort in Triton
        grid = (N, LOGN)
        stable_bitonic_sort_kernel[grid](flat, sorted_token_indices, N, LOGN, num_warps=1)

        # Compute expert_offsets using Triton histogram and prefix sum
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        BLOCK = 1024
        grid_hist = (triton.cdiv(N, BLOCK),)
        count_histogram_kernel[grid_hist](flat, counts, N, num_experts)

        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        offsets[0] = 0
        grid_ps = (1,)
        prefix_sum_kernel[grid_ps](counts, offsets, num_experts)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
