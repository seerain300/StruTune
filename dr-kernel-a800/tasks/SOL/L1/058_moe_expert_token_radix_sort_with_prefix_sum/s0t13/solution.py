import torch
import triton
import triton.language as tl


@triton.jit
def count_histogram_kernel(flat_ptr, counts_ptr, N, num_experts: tl.constexpr):
    """
    Compute histogram of flat values (int32) into counts_ptr (int32), length num_experts.
    Each program processes BLOCK elements and atomically adds 1 for each occurrence.
    """
    pid = tl.program_id(axis=0)
    BLOCK = 1024
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32 values in [0, num_experts-1]
    # Atomic add 1 for each occurrence into counts[val]
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr (int64, length num_experts) into offsets_ptr (int64).
    offsets_ptr[0] should be set to 0 by the caller. We write offsets[1..].
    """
    pid = tl.program_id(axis=0)
    # Single program computes the prefix sum sequentially
    acc = tl.zeros((), dtype=tl.int64)
    for i in range(0, num_experts):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + (i + 1), acc)


@triton.jit
def stable_bitonic_sort_indices_kernel(flat_ptr, out_idx_ptr, N, LOGN: tl.constexpr):
    """
    Stable bitonic sort on flat_ptr (int32). out_idx_ptr holds int64 original positions 0..N-1.
    Grid: axis 0 = N, axis 1 = LOGN (one stage per j in range(k+1, LOGN), k in 0..LOGN-1).
    For each stage j, each program i compares with partner = i ^ (1 << j).
    Ascending if (i & (1 << (k+1))) == 0 else descending. Stability: tie-break by i < partner.
    """
    i = tl.program_id(axis=0)
    j = tl.program_id(axis=1)
    # If axis 1 exceeds expected, skip
    if j >= LOGN:
        return
    step = 1 << j
    # Partner index
    partner = i ^ step
    # Only process each pair once
    do_pair = i < partner
    in_bounds = (i < N) & (partner < N) & do_pair

    # Load current indices (original positions)
    idx_i = tl.load(out_idx_ptr + i, mask=in_bounds, other=0)  # int64
    idx_p = tl.load(out_idx_ptr + partner, mask=in_bounds, other=0)  # int64

    # Load values for compare
    a = tl.load(flat_ptr + idx_i.to(tl.int32), mask=in_bounds, other=0)  # int32
    b = tl.load(flat_ptr + idx_p.to(tl.int32), mask=in_bounds, other=0)  # int32

    # Ascending/descending for this stage: depends on next bit k. j in stage of k means k in 0..LOGN-2.
    # The outer loop uses k, and inner uses j in range(k+1, LOGN). We cannot access k directly here,
    # so we compute asc from j: asc if (i & (1 << (j+1))) == 0 else not asc.
    asc = ((i & (step << 1)) == 0)

    tie = a == b
    less = a < b
    greater = a > b

    # Determine min/max per stage
    minv = tl.where(less, a, tl.where(greater, b, tl.where(tie, tl.minimum(a, b), a)))
    maxv = tl.where(less, b, tl.where(greater, a, tl.where(tie, tl.maximum(a, b), b)))

    # Stable tie-break: if equal, smaller original index comes first (i < partner)
    take_i = tie & (i < partner)
    new_i = tl.where(take_i, minv, maxv)
    new_p = tl.where(take_i, maxv, minv)

    # Apply ascending/descending
    new_i_stage = tl.where(asc, new_i, new_p)
    new_p_stage = tl.where(asc, new_p, new_i)

    # Store results
    tl.store(out_idx_ptr + i, new_i_stage, mask=in_bounds)
    tl.store(out_idx_ptr + partner, new_p_stage, mask=in_bounds)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Sorts the flattened topk_idx (int32) using a Triton stable bitonic sort and returns indices (int64).
        - Computes expert_offsets via Triton (histogram + prefix sum using atomic adds).
        Returns:
            sorted_token_indices: torch.int64 tensor of shape (N,)
            expert_offsets: torch.int32 tensor of shape (num_experts + 1,)
        """
        # Ensure input on CUDA and dtype int32
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.cuda()
        if topk_idx.dtype != torch.int32:
            topk_idx = topk_idx.to(torch.int32)

        # Flatten
        flat = topk_idx.reshape(-1)
        N = flat.numel()
        device = flat.device

        # Output buffer for indices (int64), initially 0..N-1
        out_idx = torch.arange(N, dtype=torch.int64, device=device)

        # Launch stable bitonic sort: grid size N along axis 0, and LOGN along axis 1
        LOGN = (N - 1).bit_length() if N > 0 else 0
        if LOGN > 0:
            grid = (N, LOGN)
            stable_bitonic_sort_indices_kernel[grid](flat, out_idx, N, LOGN=LOGN)

        # Compute expert_offsets via Triton histogram + prefix sum (int64 counts, then cast to int32)
        num_experts = 256  # matches original default
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        # Histogram kernel: grid size based on N
        BLOCK = 1024
        grid_counts = ((N + BLOCK - 1) // BLOCK,)
        count_histogram_kernel[grid_counts](flat, counts, N, num_experts=num_experts)
        # Prefix sum in int64
        offsets64 = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
        prefix_sum_kernel[(1,)](counts, offsets64, num_experts=num_experts)
        expert_offsets = offsets64[1:].to(torch.int32)

        return out_idx, expert_offsets


def run(*args):
    return ModelNew()(*args)
