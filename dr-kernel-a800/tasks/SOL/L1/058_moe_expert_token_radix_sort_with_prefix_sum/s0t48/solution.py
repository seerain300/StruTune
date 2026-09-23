import torch
import triton
import triton.language as tl


@triton.jit
def bitonic_sort_stable_kernel(flat_ptr, out_idx_ptr, N, LOGN: tl.constexpr):
    """
    Stable bitonic sort on flat_ptr (int32), length N.
    out_idx_ptr stores original positions 0..N-1 as int64, in sorted order.
    """
    pid = tl.program_id(axis=0)  # program id for element
    i = pid
    # Initialize: each element's original position is i
    tl.store(out_idx_ptr + i, tl.full((), i, dtype=tl.int64))

    # Bitonic sort network
    for k in range(1, LOGN + 1):
        j = 1 << k
        for t in range(k - 1, -1, -1):
            step = 1 << t
            p = 1 << (t + 1)
            # partner index
            partner = i ^ step
            # Load current values and original positions
            val_i = tl.load(flat_ptr + i)
            val_p = tl.load(flat_ptr + partner)
            idx_i = tl.load(out_idx_ptr + i)
            idx_p = tl.load(out_idx_ptr + partner)
            # Ascending when (i & p) == 0, else descending
            asc = ( (i & p) == 0 )
            # Compare and decide swap
            less = val_i < val_p
            equal = val_i == val_p
            # For stable tie-break, prefer lower original index
            tie = (i < partner)
            # If ascending: swap when val_i > val_p or (val_i == val_p and i > partner)
            # If descending: swap when val_i < val_p or (val_i == val_p and i < partner)
            swap = tl.where(asc,
                (val_i > val_p) | (equal & (i > partner)),
                (val_i < val_p) | (equal & (i < partner))
            )
            new_idx_i = tl.where(swap, idx_p, idx_i)
            new_idx_p = tl.where(swap, idx_i, idx_p)
            # Only process each pair once
            if (i < partner):
                tl.store(out_idx_ptr + i, new_idx_i)
                tl.store(out_idx_ptr + partner, new_idx_p)


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
    Single-program sequential prefix sum for simplicity and correctness.
    """
    pid = tl.program_id(axis=0)
    # Only one program should run this kernel.
    acc = tl.zeros((), dtype=tl.int64)
    for i in range(0, num_experts):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + (i + 1), acc)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        # Ensure input is on CUDA
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.cuda()

        # Flatten to 1D
        flat = topk_idx.reshape(-1)  # int32
        N = flat.numel()

        # Determine LOGN for bitonic sort
        # We need LOGN = ceil(log2(max(1, N))). Use 16 which covers N up to 65536.
        LOGN = 16

        # Allocate output for sorted indices (int64)
        sorted_token_indices = torch.empty(N, dtype=torch.long, device=flat.device)

        # Launch bitonic sort kernel
        grid_sort = (N,)
        bitonic_sort_stable_kernel[grid_sort](flat, sorted_token_indices, N, LOGN, num_warps=1)

        # Compute histogram counts using Triton
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=flat.device)
        grid_h = (triton.cdiv(N, 1024),)
        count_histogram_kernel[grid_h](flat, counts, N, num_experts=self.num_experts, num_warps=1)

        # Compute inclusive prefix sum for offsets (int64), then cast to int32
        offsets64 = torch.empty(self.num_experts + 1, dtype=torch.int64, device=flat.device)
        offsets64[0] = 0  # inclusive prefix starts at 0
        grid_p = (1,)
        prefix_sum_kernel[grid_p](counts, offsets64, num_experts=self.num_experts, num_warps=1)
        expert_offsets = offsets64.to(torch.int32)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
