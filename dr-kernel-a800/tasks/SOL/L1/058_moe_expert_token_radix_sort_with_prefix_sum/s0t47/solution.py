import math
import torch
import triton
import triton.language as tl


@triton.jit
def stable_bitonic_sort_kernel(flat_ptr, idx_out_ptr, N, LOGN: tl.constexpr):
    """
    Stable bitonic sort of flat_ptr (int32) producing sorted positions in idx_out_ptr (int64).
    Each program i handles its element and participates in bitonic stages. Only pairs with i < partner are updated.
    """
    i = tl.program_id(axis=0)  # element index
    # Bitonic network: for j in 0..LOGN-1, partner = i ^ (1 << j)
    for j in range(0, LOGN):
        partner = tl.bitwise_xor(i, 1 << j)
        # Only process each pair once and within bounds
        mask_pair = (partner < N) & (i < partner)
        # Ascending if the next bit in the bitonic sequence is 0
        asc = (i & (1 << (j + 1))) == 0
        # Load current and partner values
        a = tl.load(flat_ptr + i)
        b = tl.load(flat_ptr + partner)
        # Compare
        cmp_gt = a > b
        cmp_lt = a < b
        cmp_eq = a == b
        # Compute new positions depending on asc/desc and stability
        pos_a = tl.where(asc, i, partner)
        pos_b = tl.where(asc, partner, i)
        # Stability: if equal, lower index comes first
        tie = cmp_eq & (i < partner)
        # If ascending: swap if a > b; if descending: swap if a < b.
        do_swap = tl.where(asc, cmp_gt, cmp_lt)
        i_new = tl.where(do_swap | tie, pos_b, pos_a)
        partner_new = tl.where(do_swap | tie, pos_a, pos_b)
        # Write back only for this pair (i < partner) and within bounds
        tl.store(idx_out_ptr + i, tl.cast(i_new, tl.int64), mask=mask_pair)
        tl.store(idx_out_ptr + partner, tl.cast(partner_new, tl.int64), mask=mask_pair)


@triton.jit
def count_histogram_kernel(flat_ptr, counts_ptr, N, num_experts: tl.constexpr):
    """
    Histogram of int32 flat values into counts_ptr (int32) of length num_experts.
    Each program processes BLOCK elements and atomically adds 1 for each occurrence.
    """
    pid = tl.program_id(axis=0)
    BLOCK = 1024
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32 values in [0, num_experts-1]
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr (int64, length num_experts) into offsets_ptr (int64).
    offsets_ptr[0] should be set to 0 by the caller. We write offsets[1..].
    Single-program sequential prefix sum for simplicity and correctness.
    """
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

        # Flatten to 1D and make contiguous
        flat = topk_idx.reshape(-1).contiguous()  # int32

        N = flat.numel()
        LOGN = int(math.ceil(math.log2(max(N, 1)))) if N > 1 else 1

        # Allocate output indices (int64) and run Triton stable sort
        sorted_token_indices = torch.empty(N, dtype=torch.int64, device=flat.device)
        # Launch bitonic sort: axis 0 over N, axis 1 over LOGN stages
        grid = (N, LOGN)
        stable_bitonic_sort_kernel[grid](flat, sorted_token_indices, N, LOGN, num_warps=1)

        # Histogram in Triton
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=flat.device)
        grid_h = (triton.cdiv(N, 1024),)
        count_histogram_kernel[grid_h](flat, counts, N, num_experts=self.num_experts, num_warps=1)

        # Prefix sum in Triton (int64), then cast to int32 for offsets
        offsets64 = torch.empty(self.num_experts + 1, dtype=torch.int64, device=flat.device)
        offsets64[0] = 0  # inclusive prefix starts at 0
        grid_p = (1,)
        prefix_sum_kernel[grid_p](counts, offsets64, num_experts=self.num_experts, num_warps=1)

        expert_offsets = offsets64.to(torch.int32)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
