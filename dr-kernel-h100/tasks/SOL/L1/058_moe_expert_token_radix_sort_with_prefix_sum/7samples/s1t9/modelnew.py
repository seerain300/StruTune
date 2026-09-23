import torch
import triton
import triton.language as tl


@triton.jit
def histogram_counts_kernel(
    x_ptr,                  # *int32, flattened expert indices
    counts_ptr,             # *int32, length = num_experts
    n_elements: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load values; dtype is int32
    vals = tl.load(x_ptr + offsets, mask=mask, other=0)
    # Atomic add 1 to counts[vals] for valid offsets
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def inclusive_prefix_sum_kernel(
    counts_ptr,             # *int32, length = num_experts
    offsets_ptr,            # *int32, length = num_experts + 1
    num_experts: tl.constexpr,
):
    # Single program computes inclusive prefix sums
    total = 0
    for e in range(num_experts + 1):
        # Load count; for e == 0, no count exists, so we skip
        if e > 0:
            total += tl.load(counts_ptr + (e - 1))
        tl.store(offsets_ptr + e, total)


@triton.jit
def odd_even_compare_swap_even(
    arr_ptr,                # *int32, values to sort
    idx_ptr,                # *int64, indices associated with values
    n_elements: tl.constexpr,
):
    # Only even positions perform compare with next
    i = tl.program_id(axis=0)
    # Bounds and even check
    if (i < n_elements) and ((i % 2) == 0):
        j = i + 1
        # Ensure j is in bounds
        if j < n_elements:
            vi = tl.load(arr_ptr + i)
            vj = tl.load(arr_ptr + j)
            ii = tl.load(idx_ptr + i)
            ij = tl.load(idx_ptr + j)
            # Swap if vi > vj (stable: equal not swapped)
            need_swap = vi > vj
            new_vi = tl.where(need_swap, vj, vi)
            new_vj = tl.where(need_swap, vi, vj)
            new_ii = tl.where(need_swap, ij, ii)
            new_ij = tl.where(need_swap, ii, ij)
            tl.store(arr_ptr + i, new_vi)
            tl.store(arr_ptr + j, new_vj)
            tl.store(idx_ptr + i, new_ii)
            tl.store(idx_ptr + j, new_ij)


@triton.jit
def odd_even_compare_swap_odd(
    arr_ptr,                # *int32, values to sort
    idx_ptr,                # *int64, indices associated with values
    n_elements: tl.constexpr,
):
    # Only odd positions perform compare with prev
    i = tl.program_id(axis=0)
    if (i < n_elements) and ((i % 2) == 1):
        j = i - 1
        # j must be in bounds and non-negative
        if j >= 0:
            vi = tl.load(arr_ptr + i)
            vj = tl.load(arr_ptr + j)
            ii = tl.load(idx_ptr + i)
            ij = tl.load(idx_ptr + j)
            need_swap = vi > vj
            new_vi = tl.where(need_swap, vj, vi)
            new_vj = tl.where(need_swap, vi, vj)
            new_ii = tl.where(need_swap, ij, ii)
            new_ij = tl.where(need_swap, ii, ij)
            tl.store(arr_ptr + i, new_vi)
            tl.store(arr_ptr + j, new_vj)
            tl.store(idx_ptr + i, new_ii)
            tl.store(idx_ptr + j, new_ij)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        # Ensure device is CUDA and flatten
        assert topk_idx.is_cuda, "Input must be on CUDA device for Triton."
        flat = topk_idx.reshape(-1).contiguous()
        n = flat.numel()
        device = flat.device

        # 1) Histogram counts per expert via Triton
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        histogram_counts_kernel[grid](
            flat, counts, n_elements=n, num_experts=self.num_experts, BLOCK_SIZE=BLOCK_SIZE, num_warps=4
        )

        # 2) Inclusive prefix sum to get expert_offsets (int32 -> int64)
        offsets_i32 = torch.empty(self.num_experts + 1, dtype=torch.int32, device=device)
        inclusive_prefix_sum_kernel[(1,)](
            counts, offsets_i32, num_experts=self.num_experts
        )
        expert_offsets = offsets_i32.to(torch.long)

        # 3) Stable sort via Triton odd-even transposition sort:
        # Initialize values (int32) and indices (int64) for sorting.
        arr = flat.to(torch.int32).clone()
        indices = torch.arange(n, dtype=torch.long, device=device)
        # Perform 2*N passes (sufficient to sort). Use a Python while loop.
        passes = 2 * n
        i = 0
        while i < passes:
            if i % 2 == 0:
                odd_even_compare_swap_even[(n,)](arr, indices, n_elements=n)
            else:
                odd_even_compare_swap_odd[(n,)](arr, indices, n_elements=n)
            i += 1

        # Return: sorted indices (int64), expert_offsets (int64)
        return indices, expert_offsets