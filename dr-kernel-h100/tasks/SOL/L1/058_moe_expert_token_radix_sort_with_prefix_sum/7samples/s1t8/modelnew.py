import torch
import triton
import triton.language as tl


@triton.jit
def histogram_counts_kernel(x_ptr, counts_ptr, n_elements: tl.int32, num_experts: tl.int32, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-expert counts of x_ptr values (int32), using one atomic_add per element.
    counts_ptr: int32 array of size num_experts.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load values; default 0 for out-of-bounds
    val = tl.load(x_ptr + offsets, mask=mask, other=0)
    # Atomic add into counts
    tl.atomic_add(counts_ptr + val, 1, mask=mask)


@triton.jit
def inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.int32):
    """
    Compute inclusive prefix sum of counts_ptr (int32) into offsets_ptr (int32).
    offsets_ptr[0] = 0, offsets_ptr[1..] = cumsum(counts_ptr).
    """
    # Single program computes prefix sum by looping over experts.
    # offsets_ptr must have length num_experts + 1.
    running = 0
    # Store initial 0 at offsets_ptr[0] (though caller can set it to 0)
    tl.store(offsets_ptr + 0, tl.zeros((), dtype=tl.int32))
    # Loop over experts sequentially (compile-time friendly)
    for e in range(0, num_experts):
        c = tl.load(counts_ptr + e)
        running += c
        tl.store(offsets_ptr + e + 1, running)


@triton.jit
def odd_even_compare_swap_even(arr_ptr, idx_ptr, n_elements: tl.int32):
    """
    Even phase of odd-even transposition sort:
    Only even positions (including 0) compare with next; they perform swap if arr[i] > arr[i+1].
    Indices are swapped accordingly.
    """
    i = tl.program_id(axis=0)
    # Only proceed if i is even and within bounds
    mask_i = (i % 2 == 0) & (i < n_elements)
    j = i + 1
    mask_j = j < n_elements
    # Load values
    a = tl.load(arr_ptr + i, mask=mask_i, other=0)
    b = tl.load(arr_ptr + j, mask=mask_j, other=0)
    ia = tl.load(idx_ptr + i, mask=mask_i, other=0)  # int64
    ib = tl.load(idx_ptr + j, mask=mask_j, other=0)  # int64
    # Only do compare-swap if both positions valid
    do_swap = (a > b) & mask_i & mask_j
    # Compute new values
    new_a = tl.where(do_swap, b, a)
    new_b = tl.where(do_swap, a, b)
    new_ia = tl.where(do_swap, ib, ia)
    new_ib = tl.where(do_swap, ia, ib)
    # Store back
    tl.store(arr_ptr + i, new_a, mask=mask_i)
    tl.store(arr_ptr + j, new_b, mask=mask_j)
    tl.store(idx_ptr + i, new_ia, mask=mask_i)
    tl.store(idx_ptr + j, new_ib, mask=mask_j)


@triton.jit
def odd_even_compare_swap_odd(arr_ptr, idx_ptr, n_elements: tl.int32):
    """
    Odd phase of odd-even transposition sort:
    Only odd positions compare with previous; they perform swap if arr[i] > arr[i-1].
    Indices are swapped accordingly.
    """
    i = tl.program_id(axis=0)
    # Only proceed if i is odd and within bounds
    mask_i = (i % 2 == 1) & (i < n_elements)
    j = i - 1
    mask_j = j >= 0
    a = tl.load(arr_ptr + i, mask=mask_i, other=0)
    b = tl.load(arr_ptr + j, mask=mask_j, other=0)
    ia = tl.load(idx_ptr + i, mask=mask_i, other=0)  # int64
    ib = tl.load(idx_ptr + j, mask=mask_j, other=0)  # int64
    do_swap = (a > b) & mask_i & mask_j
    new_a = tl.where(do_swap, b, a)
    new_b = tl.where(do_swap, a, b)
    new_ia = tl.where(do_swap, ib, ia)
    new_ib = tl.where(do_swap, ia, ib)
    tl.store(arr_ptr + i, new_a, mask=mask_i)
    tl.store(arr_ptr + j, new_b, mask=mask_j)
    tl.store(idx_ptr + i, new_ia, mask=mask_i)
    tl.store(idx_ptr + j, new_ib, mask=mask_j)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # num_experts is known from the reference: 256
        self.num_experts = 256

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Compute counts per expert via Triton.
        - Compute expert_offsets via Triton inclusive prefix sum.
        - Perform stable sort via Triton odd-even transposition sort.
        Returns sorted_token_indices (int64 permutation) and expert_offsets (int64).
        """
        # Ensure input is on CUDA and contiguous
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to('cuda')
        flat = topk_idx.reshape(-1).contiguous()
        n = flat.numel()

        # 1) Histogram counts per expert using Triton
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=flat.device)
        BLOCK_SIZE = 1024
        grid_counts = (triton.cdiv(n, BLOCK_SIZE),)
        _histogram_counts_kernel[grid_counts](
            flat, counts, n_elements=n, num_experts=self.num_experts, BLOCK_SIZE=BLOCK_SIZE
        )

        # 2) Inclusive prefix sum to get expert_offsets (int32 -> int64)
        offsets_i32 = torch.empty(self.num_experts + 1, dtype=torch.int32, device=flat.device)
        _inclusive_prefix_sum_kernel[(1,)](
            counts, offsets_i32, num_experts=self.num_experts
        )
        offsets = offsets_i32.to(torch.long)

        # 3) Stable sort via Triton odd-even transposition sort:
        # Keep values in int32 array arr, indices as int64 array idx.
        # Initialize arr with flat values (int32), idx with 0..n-1 (int64).
        arr = flat.to(torch.int32).clone()
        indices = torch.arange(n, dtype=torch.long, device=flat.device)

        # Perform a fixed number of passes (2*N is sufficient for sorting).
        # Use Python while loop; Triton allows control flow here, and we avoid torch.sort.
        passes = 2 * n
        i = 0
        while i < passes:
            if i % 2 == 0:
                _odd_even_compare_swap_even[(n,)](arr, indices, n_elements=n)
            else:
                _odd_even_compare_swap_odd[(n,)](arr, indices, n_elements=n)
            i += 1

        # Return sorted indices (int64) and expert_offsets (int64)
        return indices, offsets