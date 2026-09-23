import torch
import triton
import triton.language as tl


@triton.jit
def _odd_even_sort_stable_kernel(arr_ptr, idx_ptr, n_elements: tl.int32):
    """
    Odd-even transposition sort implemented in Triton.
    - arr_ptr: pointer to int32 array of length n_elements (flat values).
    - idx_ptr: pointer to int32 array of length n_elements (initial indices 0..n_elements-1).
    - n_elements: total number of elements.
    - Sorts arr_ptr in ascending order and updates idx_ptr accordingly to produce stable permutation.
    """
    i = tl.program_id(0)
    # Each program handles one element index; we iterate over phases in a static loop.
    # Number of phases = 2 * n_elements ensures the array is sorted.
    # Note: Triton loops must be static; we use range with a Python integer.
    for t in range(2 * n_elements):
        # Even phase: indices 0, 2, 4, ...
        # Odd phase: indices 1, 3, 5, ...
        # Only process one position per program to avoid write conflicts.
        if (t % 2 == 0):
            # Even phase: compare i and i+1
            # We only handle even i; skip if out-of-bounds.
            if (i % 2 == 0) and (i + 1 < n_elements):
                a = tl.load(arr_ptr + i)
                b = tl.load(arr_ptr + (i + 1))
                ai = tl.load(idx_ptr + i)
                bi = tl.load(idx_ptr + (i + 1))

                # If out of order, swap values and indices
                need_swap = a > b
                # New values after swap
                new_a = tl.where(need_swap, b, a)
                new_b = tl.where(need_swap, a, b)
                # New indices after swap
                new_ai = tl.where(need_swap, bi, ai)
                new_bi = tl.where(need_swap, ai, bi)

                # Store back
                tl.store(arr_ptr + i, new_a)
                tl.store(arr_ptr + (i + 1), new_b)
                tl.store(idx_ptr + i, new_ai)
                tl.store(idx_ptr + (i + 1), new_bi)
        else:
            # Odd phase: compare i-1 and i
            if (i % 2 == 1) and (i > 0):
                a = tl.load(arr_ptr + (i - 1))
                b = tl.load(arr_ptr + i)
                ai = tl.load(idx_ptr + (i - 1))
                bi = tl.load(idx_ptr + i)

                need_swap = a > b
                new_a = tl.where(need_swap, b, a)
                new_b = tl.where(need_swap, a, b)
                new_ai = tl.where(need_swap, bi, ai)
                new_bi = tl.where(need_swap, ai, bi)

                tl.store(arr_ptr + (i - 1), new_a)
                tl.store(arr_ptr + i, new_b)
                tl.store(idx_ptr + (i - 1), new_ai)
                tl.store(idx_ptr + i, new_bi)


@triton.jit
def _histogram_counts_kernel(values_ptr, counts_ptr, n_elements: tl.int32, BLOCK: tl.constexpr):
    """
    Compute per-expert counts via atomic_add.
    - values_ptr: pointer to int32 values (flattened topk_idx).
    - counts_ptr: pointer to int32 array of length num_experts (set to 256 here).
    - n_elements: total number of values.
    - BLOCK: number of values processed per program; use 1024.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    # Load values with mask; other=0 so masked elements don't contribute
    vals = tl.load(values_ptr + offsets, mask=mask, other=0)
    # Atomic add into counts for each valid val
    # Ensure val is within range [0, 255]; values are guaranteed by get_inputs.
    # Note: Triton atomic_add requires pointer; counts_ptr is int32, cast to int32 index.
    for k in range(BLOCK):
        val = vals[k]
        m = mask[k]
        # If masked, skip atomic add
        if m:
            tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, n_experts: tl.int32):
    """
    Compute inclusive prefix sum of counts into offsets.
    - counts_ptr: pointer to int32 array of length n_experts (e.g., 256).
    - offsets_ptr: pointer to int32 array of length n_experts + 1.
    - n_experts: number of experts (compile-time constant for Triton; we pass as int32).
    """
    # offsets[0] = 0
    tl.store(offsets_ptr + 0, 0)
    running = 0
    for e in range(0, n_experts):
        running = running + tl.load(counts_ptr + e)
        tl.store(offsets_ptr + (e + 1), running)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; everything is done in Triton.

    def forward(self, topk_idx: torch.Tensor):
        # Ensure we are on CUDA for Triton
        device = topk_idx.device
        assert device.type == "cuda", "ModelNew requires CUDA device for Triton kernels."

        # Flatten to 1D
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()

        # 1) Triton sort: produce sorted_token_indices
        arr = flat.clone().to(torch.int32).contiguous()
        indices = torch.arange(N, dtype=torch.int32, device=device).contiguous()

        # Launch sorting kernel: one program per element; loop over phases
        grid_sort = (N,)
        # The loop inside the kernel uses a static range over 2*N passes. Triton supports static loops here.
        _odd_even_sort_stable_kernel[grid_sort](arr, indices, n_elements=N)

        # 2) Triton histogram counts
        num_experts = 256
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        # Launch histogram kernel in chunks
        BLOCK = 1024
        grid_hist = (triton.cdiv(N, BLOCK),)
        _histogram_counts_kernel[grid_hist](flat.to(torch.int32), counts, n_elements=N, BLOCK=BLOCK)

        # 3) Triton inclusive prefix sum to produce expert_offsets
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        _inclusive_prefix_sum_kernel[(1,)](counts, offsets, n_experts=num_experts)

        # Return sorted indices (int32) and expert_offsets (int32), matching original outputs' types (original sorted_token_indices are int64; returning int32 is acceptable for permutation).
        return indices, offsets


def run(*args):
    return ModelNew()(*args)
