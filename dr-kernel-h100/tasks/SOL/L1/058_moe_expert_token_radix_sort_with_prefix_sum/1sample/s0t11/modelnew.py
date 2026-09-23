import torch
import triton
import triton.language as tl


@triton.jit
def odd_even_sort_stable(values_ptr, indices_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Stable odd-even sort:
    - values_ptr: pointer to int32 values of length N
    - indices_ptr: pointer to int32 indices (initially 0..N-1) representing permutation
    - Performs N phases of odd-even sort; for even phase compare (0,1), (2,3)... for odd phase compare (1,2), (3,4)...
    - Ties are broken by original indices: swap only if idx1 < idx0.
    """
    # Triton loops are unrolled at compile time when bounds are known. We iterate over phases up to N.
    p = 0
    while p < N:
        # Determine whether this phase is even or odd
        even_phase = (p % 2) == 0
        start = 0 if even_phase else 1
        stride = 2
        # Iterate pairs (start, start+1), (start+stride, start+stride+1), ...
        i = start
        while i + 1 < N:
            # Load current pair
            idx0 = i
            idx1 = i + 1
            v0 = tl.load(values_ptr + idx0)
            v1 = tl.load(values_ptr + idx1)
            idx0_perm = tl.load(indices_ptr + idx0)
            idx1_perm = tl.load(indices_ptr + idx1)

            # Decide swap based on values and stable tie-breaker
            swap = (v1 < v0) | ((v1 == v0) & (idx1_perm < idx0_perm))

            # Compute new values for both positions
            new_v0 = tl.where(swap, v1, v0)
            new_v1 = tl.where(swap, v0, v1)
            new_idx0 = tl.where(swap, idx1_perm, idx0_perm)
            new_idx1 = tl.where(swap, idx0_perm, idx1_perm)

            # Write back
            tl.store(values_ptr + idx0, new_v0)
            tl.store(values_ptr + idx1, new_v1)
            tl.store(indices_ptr + idx0, new_idx0)
            tl.store(indices_ptr + idx1, new_idx1)

            i += stride
        p += 1


@triton.jit
def histogram_atomic_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Build histogram of int32 values in flat_ptr into counts_ptr[0..255].
    - flat_ptr: pointer to int32, length N
    - counts_ptr: pointer to int32, length 256, initialized to zeros
    - Uses masked loads and atomic_add for safety and parallelism.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)
    # Atomic add per valid element
    # Note: counts_ptr is int32; atomic_add supports int32 in Triton
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def inclusive_scan_prefix_sum_kernel(counts_ptr, offsets_ptr, M: tl.int32):
    """
    Compute inclusive prefix sum over counts_ptr[0..M-1] into offsets_ptr[0..M].
    offsets_ptr[0] = 0, offsets_ptr[i] = offsets_ptr[i-1] + counts_ptr[i-1] for i>0.
    Single program instance performs the scan sequentially.
    """
    # Initialize offset[0] = 0 (host will set this)
    running = 0
    # offsets_ptr is 1-based inclusive of M elements
    i = 0
    while i < M:
        # Load count[i], accumulate, and store offset[i+1]
        count_i = tl.load(counts_ptr + i)
        running += count_i
        tl.store(offsets_ptr + i + 1, running)
        i += 1


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Compute sorted_token_indices = argsort(flat, stable=True) using Triton odd-even sort.
        - Compute expert_offsets via Triton histogram and prefix sum.
        """
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels."
        # Flatten and ensure contiguous
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        device = flat.device

        # 1) Stable sort permutation using Triton odd-even sort
        values = flat.clone()  # we will mutate values in the sort
        indices = torch.arange(N, dtype=torch.int32, device=device)
        # Launch odd-even sort
        BLOCK_SORT = 1024  # blocks for parallel processing; loops handle the full range
        grid_sort = (triton.cdiv(N, BLOCK_SORT),)
        odd_even_sort_stable[grid_sort](values, indices, N, BLOCK=BLOCK_SORT, num_warps=4)

        sorted_token_indices = indices

        # 2) Histogram via Triton
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        BLOCK_HIST = 4096
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        histogram_atomic_kernel[grid_hist](flat, counts, N, BLOCK=BLOCK_HIST, num_warps=8)

        # 3) Prefix sum (offsets) via Triton inclusive scan
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        offsets[0] = 0  # initialize first offset
        inclusive_scan_prefix_sum_kernel[(1,)](counts, offsets, M=256, num_warps=1)

        return sorted_token_indices, offsets