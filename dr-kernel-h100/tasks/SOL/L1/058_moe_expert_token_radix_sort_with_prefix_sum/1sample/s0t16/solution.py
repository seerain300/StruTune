import torch
import triton
import triton.language as tl


@triton.jit
def _odd_even_sort_with_indices(values_ptr, indices_out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """
    Perform odd-even transposition sort on 'values' of length N, using a parallel 'indices' array.
    - values_ptr: pointer to int32 array of length N (will be mutated)
    - indices_out_ptr: pointer to int32 array of length N (output permutation indices)
    - N: number of elements
    - BLOCK: number of lanes in this program. We run grid = (1,). Each lane handles multiple elements
             via modulo indexing to perform all-pairs compare-swap.
    """
    # We run a single program instance with many lanes; each lane participates in all phases.
    for phase in range(0, N):
        # Determine if this is an even or odd phase
        is_even_phase = (phase % 2) == 0
        # Inner loop runs in steps of 2 (pairwise)
        t = 0
        # Loop over pairs (t, t+1). We vectorize across lanes to cover all pairs.
        while t < N - 1:
            i = t
            j = i + 1
            # Only even or odd positions depending on phase
            if (is_even_phase and (i % 2) == 0) or (not is_even_phase and (i % 2) == 1):
                # Load values at positions i and j
                v_i = tl.load(values_ptr + i)
                v_j = tl.load(values_ptr + j)
                # Load original indices at positions i and j
                idx_i = tl.load(indices_out_ptr + i)
                idx_j = tl.load(indices_out_ptr + j)

                # Stable compare-swap: if v_i > v_j, swap; if equal, keep order (preserve original index)
                should_swap = v_i > v_j
                new_v_i = tl.where(should_swap, v_j, v_i)
                new_v_j = tl.where(should_swap, v_i, v_j)
                new_idx_i = tl.where(should_swap, idx_j, idx_i)
                new_idx_j = tl.where(should_swap, idx_i, idx_j)

                # Store back
                tl.store(values_ptr + i, new_v_i)
                tl.store(values_ptr + j, new_v_j)
                tl.store(indices_out_ptr + i, new_idx_i)
                tl.store(indices_out_ptr + j, new_idx_j)
            t += 2


@triton.jit
def _histogram_atomic_kernel(values_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    """
    Histogram kernel: counts[value] += 1 for each element value in [0..255].
    Uses atomic_add to avoid races. Assumes values_ptr points to int32 array.
    counts_ptr is int32 of length 256.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N
    vals = tl.load(values_ptr + idx, mask=mask, other=0)
    # vals are int32. Atomic add 1 for each lane within bounds.
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def _inclusive_scan_prefix_sum(counts_ptr, offsets_ptr, M: tl.constexpr):
    """
    Compute inclusive prefix sum of 'counts_ptr' (length M) into 'offsets_ptr' (length M+1).
    offsets_ptr[0] = 0; offsets_ptr[i] = sum_{k=0..i-1} counts_ptr[k] for i in [1..M].
    """
    # idx is a single program instance; we sequentially compute prefix sum.
    sum_val = tl.zeros((), dtype=tl.int32)  # scalar
    for i in range(0, M):
        c = tl.load(counts_ptr + i)
        sum_val = sum_val + c
        tl.store(offsets_ptr + i + 1, sum_val)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version:
        - Sort flattened indices using a Triton odd-even transposition sort (stable),
          producing sorted_token_indices of length N.
        - Compute expert offsets using Triton histogram and prefix sum.
        """
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels."
        # Flatten indices
        flat = topk_idx.reshape(-1)  # int32 on GPU
        N = flat.numel()
        device = flat.device

        # 1) Triton stable sort: produce permutation indices
        # Prepare working copy of values and output indices
        values = flat.clone()  # int32
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=device)

        # Initialize sorted_token_indices with original positions [0..N-1]
        torch.arange(N, out=sorted_token_indices)

        # Launch odd-even sort kernel
        # Use a single program instance with many lanes. BLOCK can be set to 1024; loops cover all N.
        BLOCK = 1024
        grid = (1,)
        _odd_even_sort_with_indices[grid](values, sorted_token_indices, N=N, BLOCK=BLOCK, num_warps=4)

        # 2) Histogram via Triton (counts per expert id 0..255)
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        grid_hist = (triton.cdiv(N, BLOCK),)
        _histogram_atomic_kernel[grid_hist](values, counts, N=N, BLOCK=BLOCK, num_warps=8)

        # 3) Prefix sum (offsets) via Triton inclusive scan over 256 elements
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        offsets[0] = 0  # initialize first offset
        _inclusive_scan_prefix_sum[(1,)](counts, offsets, M=256, num_warps=1)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
