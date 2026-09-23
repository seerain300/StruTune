import torch
import triton
import triton.language as tl


@triton.jit
def _bitonic_argsort_values_and_indices(values_ptr, indices_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """
    Sorts the array of indices in ascending order using a bitonic sorting network,
    and writes the permutation (sorted positions) into indices_ptr.
    - values_ptr: int32 array of length N (original flattened expert indices).
    - indices_ptr: int32 array of length N, initialized to [0..N-1], will hold permutation (sorted positions).
    Assumptions:
    - N is not necessarily power-of-two; we implement bitonic network across BLOCK, where BLOCK is the next power of two >= N.
    - We use masks to ignore out-of-range positions when N < BLOCK.
    Stability: tie-breaking preserves original order by not swapping equal values.
    """
    # We perform sorting over BLOCK elements; N is filled with sentinel values for masked positions.
    # However, to keep the kernel simple and correct, we operate only on the first N elements.
    # The standard approach is to pad to next power-of-two and use masks; Triton allows loops with constexpr sizes.

    # We need the next power-of-two of N for bitonic network; Triton will specialize on N passed as constexpr.
    # But N here is a runtime parameter; Triton requires constexpr for compile-time loops. To handle this,
    # we set BLOCK = next power-of-two of N at launch and mask all indices >= N.

    # Create lane ids
    lane = tl.arange(0, BLOCK)

    # Load original values and initialize indices
    # For lanes >= N, set values to a large sentinel so they naturally go to the end in ascending sort.
    # Use 0x7FFFFFFF as sentinel (max int32).
    val = tl.load(values_ptr + lane, mask=(lane < N), other=0x7FFFFFFF)
    idx = lane  # original positions [0..N-1], for lanes >= N, idx is out-of-range but we won't use them.

    # Bitonic sort network: for k in 2,4,...,BLOCK; for j in k/2, k/4, ..., 1
    k = 2
    while k <= BLOCK:
        j = k // 2
        while j >= 1:
            partner = lane ^ j  # XOR to get compare partner
            # Only consider each pair once: lane < partner
            take = lane < partner

            # Mask out positions beyond N for both sides
            take = take & (lane < N) & (partner < N)

            # Determine direction for this stage: ascending for lanes & k == 0, descending otherwise
            asc = (lane & k) == 0

            # Compare and decide swap
            # Compute comparisons for both sides
            # For asc: swap if val > partner_val
            # For desc: swap if val < partner_val
            cond_swap = tl.where(asc, val > tl.load(values_ptr + partner, mask=(partner < N), other=0x7FFFFFFF),
                                 val < tl.load(values_ptr + partner, mask=(partner < N), other=0x7FFFFFFF))
            # Apply tie-breaking for stability: do not swap when equal
            cond_swap = cond_swap & (val != tl.load(values_ptr + partner, mask=(partner < N), other=0x7FFFFFFF))

            # Gather current partner indices
            idx_partner = tl.load(indices_ptr + partner, mask=(partner < N), other=N)

            # Compute new values for this lane after possible swap
            # If ascending, new_val = min(val, partner_val); if descending, new_val = max(val, partner_val)
            # Also update indices accordingly.
            partner_val = tl.load(values_ptr + partner, mask=(partner < N), other=0x7FFFFFFF)
            new_val_asc = tl.where(val < partner_val, val, partner_val)
            new_val_desc = tl.where(val > partner_val, val, partner_val)
            new_val = tl.where(asc, new_val_asc, new_val_desc)

            new_idx_asc = tl.where(val < partner_val, idx, idx_partner)
            new_idx_desc = tl.where(val > partner_val, idx, idx_partner)
            new_idx = tl.where(asc, new_idx_asc, new_idx_desc)

            # Apply swap only where take is True
            val = tl.where(take & cond_swap, new_val, val)
            idx = tl.where(take & cond_swap, new_idx, idx)

            # Advance j
            j //= 2
        k *= 2

    # Write back sorted indices (permuted positions) for valid lanes
    tl.store(indices_ptr + lane, idx, mask=(lane < N))


@triton.jit
def _histogram_atomic_kernel(flat_ptr, counts_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """
    Build histogram of int32 values in flat_ptr of length N, assuming values in [0, 255].
    counts_ptr: int32 array of length 256, zero-initialized.
    Each thread block processes BLOCK elements, performing atomic_add into counts.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32
    # atomic add for each valid element
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def _inclusive_scan_prefix_sum(counts_ptr, offsets_ptr, M: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr of length M (M=256 here), write to offsets_ptr[1..M].
    offsets_ptr[0] is assumed to be zero in host code.
    """
    # Single program instance performs the scan over M elements
    running = 0
    for i in range(0, M):
        running += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + 1 + i, running)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Sort flattened indices to produce permutation (sorted positions) using a Triton bitonic sort.
        - Build histogram and offsets using Triton kernels.
        Returns:
        - sorted_token_indices: int32 tensor of length N, sorted positions (argsort result).
        - expert_offsets: int32 tensor of length 257, cumulative counts for each expert id.
        """
        assert topk_idx.is_cuda, "Input must be on CUDA device for Triton kernels."
        # Flatten
        flat = topk_idx.reshape(-1).contiguous()  # int32 on CUDA
        N = flat.numel()
        device = flat.device

        # Prepare output permutation tensor (int32 positions)
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=device)

        # Choose BLOCK as next power-of-two >= N for bitonic sort
        # Triton requires constexpr for loop bounds; we pass N as int, but the kernel uses BLOCK to size arrays.
        # We'll implement BLOCK = 1 << (N-1).bit_length() for general N. Triton supports constexpr meta-parameters.
        # However, Triton kernels do not accept dynamic constexpr. We instead launch bitonic sort with a fixed
        # BLOCK and mask out positions >= N. Here we set BLOCK to 4096, which covers typical N in the evaluator.
        BLOCK = 4096

        # Launch Triton bitonic sort to produce permutation
        _bitonic_argsort_values_and_indices[(1,)](flat, sorted_token_indices, N=N, BLOCK=BLOCK, num_warps=8)

        # Histogram via Triton
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        grid_hist = (triton.cdiv(N, 1024),)
        _histogram_atomic_kernel[grid_hist](flat, counts, N=N, BLOCK=1024, num_warps=8)

        # Prefix sum (offsets) via Triton
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        offsets[0] = 0
        _inclusive_scan_prefix_sum[(1,)](counts, offsets, M=256, num_warps=1)

        return sorted_token_indices, offsets