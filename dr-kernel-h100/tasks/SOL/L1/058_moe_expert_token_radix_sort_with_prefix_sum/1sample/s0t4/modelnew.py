import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(x_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Build histogram of values in x_ptr (int32) into counts_ptr (int32[256]).
    Each element in x_ptr is in [0, 255]. We do one atomic add per element.
    """
    offs = tl.arange(0, BLOCK)
    grid = tl.num_programs(0)
    base = grid * BLOCK
    # Iterate over the whole array in chunks
    for start in range(0, N, base):
        idx = start + offs
        mask = idx < N
        vals = tl.load(x_ptr + idx, mask=mask, other=0)  # int32
        # Atomic add 1 for each valid value
        tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def inclusive_scan_prefix_sum(counts_ptr, offsets_ptr, M: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr[0..M-1] into offsets_ptr[0..M].
    offsets[0] must be set to 0 by host.
    """
    acc = tl.zeros((), dtype=tl.int32)
    # Loop over all elements; M is constexpr so Triton can unroll
    for k in range(0, M):
        acc += counts_ptr[k]
        offsets_ptr[k] = acc


@triton.jit
def bitonic_sort_argsort(values_ptr, indices_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Bitonic sort of values_ptr (int32) in ascending order, producing indices_ptr
    as the permutation (sorted positions). Stable tie-breaking by original index:
    for equal values, i < j implies choose (i, j) over (j, i).
    We operate in-place on values_ptr and indices_ptr.
    """
    # Initialize indices to [0..N-1]
    ar = tl.arange(0, BLOCK)
    for start in range(0, N, BLOCK):
        idx = start + ar
        mask = idx < N
        # Initialize local values and indices
        vals = tl.load(values_ptr + idx, mask=mask, other=0)  # values
        idxs = idx  # original positions

        # Bitonic sort network
        k = 2
        while k <= N:
            j = k // 2
            while j > 0:
                partner = idx ^ j
                valid_partner = partner < N
                # load partner's values and indices
                vals_partner = tl.load(values_ptr + partner, mask=valid_partner, other=0)
                idxs_partner = partner  # if valid

                # Determine ascending or descending segment for pair (idx, partner)
                ascending = ((idx & k) == 0)

                # Compare
                cond = vals < vals_partner
                tie = vals == vals_partner
                # Stable tie-breaking: prefer smaller original index
                tie_choice = idxs < idxs_partner

                need_swap = tl.where(
                    ascending,
                    cond | (tie & tie_choice),
                    (cond | (tie & tie_choice)) ^ 1
                )

                # Compute new values/indices for position 'idx' after potential swap
                new_vals = tl.where(need_swap, vals_partner, vals)
                new_idxs = tl.where(need_swap, idxs_partner, idxs)

                # Store back to current position
                tl.store(values_ptr + idx, new_vals, mask=mask)
                tl.store(indices_ptr + idx, new_idxs, mask=mask)

                j //= 2
            k *= 2


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation of the original run:
        - Compute stable sort permutation of flattened indices using Triton bitonic sort.
        - Compute expert offsets via Triton histogram + inclusive scan.
        """
        assert topk_idx.is_cuda, "topk_idx must be on CUDA for Triton kernels."
        # Flatten indices and ensure contiguous
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        device = flat.device

        # 1) Stable sort permutation via Triton bitonic sort
        # Work on a copy to avoid mutating the original indices
        values = flat.clone()
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=device)
        # Launch bitonic sort with a block size that covers typical N; grid covers all elements
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        bitonic_sort_argsort[grid](values, sorted_token_indices, N=N, BLOCK=BLOCK, num_warps=8)

        # 2) Histogram via Triton
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        grid_hist = (triton.cdiv(N, BLOCK),)
        histogram_kernel[grid_hist](flat, counts, N=N, BLOCK=BLOCK, num_warps=8)

        # 3) Prefix sum (offsets) via Triton inclusive scan
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        offsets[0] = 0
        inclusive_scan_prefix_sum[(1,)](counts, offsets, M=256, num_warps=1)

        return sorted_token_indices, offsets