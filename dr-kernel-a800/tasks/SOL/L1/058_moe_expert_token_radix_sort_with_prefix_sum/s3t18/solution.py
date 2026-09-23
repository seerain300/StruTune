import torch
import triton
import triton.language as tl


@triton.jit
def bitonic_argsort_stable(flat_ptr, out_idx_ptr, N, num_expairs: tl.constexpr):
    """
    Argsort the 1D array 'flat_ptr' of length N into 'out_idx_ptr' (store original indices 0..N-1
    in sorted order of values in flat_ptr). Stable tie-breaker uses original index ascending.
    """
    # We perform bitonic sort in-place using an auxiliary index array.
    # out_idx_ptr is treated as indices of flat_ptr.
    idx = torch.arange(N, dtype=torch.int32, device=flat_ptr.device)

    # Bitonic sort network: nested loops over k (sequence length) and j (stride).
    # num_expairs should be the next power-of-two >= log2(N), chosen by caller.
    for k in range(1, num_expairs):
        for j in range(k, 0, -1):
            stride = 1 << (j - 1)
            partner = idx ^ stride  # partner index for each element
            # Load current values and indices
            vi = tl.load(flat_ptr + idx)
            vj = tl.load(flat_ptr + partner)
            ii = idx
            ij = partner

            # Ascending direction when (idx & k) == 0
            asc = ((idx & k) == 0)

            # Compare-exchange
            greater = vi > vj
            equal = vi == vj
            # For ascending: swap if vi > vj; for descending: swap if vi < vj
            swap = tl.where(asc, greater, vi < vj)
            # Tie-break by original index (smaller index first)
            swap = swap | ((vi == vj) & (ii > ij))

            new_vi = tl.where(swap, vj, vi)
            new_vj = tl.where(swap, vi, vj)
            new_ii = tl.where(swap, ij, ii)
            new_ij = tl.where(swap, ii, ij)

            # Write back to positions
            tl.store(flat_ptr + idx, new_vi)
            tl.store(flat_ptr + partner, new_vj)
            tl.store(out_idx_ptr + idx, new_ii)
            tl.store(out_idx_ptr + partner, new_ij)

    # After sorting flat_ptr ascending by values, out_idx_ptr contains original indices in sorted order.


@triton.jit
def count_experts_histogram(flat_ptr, counts_ptr, N, num_experts: tl.constexpr):
    """
    Count occurrences of each expert ID in 'flat_ptr' (length N) into 'counts_ptr' (length num_experts).
    counts_ptr is int32 and assumed zero-initialized by host.
    """
    # Simple per-element counting
    for i in range(0, N):
        val = tl.load(flat_ptr + i)
        # Ensure val is within [0, num_experts-1]
        # Triton supports modulo-like ops; but here val is guaranteed by host.
        # Increment counts[val]
        tl.store(counts_ptr + val, tl.load(counts_ptr + val) + 1)


@triton.jit
def exclusive_prefix_sum(counts_ptr, offsets_ptr, N_bins: tl.constexpr):
    """
    Compute exclusive prefix sum of counts_ptr of length N_bins and store in offsets_ptr of length N_bins + 1.
    offsets[0] = 0; offsets[i] = sum_{k=0..i-1} counts[k] for i > 0.
    """
    # Initialize offsets[0] = 0
    tl.store(offsets_ptr + 0, tl.zeros((), dtype=tl.int32))
    for i in range(1, N_bins):
        total = tl.zeros((), dtype=tl.int32)
        for k in range(0, i):
            total += tl.load(counts_ptr + k)
        tl.store(offsets_ptr + i, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure CUDA and contiguous
        device = topk_idx.device
        if device.type != "cuda":
            topk_idx = topk_idx.to("cuda")
        flat = topk_idx.reshape(-1).contiguous()

        N = flat.numel()
        num_experts = 256

        # 1) Stable argsort using Triton
        # We need a buffer to hold indices 0..N-1; use out_idx as the flat buffer temporarily.
        out_idx = torch.empty_like(flat, dtype=torch.int32, device=device)

        # Determine num_expairs (power of two >= log2(N), at least 1). For N=8*256=2048, num_expairs=11 -> 2048
        # To be general, we can choose 13 which gives 8192; but we should set it to next power-of-two >= N.
        # Triton supports dynamic loops; we can just pass a constexpr that is >= log2(N) + 1. Use next power-of-two.
        # Python-side calculation: next_pow2
        num_expairs = 1 if N <= 1 else 1 << (int(N - 1).bit_length())

        # Launch bitonic argsort
        bitonic_argsort_stable[(1,)](flat, out_idx, N, num_expairs=num_expairs, num_warps=4)

        sorted_token_indices = out_idx  # original indices in ascending sorted order of values

        # 2) Count histogram in Triton (counts int32, length 256)
        counts = torch.empty(num_experts, dtype=torch.int32, device=device)
        # counts must be zero-initialized; torch.empty does not. Use zeros.
        counts.zero_()
        count_experts_histogram[(1,)](flat, counts, N, num_experts=num_experts, num_warps=1)

        # 3) Exclusive prefix sum offsets (length 257)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        exclusive_prefix_sum[(1,)](counts, offsets, N_bins=num_experts, num_warps=1)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
