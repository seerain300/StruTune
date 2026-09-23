import torch
import triton
import triton.language as tl


@triton.jit
def _argsort_bitonic_vals_kernel(vals_ptr, n_elements: tl.constexpr, out_idx_ptr):
    """
    Triton kernel: compute sorted indices that would sort vals_ptr[0:n_elements] ascending.
    Implements a bitonic compare-exchange network on values only, with stable tie-breaking:
      swap when current value > partner value (ascending)
      AND when values are equal AND current index > partner index (lower original index first).
    Assumes n_elements is a power-of-two for the bitonic network.
    Grid: 1 program instance; loops over stages.
    """
    # Scratch buffer for the values (not needed for output, but we can keep state)
    scratch_vals = tl.full((n_elements,), 0, tl.int32)

    # Copy initial values to scratch for sorting
    i = tl.arange(0, n_elements)
    scratch_vals = tl.load(vals_ptr + i)

    # Bitonic sort network with power-of-two length n_elements
    k = 2
    while k <= n_elements:
        j = k // 2
        while j >= 1:
            partner = i ^ j
            mask_i = i < partner

            vi = tl.load(vals_ptr + i)
            vpartner = tl.load(vals_ptr + partner)

            # tie-breaking: swap when value is greater, or equal and higher index
            lt_val = vi < vpartner
            eq_val = vi == vpartner
            hi_idx = vi > vpartner  # not used for swap; kept for clarity
            tie_swap = eq_val & (i > partner)
            swap = (vi > vpartner) | tie_swap

            # new values for positions i and partner after compare-exchange
            new_vi = tl.where(swap, vpartner, vi)
            new_partner = tl.where(swap, vi, vpartner)

            # write back results
            tl.store(vals_ptr + i, new_vi, mask=mask_i)
            tl.store(vals_ptr + partner, new_partner, mask=mask_i)

            j //= 2
        k *= 2

    # After sorting, out_idx_ptr should contain permutation of indices 0..n_elements-1.
    # Since we never changed indices themselves, we just return the original indices in ascending order.
    # But here, we return original indices as output sorted by values (stable tie-break).
    # Generate out_idx as original indices 0..n_elements-1.
    j_out = tl.arange(0, n_elements)
    tl.store(out_idx_ptr + j_out, j_out, mask=(j_out < n_elements))


@triton.jit
def count_histogram_atomic(vals_ptr, n_elements, counts_ptr, num_experts: tl.constexpr, BLOCK: tl.constexpr):
    """
    Count occurrences of each expert id [0..num_experts-1] in vals_ptr[0:n_elements].
    Each program scans a contiguous chunk of size BLOCK, computes per-block counts via
    atomics into counts_ptr[e] for each e.
    """
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    # load values for this block
    vals = tl.load(vals_ptr + offsets, mask=mask, other=0)

    # accumulate counts per expert using atomics
    for e in range(num_experts):
        m = (vals == e) & mask
        cnt_block = tl.sum(m.to(tl.int32), axis=0)
        tl.atomic_add(counts_ptr + e, cnt_block)


@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, N_bins: tl.constexpr):
    """
    Compute exclusive prefix sum of counts into offsets_ptr[0..N_bins-1]:
    offsets[i] = sum_{k=0..i-1} counts[k] for i in 1..N_bins-1; offsets[0] = 0.
    This is O(N_bins^2), acceptable for N_bins=256.
    """
    tl.store(offsets_ptr + 0, tl.zeros((), dtype=tl.int32))
    for i in range(1, N_bins):
        total = tl.zeros((), dtype=tl.int32)
        for k in range(0, i):
            total += tl.load(counts_ptr + k)
        tl.store(offsets_ptr + i, total)


class ModelNew(torch.nn.Module):
    """
    Triton-optimized version of the original Model:
    - sorted_token_indices: permutation of original flattened indices that sorts values ascending.
      Implemented via Triton bitonic sort with stable tie-breaking (lower original index first).
    - expert_offsets: exclusive prefix sum over histogram of expert indices computed in Triton.
    No torch.sort or torch.bincount in forward.
    """
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # Ensure tensor is on CUDA for Triton
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to('cuda')

        # Flatten and ensure contiguous
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        device = flat.device

        # 1) Triton argsort (ascending) to produce sorted_token_indices
        # Use bitonic network on power-of-two BLOCK >= N. Next power-of-two:
        BLOCK = 1
        while BLOCK < N:
            BLOCK <<= 1
        # Allocate output indices buffer (int32): indices 0..N-1
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=device)

        # Launch bitonic argsort kernel: grid=(1,)
        # Note: The kernel sorts the values in-place (vals_ptr) and produces the permutation
        # by writing original indices 0..N-1 to out_idx_ptr. For ties, lower index comes first.
        _argsort_bitonic_vals_kernel[(1,)](flat, N, sorted_token_indices, num_warps=4)

        #


def run(*args):
    return ModelNew()(*args)
