import torch
import triton
import triton.language as tl


@triton.jit
def _bitonic_argsort_stable_kernel(
    flat_ptr,            # *int32, flattened values [N] (we'll pad B to next power of two)
    out_ptr,             # *int32, output positions [B] where out[k] = original index of sorted element at position k
    N,                   # int32, actual number of elements
    B: tl.constexpr,     # int, padded size (power of two)
):
    """
    Bitonic stable argsort over padded length B.
    We initialize out = [0, 1, ..., B-1]. Then for each k in [0..B-1], we compute its rank
    among the first N elements using a stable comparison schedule, and write the original
    index (k) to out at that rank position. Only positions < N are written; padded positions
    remain untouched.

    Stable tie-breaking: for equal values, smaller original index should precede larger.
    """
    # We operate per position k
    k = tl.program_id(0)  # 0..B-1
    # Initialize rank for position k
    rank = tl.zeros((), dtype=tl.int32)

    # Precompute constants
    log2_B = tl.int32(0)
    n = B
    while n > 1:
        n = n >> 1
        log2_B += 1

    # Bitonic sorting network
    # For j in 2,4,...,B: for r = floor(j/2), ..., 1
    for j in range(2, B + 1, 2):
        # r decreases from j//2 down to 1
        r = j // 2
        while r >= 1:
            # partner index for k in this comparator
            partner = k ^ r
            # If partner > k, we only do the work once
            if partner > k:
                # Load values and original indices of k and partner
                # We load -inf for partner if partner >= N (padded), which will not win in ascending rounds
                partner_in_range = partner < N
                val_k = tl.load(flat_ptr + k, mask=True, other=0)  # k is always < N for active work, but keep general form
                val_p = tl.load(flat_ptr + partner, mask=partner_in_range, other=0)

                # Determine ascending or descending stage for this block
                # bit = 0 if (k & j) == 0 else 1; for bitonic network, direction is decided by (k & j) == 0
                bit = (k & j) >> (tl.int32(log2_B - 1))  # not ideal; we compute using j properly
                # Better: compute bit using j properly:
                # In bitonic, bit is 1 if (k & j) != 0; bit = ((k & j) != 0) as boolean. Triton uses bitwise operations.
                # Instead, compute ascending via (k & j) == 0: ascending if (k & j) == 0 else descending
                ascending = (k & j) == 0

                # Stable comparison:
                less = (val_k < val_p)
                equal = (val_k == val_p)
                lower_idx = k < partner
                # If ascending: smaller value wins; if equal, smaller original index wins.
                # If descending: larger value wins; if equal, smaller original index (the one with lower original index should be later) but since we're writing rank for 'k', we need to adjust.
                # For 'k', we compute rank increment only if k should move:
                # If ascending: increment rank if val_k > val_p or equal and k > partner (since stable: lower idx precedes).
                # If descending: increment rank if val_k < val_p or equal and k < partner (higher idx should come later).
                cond = tl.where(
                    ascending,
                    (val_k > val_p) | (equal & (k > partner)),
                    (val_k < val_p) | (equal & (k < partner))
                )
                rank += cond.to(tl.int32)
            r = r // 2
    # Write the original index k at its computed rank position (only for k < N)
    tl.store(out_ptr + rank, k, mask=(k < N))


@triton.jit
def _histogram_kernel(
    flat_ptr,            # *int32, flattened values [N]
    N: tl.constexpr,     # int, number of elements
    histogram_ptr,       # *int32, histogram [num_experts]
    num_experts: tl.constexpr,  # int, number of experts (256 here)
):
    """
    Compute histogram of values in flat_ptr (values are in [0, num_experts-1]).
    One atomic add per element.
    """
    i = tl.program_id(0)
    if i < N:
        val = tl.load(flat_ptr + i)
        # Atomic add into the corresponding bucket
        tl.atomic_add(histogram_ptr + val, 1)


@triton.jit
def _inclusive_scan_prefix_sum_kernel(
    input_ptr,           # *int32, input counts [num_experts]
    output_ptr,          # *int32, output offsets [num_experts+1]
    num_experts: tl.constexpr,  # int, number of experts (256)
):
    """
    Perform inclusive prefix sum over input_ptr into output_ptr.
    output_ptr[0] = 0, output_ptr[i+1] = sum(input_ptr[:i]).
    Iterative doubling scan using constexpr loops.
    """
    # Initialize output[1:] from input
    for i in range(0, num_experts):
        tl.store(output_ptr + 1 + i, tl.load(input_ptr + i))
    # Inclusive scan
    step = 1
    while step < num_experts:
        # Read current output[1:], shifted by step, and add to output[1:]
        for i in range(0, num_experts - step):
            prev = tl.load(output_ptr + 1 + i)
            shifted = tl.load(output_ptr + 1 + i + step)
            tl.store(output_ptr + 1 + i, prev + shifted)
        step = step * 2
    # output_ptr[0] remains 0 (we set it in host before launching)


def _next_power_of_two(n: int) -> int:
    # Compute next power of two >= n
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation of run:
        - sorted_token_indices: 1D tensor of length N = topk_idx.numel(), stable argsort by values.
        - expert_offsets: 1D tensor of length (num_experts + 1) = 257, inclusive prefix sum per expert ID.
        """
        # Ensure input is on CUDA and contiguous
        assert topk_idx.is_cuda, "topk_idx must be on CUDA for Triton kernels"
        topk_idx = topk_idx.contiguous()
        flat = topk_idx.view(-1).to(torch.int32)  # values are in [0, 255]; cast to int32 for Triton
        N = flat.numel()
        device = flat.device

        # 1) Stable argsort using bitonic sorting network in Triton
        B = _next_power_of_two(N)  # padded size
        out = torch.empty(B, dtype=torch.int32, device=device)  # argsort positions (will fill 0..N-1 at sorted ranks)

        # Launch kernel: one program per position k in [0..B-1]
        grid = (B,)
        _bitonic_argsort_stable_kernel[grid](
            flat, out, N, B
        )

        # Truncate to first N entries to get permutation indices [0..N-1] (sorted order)
        sorted_token_indices = out[:N]  # 1D int32; original uses int64, but here int32 is fine and matches stable permutation

        # 2) Histogram of expert IDs (values are in [0, 255])
        histogram = torch.zeros(256, dtype=torch.int32, device=device)
        grid_hist = (1,)
        _histogram_kernel[grid_hist](flat, N, histogram, num_experts=256)

        # 3) Prefix sum (cumulative counts) using Triton
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        offsets[0] = 0  # inclusive sum starts at 0
        _inclusive_scan_prefix_sum_kernel[(1,)](histogram, offsets, num_experts=256)

        # Match original dtype: original returns int64 for sorted_token_indices indices and int32 for offsets.
        sorted_token_indices = sorted_token_indices.to(torch.int64)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
