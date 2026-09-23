import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _argsort_bitonic_vals_kernel(
    vals_ptr,            # *int32, flattened values
    N,                   # int32, total number of elements
    out_idx_ptr,         # *int32, output permutation of original indices (length N)
    BLOCK: tl.constexpr  # int, next power-of-two >= N
):
    """
    Bitonic argsort ascending. For ties (vals equal), lower original index comes first (stable-like behavior).
    Writes permutation out_idx_ptr[0..N-1] which are the original indices sorted by vals ascending.
    We operate on a power-of-two BLOCK length; out-of-range indices are ignored via mask.
    """
    i = tl.arange(0, BLOCK)
    mask = i < N

    # Load values; for masked out-of-range positions, set sentinel value > max int32 to keep them at the end.
    vals = tl.load(vals_ptr + i, mask=mask, other=2**31 - 1)

    # Store original indices in out_idx_ptr
    idx = i
    tl.store(out_idx_ptr + i, idx.to(tl.int32), mask=mask)

    # Bitonic sort network. We pad out-of-range positions with sentinel values so they are at the end.
    size = 2
    while size <= BLOCK:
        stride = size // 2
        while stride > 0:
            j = i ^ stride
            # Only process each pair once (i < j)
            do_pair = i < j

            vj = tl.load(vals_ptr + j, mask=mask, other=2**31 - 1)
            idxj = tl.load(out_idx_ptr + j, mask=mask, other=0)

            ascending = (i & size) == 0
            # If equal, choose lower original index first (tie-breaker) to emulate stable behavior.
            tie_low = (idx < idxj)

            cond = tl.where(
                ascending,
                (vals < vj) | ((vals == vj) & tie_low),
                (vals > vj) | ((vals == vj) & (~tie_low)),
            )
            need_swap = cond & do_pair

            new_vals = tl.where(need_swap, vj, vals)
            new_idx = tl.where(need_swap, idxj, idx)

            tl.store(out_idx_ptr + i, new_idx, mask=do_pair)
            tl.store(out_idx_ptr + j, new_idx, mask=do_pair)  # j's entry also updated as symmetric write

            # Update vals for i to reflect the swap decision; we re-read to keep code simple.
            # Note: Triton does not support direct in-register blocking, so we re-load after each stride.
            vals = tl.load(vals_ptr + i, mask=mask, other=2**31 - 1)
            idx = tl.load(out_idx_ptr + i, mask=mask, other=0)

            stride //= 2
        size *= 2


@triton.jit
def count_histogram_atomic(vals_ptr, counts_ptr, N, num_experts, BLOCK: tl.constexpr):
    """
    Parallel histogram using atomic adds:
    - Grid size: 1 (single program handles entire array; for larger N, grid can be increased).
    - Scans vals_ptr in chunks of BLOCK, counts occurrences of each expert id in range [0, num_experts),
      and atomically adds to counts_ptr[e].
    """
    # Each program handles a chunk; single program here, but we keep loop structure for clarity.
    offset = 0
    while offset < N:
        idx = offset + tl.arange(0, BLOCK)
        mask = idx < N

        vals = tl.load(vals_ptr + idx, mask=mask, other=0)
        vals_i32 = vals.to(tl.int32)

        # For each expert e, count matches in this chunk and atomically add to global counts[e]
        for e in range(0, num_experts):
            matches = (vals_i32 == e) & mask
            cnt_block = tl.sum(matches.to(tl.int32), axis=0)  # scalar count for this chunk
            tl.atomic_add(counts_ptr + e, cnt_block)

        offset += BLOCK


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
    Triton-optimized version:
    - Produces sorted_token_indices (permutation of original indices) via Triton bitonic sort.
    - Produces expert_offsets (exclusive prefix sum) via Triton histogram + prefix sum.
    No torch.sort or torch.bincount in forward.
    """
    def forward(self, topk_idx: torch.Tensor):
        if not TRITON_AVAILABLE:
            # Fallback to original PyTorch behavior if Triton is not available
            flat = topk_idx.reshape(-1)
            sorted_token_indices, _ = torch.sort(flat, dim=0, stable=True)
            counts = torch.bincount(flat.long(), minlength=256)
            expert_offsets = torch.zeros(257, dtype=torch.int32, device=flat.device)
            expert_offsets[1:] = counts.cumsum(0)
            return sorted_token_indices.to(torch.int32), expert_offsets

        # Ensure tensor is on CUDA for Triton
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to('cuda')

        # Flatten and ensure contiguous
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        device = flat.device

        # 1) Triton argsort to produce sorted_token_indices (grid=(1,))
        # Choose BLOCK as next power-of-two >= N
        BLOCK = 1
        while BLOCK < N:
            BLOCK <<= 1
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=device)

        _argsort_bitonic_vals_kernel[(1,)](flat, N, sorted_token_indices, BLOCK, num_warps=4)

        # 2) Triton histogram (counts per expert) using parallel atomics (grid=1)
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        _count_histogram_atomic[(1,)](flat, counts, N, 256, BLOCK)  # pass num_experts=256

        # 3) Triton exclusive prefix sum to produce expert_offsets (length 257)
        expert_offsets = torch.empty(257, dtype=torch.int32, device=device)
        _exclusive_prefix_sum_kernel[(1,)](counts, expert_offsets, 256)

        return sorted_token_indices, expert_offsets


# Note: The kernels must be defined before ModelNew usage. The above defines them.
# Helper functions from the original snippet can be reused:
def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]
    topk_idx = torch.randint(
        0, num_experts,
        (batch_size, seq_len, num_experts_per_tok),
        dtype=torch.int32,
        device=device
    )
    return {"topk_idx": topk_idx}


def run(*args):
    return ModelNew()(*args)
