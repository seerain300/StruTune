import torch
import triton
import triton.language as tl


@triton.jit
def odd_even_sort_kernel(vals_in_ptr, perm_in_ptr, vals_out_ptr, perm_out_ptr, N, BLOCK: tl.constexpr):
    """
    Odd-even transposition sort implemented in Triton for a 1D int32 array.
    We produce a permutation of indices [0..N-1] such that sorting the original
    array would place elements in ascending order. We write both the sorted
    values and the corresponding permutation (mapping original position to
    sorted position).
    """
    # Initialize output buffers: copy input to output
    i = tl.arange(0, BLOCK)
    mask = i < N
    vals = tl.load(vals_in_ptr + i, mask=mask, other=0)
    p = tl.where(mask, i, 0)  # original positions
    tl.store(vals_out_ptr + i, vals, mask=mask)
    tl.store(perm_out_ptr + i, p, mask=mask)

    # Perform N passes (sufficient for sorting; makes sorting deterministic)
    for t in tl.static_range(0, N):
        # Even phase: compare (0,1), (2,3), ...
        if (t % 2) == 0:
            idx = tl.arange(0, BLOCK)
            mask_pairs = ((idx % 2) == 0) & ((idx + 1) < N)
            left = idx
            right = idx + 1
            val_left = tl.load(vals_out_ptr + left, mask=mask_pairs, other=0)
            val_right = tl.load(vals_out_ptr + right, mask=mask_pairs, other=0)
            asc = val_left > val_right
            new_left = tl.where(asc, val_right, val_left)
            new_right = tl.where(asc, val_left, val_right)
            tl.store(vals_out_ptr + left, new_left, mask=mask_pairs)
            tl.store(vals_out_ptr + right, new_right, mask=mask_pairs)

            # For permutation: if we swap values, swap perm at left/right positions as well.
            p_left = tl.load(perm_out_ptr + left, mask=mask_pairs, other=0)
            p_right = tl.load(perm_out_ptr + right, mask=mask_pairs, other=0)
            swap_mask = asc  # swap when left > right
            p_new_left = tl.where(swap_mask, p_right, p_left)
            p_new_right = tl.where(swap_mask, p_left, p_right)
            tl.store(perm_out_ptr + left, p_new_left, mask=mask_pairs)
            tl.store(perm_out_ptr + right, p_new_right, mask=mask_pairs)

        # Odd phase: compare (1,2), (3,4), ...
        else:
            idx = tl.arange(0, BLOCK)
            mask_pairs = ((idx % 2) == 1) & ((idx + 1) < N)
            left = idx
            right = idx + 1
            val_left = tl.load(vals_out_ptr + left, mask=mask_pairs, other=0)
            val_right = tl.load(vals_out_ptr + right, mask=mask_pairs, other=0)
            asc = val_left > val_right
            new_left = tl.where(asc, val_right, val_left)
            new_right = tl.where(asc, val_left, val_right)
            tl.store(vals_out_ptr + left, new_left, mask=mask_pairs)
            tl.store(vals_out_ptr + right, new_right, mask=mask_pairs)

            # For permutation: swap when left > right
            p_left = tl.load(perm_out_ptr + left, mask=mask_pairs, other=0)
            p_right = tl.load(perm_out_ptr + right, mask=mask_pairs, other=0)
            swap_mask = asc
            p_new_left = tl.where(swap_mask, p_right, p_left)
            p_new_right = tl.where(swap_mask, p_left, p_right)
            tl.store(perm_out_ptr + left, p_new_left, mask=mask_pairs)
            tl.store(perm_out_ptr + right, p_new_right, mask=mask_pairs)


@triton.jit
def histogram_kernel(vals_ptr, counts_ptr, N, NUM_EXPS: tl.constexpr, BLOCK: tl.constexpr):
    """
    Triton kernel to compute histogram of int32 values in vals_ptr of length N
    across [0..NUM_EXPS-1]. We process the array in chunks of size BLOCK and
    aggregate per-bin counts locally, then emit a single atomic_add per bin.
    counts_ptr is int32 and will be initialized to zeros on host.
    """
    chunk_id = tl.program_id(0)
    start = chunk_id * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load values; if masked, set other to 0 (safe, masked indices won't be used)
    vals = tl.load(vals_ptr + offsets, mask=mask, other=0)
    # For masked positions, set vals to -1 so they don't contribute
    vals = tl.where(mask, vals, -1)

    # Local aggregation: for each bin, count how many equals to bin appear in this chunk
    local = tl.zeros((NUM_EXPS,), dtype=tl.int32)
    for j in tl.static_range(0, NUM_EXPS):
        eq = vals == j
        # eq is boolean; sum over the vector to count occurrences
        local[j] = tl.sum(eq.to(tl.int32), axis=0)

    # Atomically add to global counts
    for j in tl.static_range(0, NUM_EXPS):
        tl.atomic_add(counts_ptr + j, local[j])


@triton.jit
def prefix_sum_kernel(counts_ptr, offsets_ptr, NUM_EXPS: tl.constexpr):
    """
    Triton kernel to compute inclusive prefix sums of counts (length NUM_EXPS)
    and write to offsets_ptr (length NUM_EXPS+1), with offsets[0]=0.
    """
    # Program id not needed; we run one program and loop
    acc = tl.zeros((), dtype=tl.int32)
    tl.store(offsets_ptr + 0, tl.zeros((), dtype=tl.int32))  # offsets[0]=0
    for i in tl.static_range(0, NUM_EXPS):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + i + 1, acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Sort flattened topk_idx to produce sorted_token_indices (permutation),
          using Triton odd-even transposition sort.
        - Compute expert_offsets via Triton histogram and prefix sum.
        Returns: sorted_token_indices (int32), expert_offsets (int32).
        """
        assert topk_idx.is_cuda, "Input tensor must be on CUDA for Triton execution."
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32."

        # Ensure contiguous
        topk_idx = topk_idx.contiguous()
        N = topk_idx.numel()
        NUM_EXPS = 256  # as in the original code

        # Allocate output buffers for sort
        vals_in = topk_idx
        perm_in = torch.arange(N, device=topk_idx.device, dtype=torch.int32)
        vals_out = torch.empty(N, dtype=torch.int32, device=topk_idx.device)
        perm_out = torch.empty(N, dtype=torch.int32, device=topk_idx.device)

        # Launch odd-even sort: we need to choose BLOCK size. Use up to 1024 elements per program.
        BLOCK = 1024
        grid_sort = (triton.cdiv(N, BLOCK),)
        odd_even_sort_kernel[grid_sort](vals_in, perm_in, vals_out, perm_out, N, BLOCK=BLOCK, num_warps=4)

        # sorted_token_indices is the permutation 'perm_out' which maps original positions to sorted order.
        sorted_token_indices = perm_out

        # Histogram counts of expert ids in vals_out (which equals the original topk_idx values)
        counts = torch.zeros(NUM_EXPS, dtype=torch.int32, device=topk_idx.device)
        grid_hist = (triton.cdiv(N, BLOCK),)
        histogram_kernel[grid_hist](topk_idx, counts, N, NUM_EXPS, BLOCK=BLOCK, num_warps=4)

        # Prefix sum to produce expert_offsets
        expert_offsets = torch.empty(NUM_EXPS + 1, dtype=torch.int32, device=topk_idx.device)
        prefix_sum_kernel[(1,)](counts, expert_offsets, NUM_EXPS, num_warps=1)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
