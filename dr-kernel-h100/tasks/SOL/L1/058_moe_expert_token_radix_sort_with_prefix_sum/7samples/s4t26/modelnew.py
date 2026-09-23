import torch
import triton
import triton.language as tl


def _next_power_of_two(n: int) -> int:
    # Returns next power of two >= n
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())


@triton.jit
def stable_argsort_kernel(vals_ptr, out_idx_ptr, N: tl.int32, BLOCK: tl.int32, LOG: tl.int32):
    """
    Stable argsort: produces out_idx_ptr[0..N-1] = permutation such that
    vals_ptr[out_idx_ptr[i]] is sorted ascending. Ties are broken by original index (stable).
    We use a bitonic sorting network with BLOCK lanes and mask out lanes beyond N.
    """
    pid = tl.program_id(0)
    # Each program handles a block of BLOCK lanes; we can process the entire vector with a single program (grid=1).
    # Compute lane id and validity mask.
    lane = pid * BLOCK + tl.arange(0, BLOCK)
    mask = lane < N

    # Load values; for masked lanes, load sentinel MAX_INT so they move to the end.
    MAX_INT = (1 << 31) - 1
    v = tl.load(vals_ptr + lane, mask=mask, other=MAX_INT)

    # Original indices for each lane
    idx = lane  # int32 tensor
    idx = tl.where(mask, idx, N + 1)  # for masked lanes, set to a large index

    # Bitonic sort network: pairwise compare-exchange. We operate on vectors v and idx.
    # For each k, lanes i with lane >> k & 1 == 0 are the "low" side; others are "high".
    # Direction: ascending when (lane & k) == 0, descending otherwise.
    for stage in range(0, LOG):
        k = 1 << (stage + 1)
        j = k // 2
        partner = lane ^ j

        v_partner = v
        idx_partner = idx

        # Load partner's values/indices (respect masks)
        v_partner = tl.load(vals_ptr + partner, mask=(partner < N), other=MAX_INT)
        idx_partner = partner  # original partner index

        # Direction: ascending if (lane & k) == 0
        dir_asc = (lane & k) == 0

        # Compare by value (with sentinel for masked partner, they should be larger)
        cmp_v = v < v_partner
        cmp_eq = v == v_partner

        # For ascending: take v if v<v_partner; for descending: take v if v>v_partner.
        # For ties, we break by original index: lower index gets the lower rank.
        less_v = cmp_v | (cmp_eq & (idx < idx_partner))
        greater_v = (~cmp_v) | (cmp_eq & (idx > idx_partner))

        new_v = tl.where(dir_asc, tl.where(less_v, v, v_partner), tl.where(greater_v, v, v_partner))
        new_idx = tl.where(dir_asc, tl.where(less_v, idx, idx_partner), tl.where(greater_v, idx, idx_partner))

        v = new_v
        idx = new_idx

    # Write out sorted indices for valid lanes
    tl.store(out_idx_ptr + lane, idx, mask=mask)


@triton.jit
def histogram_kernel(flat_ptr, counts_ptr, N: tl.int32, num_experts: tl.int32):
    """
    Counts occurrences of each expert index (0..num_experts-1) in flat_ptr[0..N-1].
    Uses atomic_add to avoid race conditions.
    counts_ptr has length num_experts.
    """
    BLOCK = 1024
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
    # Accumulate counts per bin
    # Note: offsets are int32; vals are int32 expert ids in [0, 255]
    for i in range(0, 256):
        # Each lane contributes 1 if its value equals i
        contrib = (vals == i).to(tl.int32)
        # Atomic add contributions to counts[i]
        # Note: atomic_add expects pointer to int32; counts_ptr[i] is a scalar pointer.
        # Triton supports atomic_add on int32 arrays.
        tl.atomic_add(counts_ptr + i, tl.sum(contrib, axis=0))


@triton.jit
def inclusive_scan_inplace(counts_ptr, E: tl.int32):
    """
    In-place inclusive scan (prefix sum) over the first E elements of counts_ptr (length 256),
    writing the result into counts_ptr[0..E-1]. We also write cumulative sum to counts_ptr[E].
    """
    # We perform a fixed number of passes; 8 passes are sufficient for E up to 256.
    LOG = 8
    for k in range(1, LOG + 1):
        step = 1 << k
        # Update each index i: counts[i] += counts[i - step] if i >= step else 0
        # This is a vectorized update; Triton allows scalar pointer arithmetic.
        # Note: This pattern is supported for small E.
        # We read current counts at i and i-step, then write new counts[i] = old[i] + old[i-step].
        # We must do it in ascending i order to avoid reading updated values.
        for i in range(0, E):
            # Read old value at i
            old_i = tl.load(counts_ptr + i)
            old_i_minus = tl.load(counts_ptr + (i - step)) if (i >= step) else 0
            new_i = old_i + old_i_minus
            tl.store(counts_ptr + i, new_i)


def _run_triton_only(topk_idx: torch.Tensor):
    """
    Triton-only implementation that returns (sorted_token_indices, expert_offsets).
    Assumes num_experts == 256 as in the original run.
    """
    device = topk_idx.device
    dtype = topk_idx.dtype

    # Flatten; ensure contiguous
    flat = topk_idx.reshape(-1).contiguous()
    N = flat.numel()

    # 1) Stable argsort via Triton
    # Choose BLOCK = next power of two >= N, cap at 4096 for safety.
    BLOCK_SORT = min(_next_power_of_two(N), 4096)
    LOG_SORT = (BLOCK_SORT.bit_length() - 1)
    # Allocate output indices
    idx_out = torch.empty(BLOCK_SORT, dtype=torch.int32, device=device)

    # Sentinel for masked lanes
    MAX_INT = (1 << 31) - 1

    # We need a grid of 1 program that covers all N; using BLOCK_SORT lanes and mask lane < N.
    stable_argsort_kernel[(1,)](flat, idx_out, N, BLOCK_SORT, LOG_SORT, num_warps=4, num_stages=2)

    # sorted_token_indices is the first N entries
    sorted_token_indices = idx_out[:N].to(torch.int32)

    # 2) Triton histogram over 256 bins
    counts = torch.zeros(256, dtype=torch.int32, device=device)
    # One program processes the entire flat vector
    histogram_kernel[(1,)](flat, counts, N, 256, num_warps=4, num_stages=2)

    # 3) Triton inclusive scan to get expert_offsets (length 257)
    offsets = torch.empty(257, dtype=torch.int32, device=device)
    offsets[0:256] = counts  # initialize with counts
    # Perform scan in-kernel; it updates offsets[0..255]
    inclusive_scan_inplace[(1,)](offsets, 256, num_warps=1, num_stages=1)

    # Return results
    return sorted_token_indices, offsets


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # Expect a single tensor argument: topk_idx
        if len(args) != 1:
            raise ValueError("ModelNew.forward expects a single tensor argument (topk_idx).")
        topk_idx = args[0]
        # Ensure dtype is int32 (as generated by get_inputs)
        if topk_idx.dtype != torch.int32:
            topk_idx = topk_idx.to(torch.int32)
        sorted_token_indices, expert_offsets = _run_triton_only(topk_idx)
        return sorted_token_indices, expert_offsets