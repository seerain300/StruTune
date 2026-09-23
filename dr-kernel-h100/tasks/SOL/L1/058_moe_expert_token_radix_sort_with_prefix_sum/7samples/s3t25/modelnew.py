import torch
import triton
import triton.language as tl


@triton.jit
def _argsort_indices_by_values_stable_kernel(a_ptr, N, out_ptr):
    """
    Compute stable argsort permutation of 'a_ptr' (length N) into 'out_ptr'.
    out_ptr[i] = position of the i-th smallest element in 'a_ptr', ties broken by original index.
    We implement this by assigning each original index i a rank and reserving a position via atomic_add.
    """
    # Triton uses 1D grid: each program handles one original index i
    i = tl.program_id(axis=0)
    if i >= N:
        return

    # Load original value and original index i
    val = tl.load(a_ptr + i)

    # Compute rank: number of elements strictly less than val + number of equal elements with j < i
    # We accumulate in int64 to avoid overflow.
    rank = tl.zeros((), dtype=tl.int64)

    # Loop over all j in [0, N)
    # Note: Triton supports loops with dynamic bounds; this is acceptable for these sizes.
    for j in range(0, N):
        vj = tl.load(a_ptr + j)
        # If vj < val, j precedes i in sorted order -> increment rank
        rank += (vj < val)
        # If vj == val and j < i, j precedes i (stable tie-break by original index) -> increment rank
        rank += (vj == val) & (j < i)

    # Reserve a unique output position and place 'i' there
    # out_ptr is initialized to zeros; atomic_add increments by 1 at position 'rank'.
    tl.atomic_add(out_ptr + rank, 1)
    tl.store(out_ptr + rank, i)


@triton.jit
def _histogram_kernel(a_ptr, N, histogram_ptr, num_buckets: tl.constexpr):
    """
    Histogram of int32 values in 'a_ptr' of length N into 'histogram_ptr' of length num_buckets.
    histogram_ptr[idx] += 1 for each occurrence of a_ptr[i] == idx.
    """
    # Each program increments one bucket for each occurrence of that bucket value.
    # We iterate over N and atomic_add to the histogram.
    for i in range(0, N):
        vi = tl.load(a_ptr + i)
        # vi should be in [0, num_buckets-1]; atomic_add by 1
        tl.atomic_add(histogram_ptr + vi, 1)


@triton.jit
def _inclusive_scan_prefix_sum(vals_ptr, out_ptr, n: tl.constexpr):
    """
    Compute inclusive scan (prefix sum) of 'vals_ptr' (length n) into 'out_ptr' (length n),
    and then place inclusive sums into offsets starting at out_ptr[1]. Finally, copy to out_ptr[0:n].
    """
    # First, copy vals to out[1..]
    for i in range(0, n):
        tl.store(out_ptr + 1 + i, tl.load(vals_ptr + i))
    # Inclusive scan using iterative doubling
    size = 1
    while size < n:
        for i in range(0, n):
            right = i + size
            # If right within bounds, add previous partial sum
            add_val = tl.load(out_ptr + right) if right < n else 0
            tl.store(out_ptr + i, tl.load(out_ptr + i) + add_val)
        size *= 2
    # Now out_ptr[1..n] contains inclusive prefix sums. We need them into offsets[1..].
    # Just copy back to offsets[1..] (host code will copy to offsets).
    for i in range(1, n + 1):
        tl.store(out_ptr + i, tl.load(out_ptr + i))


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten and ensure contiguous int32 (values are indices in [0, 255], so int32 is fine)
        flat = topk_idx.reshape(-1).to(torch.int32).contiguous()
        device = flat.device
        N = flat.numel()
        num_experts = 256  # matches original code's num_experts

        # 1) Stable argsort permutation: sorted_token_indices of length N
        out = torch.zeros(N, dtype=torch.int64, device=device)  # use int64 for atomic_add
        grid = (N,)
        _argsort_indices_by_values_stable_kernel[grid](flat, N, out)

        # 2) Histogram of expert IDs
        histogram = torch.zeros(num_experts, dtype=torch.int64, device=device)
        _histogram_kernel[(1,)](flat, N, histogram, num_buckets=num_experts)

        # 3) Prefix sum to get expert offsets (length num_experts + 1)
        offsets = torch.empty(num_experts + 1, dtype=torch.int64, device=device)
        _inclusive_scan_prefix_sum[(1,)](histogram, offsets, n=num_experts)

        # Convert to int32 for consistency with original (original uses int32)
        sorted_token_indices = out.to(torch.int32)
        offsets = offsets.to(torch.int32)

        return sorted_token_indices, offsets