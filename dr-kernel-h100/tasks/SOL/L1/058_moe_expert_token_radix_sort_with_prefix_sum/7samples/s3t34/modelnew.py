import torch
import triton
import triton.language as tl


@triton.jit
def _stable_argsort_indices_by_values_kernel(a_ptr, N, out_ptr, BLOCK_N: tl.constexpr):
    """
    Compute stable argsort permutation of the values in 'a_ptr' (length N).
    out_ptr[rank] = i, where 'rank' is the number of elements strictly less than a[i]
    plus the number of equal elements with original index j < i (tie-break by original index).
    One program per i; launches N programs. This is O(N^2) but simple and correct.
    """
    i = tl.program_id(0)
    # Load the value at position i
    a_i = tl.load(a_ptr + i)

    # Accumulate rank: count of elements strictly less than a_i, plus stable tie-break count
    rank = tl.zeros((), dtype=tl.int32)
    j = 0
    while j < BLOCK_N:
        mask_j = j < N
        a_j = tl.load(a_ptr + j, mask=mask_j, other=0x7FFFFFFF)  # large 'other' for masked loads
        less = a_j < a_i
        equal = a_j == a_i
        tie = equal & (j < i)
        rank += tl.where(less | tie, 1, 0)
        j += 1

    # Write index i into out_ptr at position 'rank'
    tl.store(out_ptr + rank, i)


@triton.jit
def _histogram_kernel(a_ptr, N, histogram_ptr, num_buckets: tl.constexpr):
    """
    Count occurrences of each value in 'a_ptr' (int32, values in [0, num_buckets-1]).
    Each element performs an atomic add to histogram[value].
    """
    i = tl.program_id(0)
    if i < N:
        val = tl.load(a_ptr + i)
        # Cast to int32 for atomic add
        val32 = val.to(tl.int32)
        tl.atomic_add(histogram_ptr + val32, 1)


@triton.jit
def _inclusive_scan_prefix_sum(histogram_ptr, offsets_ptr, num_buckets: tl.constexpr):
    """
    Compute inclusive scan of 'histogram_ptr' (length num_buckets) into offsets_ptr[1..].
    offsets_ptr[0] must be set to 0 before calling.
    Single program performs iterative doubling scan.
    """
    # Copy histogram to offsets[1..] (inclusive)
    j = 0
    while j < num_buckets:
        offsets_ptr[1 + j] = histogram_ptr[j]
        j += 1

    # Inclusive scan using iterative doubling
    stride = 1
    while stride < num_buckets:
        k = stride
        while k > 0:
            # offsets[k + stride] += offsets[k]
            offsets_ptr[k + stride] += offsets_ptr[k]
            k -= 1
        stride *= 2


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized forward:
        - Compute sorted_token_indices = argsort of flattened topk_idx values (stable) as 1D int32.
        - Compute expert_offsets as cumulative counts per expert ID (length num_experts + 1).
        Returns (sorted_token_indices, expert_offsets).
        """
        # Flatten and ensure int32, contiguous
        flat = topk_idx.reshape(-1).to(torch.int32).contiguous()
        device = flat.device
        N = flat.numel()
        num_experts = 256  # matches original code's num_experts

        # 1) Stable argsort: permutation indices
        out = torch.empty(N, dtype=torch.int32, device=device)
        # Launch one program per i
        grid = (N,)
        # Note: BLOCK_N is not strictly needed for this kernel; we pass any constexpr. Using N is fine.
        _stable_argsort_indices_by_values_kernel[grid](flat, N, out, BLOCK_N=N)

        # 2) Histogram of expert IDs using Triton
        histogram = torch.zeros(num_experts, dtype=torch.int32, device=device)
        _histogram_kernel[grid](flat, N, histogram, num_buckets=num_experts)

        # 3) Prefix sum to get expert_offsets (num_experts + 1), starting at 0
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0
        _inclusive_scan_prefix_sum[(1,)](histogram, offsets, num_buckets=num_experts)

        return out, offsets