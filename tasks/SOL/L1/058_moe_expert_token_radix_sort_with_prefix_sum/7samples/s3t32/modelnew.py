import torch
import triton
import triton.language as tl


@triton.jit
def _stable_argsort_indices_kernel(a_ptr, N, out_idx_ptr, BLOCK_N: tl.constexpr):
    """
    Compute stable argsort permutation indices:
    out_idx[i] = index of the i-th smallest element in 'a_ptr'.
    Ties are broken by original index (stable).
    Complexity: O(N^2), but acceptable for the given sizes.
    """
    i = tl.program_id(0)
    # Load a_i
    a_i = tl.load(a_ptr + i)

    # Accumulate rank (number of elements strictly less than a_i) + stable tie-break for equal values.
    rank = tl.zeros((), dtype=tl.int32)
    j = 0
    while j < BLOCK_N:
        # Mask for j < N
        mask_j = j < N
        a_j = tl.load(a_ptr + j, mask=mask_j, other=0x7FFFFFFF)  # use a large sentinel for invalid j
        less = a_j < a_i
        equal = a_j == a_i
        tie = equal & (j < i)  # ensure stability: smaller original index precedes
        rank += tl.where(less | tie, 1, 0)
        j += 1

    # Write i to position 'rank' in out_idx
    tl.store(out_idx_ptr + rank, i)


@triton.jit
def _histogram_kernel(vals_ptr, N, histogram_ptr, num_buckets: tl.constexpr):
    """
    Histogram of integer values in [0, num_buckets-1] via atomic_add.
    vals_ptr is assumed to contain int32 values.
    """
    i = tl.program_id(0)
    # Each program handles one element
    val = tl.load(vals_ptr + i)
    # Mask to ensure valid index; num_buckets is constexpr
    if val >= 0 and val < num_buckets:
        tl.atomic_add(histogram_ptr + val, 1)


@triton.jit
def _inclusive_scan_prefix_sum(hist_ptr, out_ptr, num_buckets: tl.constexpr):
    """
    Compute inclusive prefix sum of hist_ptr into out_ptr[1..num_buckets].
    out_ptr[0] is assumed to be zero by caller.
    Single-program loop over num_buckets is acceptable here (num_buckets=256).
    """
    total = tl.zeros((), dtype=tl.int32)
    # out_ptr[0] already zero
    for b in range(num_buckets):
        total += tl.load(hist_ptr + b)
        tl.store(out_ptr + b + 1, total)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version of run:
        - sorted_token_indices: 1D int32 of length N = topk_idx.numel(), argsort by values (stable)
        - expert_offsets: 1D int32 of length (num_experts + 1) = 257, cumulative counts of values.
        """
        # Flatten values and prepare
        a = topk_idx.reshape(-1).contiguous()
        device = a.device
        N = a.numel()
        # We'll operate on int32 for Triton kernels
        a = a.to(torch.int32)

        # 1) Compute stable argsort permutation (indices) via Triton
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=device)
        # Launch one program per element i; BLOCK_N must cover N, but the while loop masks j < N.
        # Using BLOCK_N=N avoids extra masked iterations. Triton supports passing N as constexpr or runtime.
        # Here, we pass N as runtime and loop up to N.
        grid_argsort = (N,)
        _stable_argsort_indices_kernel[grid_argsort](a, N, sorted_token_indices, BLOCK_N=N)

        # 2) Compute histogram of expert IDs using Triton
        num_experts = 256  # matches original run
        histogram = torch.zeros(num_experts, dtype=torch.int32, device=device)
        grid_hist = (N,)
        _histogram_kernel[grid_hist](a, N, histogram, num_buckets=num_experts)

        # 3) Compute expert_offsets (inclusive prefix sum) via Triton
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0  # ensure starting at 0
        grid_scan = (1,)
        _inclusive_scan_prefix_sum[grid_scan](histogram, offsets, num_buckets=num_experts)

        return sorted_token_indices, offsets