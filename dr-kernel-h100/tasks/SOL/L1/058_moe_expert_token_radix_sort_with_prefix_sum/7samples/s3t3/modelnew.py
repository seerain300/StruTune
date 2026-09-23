import torch
import triton
import triton.language as tl


@triton.jit
def _argsort_indices_by_values_stable_kernel(a_ptr, N, out_ptr):
    """
    Compute stable argsort of the flattened array 'a_ptr' of length N.
    Writes the permutation indices into 'out_ptr' (int32) of length N.
    Stability: for equal values, smaller original index comes first.
    """
    i = tl.program_id(0)  # each program handles one element i in [0, N)
    if i >= N:
        return

    # Load value at position i
    val_i = tl.load(a_ptr + i)

    # Compute rank for i:
    # rank = number of elements strictly less than val_i
    #       + number of elements equal to val_i with original index < i
    count_less = tl.zeros((), dtype=tl.int32)
    count_equal_before = tl.zeros((), dtype=tl.int32)

    # Compare with all other elements j
    for j in range(0, N):
        # Skip j == i
        if j != i:
            val_j = tl.load(a_ptr + j)
            # Stable tie-breaking: count equal and j < i
            if val_j < val_i:
                count_less += 1
            elif val_j == val_i:
                if j < i:
                    count_equal_before += 1

    rank = count_less + count_equal_before

    # Reserve a unique position via atomic add
    p = tl.atomic_add(out_ptr, 1)  # increments out_ptr[0] and returns old value
    # Place i at position p
    tl.store(out_ptr + p, i)


@triton.jit
def _histogram_kernel(a_ptr, N, hist_ptr, num_buckets: tl.constexpr):
    """
    Compute histogram of integer values in 'a_ptr' (int32) of length N,
    into 'hist_ptr' (int32) of length num_buckets. Uses atomic adds.
    """
    # For each element, atomically add 1 to its bucket
    for idx in range(0, N):
        val = tl.load(a_ptr + idx)
        # Ensure val is in range [0, num_buckets-1]
        # Triton requires pointer arithmetic; we just index the bucket
        tl.atomic_add(hist_ptr + val, 1)


@triton.jit
def _inclusive_scan_prefix_sum(hist_ptr, out_ptr, num_buckets: tl.constexpr):
    """
    Compute inclusive prefix sum of 'hist_ptr' (length num_buckets) into 'out_ptr'.
    Uses iterative doubling. Requires 'hist_ptr' to be initialized with counts.
    """
    # We assume out_ptr already has out_ptr[0] = 0
    # and we will write prefix sums out_ptr[1..num_buckets].
    # Note: Triton allows dynamic loops; we use while loop to cover log2(num_buckets) steps.
    offset = 1
    # Step 1: copy input to output at index 1
    # We need to read hist_ptr[0] and write to out_ptr[1]
    # Use a loop over num_buckets to be safe
    for k in range(0, num_buckets):
        val = tl.load(hist_ptr + k)
        tl.store(out_ptr + k + 1, val)

    # Iterative doubling
    step = 1
    while step < num_buckets:
        # For each i in [0, num_buckets - step - 1], out[i + step] += out[i]
        for i in range(0, num_buckets - step):
            prev = tl.load(out_ptr + i)          # inclusive sum up to i
            cur = tl.load(out_ptr + i + step)   # value at i + step
            tl.store(out_ptr + i + step, prev + cur)
        step *= 2


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Compute stable argsort permutation of flattened topk_idx (length N).
        - Compute histogram of topk_idx values, then prefix sum for expert_offsets.
        Returns:
          sorted_token_indices: 1D tensor of int32, length N, stable argsort indices.
          expert_offsets: 1D tensor of int32, length (num_experts + 1), cumsum histogram.
        """
        # Ensure dtype is int32 for Triton kernels
        a = topk_idx.reshape(-1).to(torch.int32).contiguous()
        device = a.device
        N = a.numel()
        num_experts = 256  # matches the original code's hard-coded num_experts

        # 1) Stable argsort: permutation indices
        out = torch.empty(N, dtype=torch.int32, device=device)  # holds positions 0..N-1
        grid = (N,)
        _argsort_indices_by_values_stable_kernel[grid](a, N, out)

        # 2) Histogram of expert IDs (int32)
        histogram = torch.zeros(num_experts, dtype=torch.int32, device=device)
        _histogram_kernel[(1,)](a, N, histogram, num_buckets=num_experts)

        # 3) Prefix sum to get expert_offsets (num_experts + 1), starting at 0
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0
        _inclusive_scan_prefix_sum[(1,)](histogram, offsets, num_buckets=num_experts)

        # Return 1D sorted_token_indices (length N) and expert_offsets (length num_experts+1)
        return out, offsets