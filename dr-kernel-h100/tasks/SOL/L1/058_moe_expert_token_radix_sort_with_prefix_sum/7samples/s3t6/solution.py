import torch
import triton
import triton.language as tl


@triton.jit
def _argsort_indices_by_values_stable_kernel(a_ptr, N, out_ptr):
    """
    Compute stable argsort indices based on values in 'a_ptr'.
    - a_ptr: pointer to int32 values (flattened topk_idx)
    - N: number of elements
    - out_ptr: pointer to int32 output permutation of length N
    Each program handles one original index i and computes its stable rank,
    then reserves a unique position via atomic_add and writes i to out[pos].
    """
    i = tl.program_id(0)  # original index [0..N-1]
    # Load value at position i
    val_i = tl.load(a_ptr + i)
    # Compute stable rank: count of elements strictly less than val_i, plus
    # count of equal elements with original index < i (for stability)
    less_count = 0
    equal_count = 0
    # Loop over all j
    # Note: Triton will vectorize the operation; this simple nested loop is
    # acceptable for the given sizes. If needed, you can optimize with block-wise scanning.
    for j in range(0, N):
        val_j = tl.load(a_ptr + j)
        # Handle out-of-range j safely (though Triton allows index beyond N;
        # here j < N so it's fine, but keep the logic clear.)
        if j < N:
            if val_j < val_i:
                less_count += 1
            elif val_j == val_i:
                # Stable tie-break: smaller original index comes first
                if j < i:
                    equal_count += 1

    rank = less_count + equal_count
    # Reserve position 'rank' via atomic add to avoid collisions
    pos = tl.atomic_add(out_ptr, 1)  # starting from 0, get the next position
    # If pos < N, place i at out[pos]
    if pos < N:
        tl.store(out_ptr + pos, i)


@triton.jit
def _histogram_kernel(a_ptr, N, hist_ptr, num_buckets: tl.constexpr):
    """
    Count occurrences of each value in a_ptr into hist_ptr.
    a_ptr contains int32 values in [0, num_buckets-1].
    hist_ptr is int32 of length num_buckets, initialized to 0.
    Each program handles one element and atomically increments the bucket.
    """
    idx = tl.program_id(0)  # element index
    if idx < N:
        val = tl.load(a_ptr + idx)
        # Ensure val is within range (since we know topk_idx in [0, num_experts-1])
        tl.atomic_add(hist_ptr + val, 1)


@triton.jit
def _inclusive_scan_prefix_sum(hist_ptr, out_ptr, num_buckets: tl.constexpr):
    """
    Compute inclusive prefix sum of hist_ptr into out_ptr[1..].
    out_ptr[0] must be set to 0 before calling this kernel.
    """
    # Simple iterative doubling scan for fixed num_buckets
    # We use a single-program kernel with loop to compute scan.
    acc = 0
    for k in range(0, num_buckets):
        # load current bucket, add accumulator, store to next position
        current = tl.load(hist_ptr + k)
        acc += current
        tl.store(out_ptr + k + 1, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation of:
          sorted_token_indices = torch.sort(torch.flatten(topk_idx), stable=True).indices  # 1D (N,)
          expert_offsets = torch.bincount(torch.flatten(topk_idx).to(torch.long), minlength=256).cumsum(0)  # 1D (257,)
        All computations are done via Triton kernels.
        """
        # Flatten to 1D
        a = topk_idx.reshape(-1)
        # Cast to int32 for Triton
        a = a.to(torch.int32).contiguous()
        N = a.numel()
        device = a.device

        # 1) Stable argsort by values: compute permutation out of length N
        out = torch.zeros(N, dtype=torch.int32, device=device)  # positions holder, initialized to 0
        grid = (N,)
        _argsort_indices_by_values_stable_kernel[grid](a, N, out)  # launch Triton kernel

        # 2) Histogram of expert IDs (values in a)
        num_experts = 256  # matches original hard-coded num_experts
        histogram = torch.zeros(num_experts, dtype=torch.int32, device=device)
        # Use a 1D grid over N; each program handles one element and atomically increments bucket
        grid_hist = (N,)
        _histogram_kernel[grid_hist](a, N, histogram, num_buckets=num_experts)

        # 3) Prefix sum to get expert_offsets (inclusive cumulative counts)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0  # inclusive prefix sum will write into offsets[1..]
        _inclusive_scan_prefix_sum[(1,)](histogram, offsets, num_buckets=num_experts)

        # Return 1D sorted_token_indices (length N) and expert_offsets (length num_experts+1)
        return out, offsets


def run(*args):
    return ModelNew()(*args)
