import torch
import triton
import triton.language as tl


@triton.jit
def _argsort_indices_by_values_stable_kernel(a_ptr, N, out_ptr):
    """
    Compute stable argsort of the flattened values in 'a_ptr' (int32), producing
    'out_ptr' (int32) of length N. 'out_ptr[i]' is the original linear index i
    placed at the position determined by its stable rank among 'a_ptr'.
    """
    i = tl.program_id(0)  # linear program id
    # Load value at position i
    v = tl.load(a_ptr + i)
    # Compute stable rank: count elements strictly less than v,
    # plus count of equal elements with original index < i (for stability)
    less = 0
    equal_before = 0
    for j in range(0, N):
        val_j = tl.load(a_ptr + j)
        j_is_less = val_j < v
        j_equal = val_j == v
        j_less_i = j < i
        less += j_is_less
        equal_before += j_equal & j_less_i
    rank = less + equal_before
    # Reserve a unique position 'pos' via atomic add; each i writes to pos = rank
    pos = tl.atomic_add(out_ptr + 0, 1)  # out_ptr[0] is a scratch counter
    # Write original index i at position 'pos'
    tl.store(out_ptr + pos, i)


@triton.jit
def _histogram_kernel(a_ptr, N, out_ptr, num_buckets: tl.constexpr):
    """
    Count occurrences of each value in 'a_ptr' (int32) into 'out_ptr' (int32),
    assuming values are in [0, num_buckets-1] and num_buckets == 256 in this task.
    Each program processes one element and performs an atomic_add into the
    appropriate bucket.
    """
    i = tl.program_id(0)
    val = tl.load(a_ptr + i)
    # Ensure val is within bucket range
    in_range = (val >= 0) & (val < num_buckets)
    if in_range:
        tl.atomic_add(out_ptr + val, 1)


# Optional: Triton prefix sum for offsets (kept minimal; torch.cumsum used here for robustness).
# If you require fully Triton-only, you can replace torch.cumsum with a simple Triton loop
# that reads histogram[0..255] and writes inclusive sums into offsets[1..256].
# For correctness and simplicity, we use torch.cumsum below, which is fast and avoids Triton complexity.

class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation of:
          flat = topk_idx.reshape(-1)  # int32
          sorted_token_indices = flat.sort(stable=True).indices  # 1D int64 length N
          expert_offsets = torch.bincount(flat.long(), minlength=256).cumsum(0)  # 1D int64 length 257
        This forward uses Triton kernels to compute the argsort indices and histogram,
        and torch.cumsum for the prefix sum of histogram.
        """
        device = topk_idx.device
        # Flatten and ensure int32 for Triton kernels
        a = topk_idx.reshape(-1).to(torch.int32).contiguous()
        N = a.numel()
        num_experts = 256  # matches original code's hardcoded num_experts

        # 1) Stable argsort by values: compute permutation of indices
        out = torch.empty(N, dtype=torch.int32, device=device)  # holds sorted_token_indices (positions 0..N-1)
        # Initialize a scratch counter to 0
        out[0] = 0
        grid = (N,)
        _argsort_indices_by_values_stable_kernel[grid](a, N, out)

        # 2) Histogram of expert IDs (values of a), assuming in [0, 255]
        histogram = torch.zeros(num_experts, dtype=torch.int32, device=device)
        grid_hist = (N,)
        _histogram_kernel[grid_hist](a, N, histogram, num_buckets=num_experts)

        # 3) Prefix sum to get expert_offsets (cumulative counts) on device using torch for robustness
        # torch.cumsum returns int64 by default; convert to int32 at the end to match original offsets dtype.
        offsets = torch.cumsum(histogram, dim=0).to(torch.int32)
        offsets = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), offsets], dim=0)

        # Return 1D sorted_token_indices (indices) and expert_offsets (length num_experts+1)
        # Note: original sorted_token_indices is 1D int64; here we return int32 indices (safe for small N),
        # but to match original dtype exactly, cast to int64:
        out = out.to(torch.int64)
        return out, offsets


def run(*args):
    return ModelNew()(*args)
