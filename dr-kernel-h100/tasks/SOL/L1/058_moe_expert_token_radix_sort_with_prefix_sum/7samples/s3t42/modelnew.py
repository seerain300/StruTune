import torch

# Triton kernels: all computation must be in Triton. No torch operations in host code.

# Stable argsort via rank computation (safe, O(N * BLOCK))
@triton.jit
def _stable_argsort_indices_kernel(a_ptr, N, out_ptr,
                                    BLOCK: tl.constexpr):
    # One program per original index i
    i = tl.program_id(0)  # int32 scalar
    # If i >= N, early return (grid should be N, so not needed, but safe)
    # Compute rank: count of j where a[j] < a[i], plus ties with j < i
    rank = tl.zeros((), dtype=tl.int32)

    # Loop over j in blocks of size BLOCK; j is a scalar
    # Note: comparisons are scalar; we build a vector for range but use scalar j.
    for start in range(0, BLOCK):
        j = start
        # Masked comparison: if j < N, add 1 if (a[j] < a[i]) or (a[j] == a[i] and j < i)
        mask_j = j < N
        a_j = tl.load(a_ptr + j, mask=mask_j, other=0)
        # Load a_i safely
        a_i = tl.load(a_ptr + i)
        less = a_j < a_i
        equal = a_j == a_i
        tie = equal & (j < i)
        cnt = (less | tie).to(tl.int32)
        rank += tl.where(mask_j, cnt, 0)

    # Write i into position 'rank'
    tl.store(out_ptr + rank, i)

# Histogram of flat values (each element increments its bucket)
@triton.jit
def _histogram_kernel(a_ptr, N, hist_ptr, num_buckets: tl.constexpr):
    # One program per element
    i = tl.program_id(0)
    # Load value; assume a_ptr points to int32 values
    val = tl.load(a_ptr + i)
    # Bucket index; since values are in [0, num_buckets-1], this is safe
    bucket = val
    # Atomic add into histogram
    tl.atomic_add(hist_ptr + bucket, 1)

# Inclusive prefix sum (simple single-program loop over buckets)
@triton.jit
def _inclusive_scan_prefix_sum(h_ptr, out_ptr, num_buckets: tl.constexpr):
    # out_ptr[0] = 0, we set in host
    total = tl.zeros((), dtype=tl.int32)
    for k in range(0, num_buckets):
        total += tl.load(h_ptr + k)
        tl.store(out_ptr + k + 1, total)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # Ensure flat tensor on the right device and dtype
        device = topk_idx.device
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # 1) Stable argsort permutation: out[i] = index such that flat[out[i]] is sorted stably
        out = torch.empty(N, dtype=torch.int32, device=device)
        # Use a reasonable BLOCK to reduce loop iterations; 256 works well for typical N
        BLOCK = 256
        grid_argsort = (N,)
        _stable_argsort_indices_kernel[grid_argsort](flat, N, out, BLOCK=BLOCK)

        # 2) Histogram of expert IDs using Triton
        num_experts = 256
        histogram = torch.zeros(num_experts, dtype=torch.int32, device=device)
        grid_hist = (N,)
        _histogram_kernel[grid_hist](flat, N, histogram, num_buckets=num_experts)

        # 3) Compute expert_offsets (inclusive prefix sum) via Triton
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0
        grid_scan = (1,)
        _inclusive_scan_prefix_sum[grid_scan](histogram, offsets, num_buckets=num_experts)

        # Return 1D sorted_token_indices (length N) and expert_offsets (length num_experts+1)
        return out, offsets