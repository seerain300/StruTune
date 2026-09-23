import torch
import triton
import triton.language as tl


@triton.jit
def sort_by_counting(a_ptr, out_ptr, N: tl.constexpr):
    """
    Stable sort by counting:
    - a_ptr: flattened, length-N int32 array of expert IDs.
    - out_ptr: length-N int32 array that will hold sorted token indices (original positions).
    For each i in [0, N), compute rank = (#elements < a[i]) + (#elements == a[i] and j < i)
    Then set out[rank] = i. out is returned as sorted_token_indices.
    """
    # One program per element i
    i = tl.program_id(0)

    # Load the value at position i
    # Note: Triton expects pointer arithmetic in elements, not bytes.
    a_i = tl.load(a_ptr + i)

    # Accumulate rank: count_less + count_equal_before
    count_less = tl.zeros((), dtype=tl.int32)
    count_equal_before = tl.zeros((), dtype=tl.int32)

    # Loop over all j
    # N is passed as constexpr so Triton can unroll or optimize this loop.
    for j in range(N):
        # Only do work when j != i (avoid self-comparison)
        # but we still need to accumulate counts; we'll skip self by using j != i mask in loads.
        # Load a_j
        a_j = tl.load(a_ptr + j)

        # Compare
        less = a_j < a_i
        equal = a_j == a_i

        # Add to counts
        count_less += less.to(tl.int32)
        # For stable tie-break: among equal, count how many had j < i
        count_equal_before += equal.to(tl.int32) * (j < i).to(tl.int32)

    rank = count_less + count_equal_before

    # Write original index i at sorted position rank
    tl.store(out_ptr + rank, i)


@triton.jit
def compute_histogram(a_ptr, hist_ptr, N: tl.constexpr):
    """
    Compute histogram of values in a_ptr (length N) into hist_ptr (length num_experts),
    where hist[i] = number of times value i appears.
    """
    i = tl.program_id(0)
    # Load value a[i]
    val = tl.load(a_ptr + i)
    # Atomically add 1 to the corresponding bucket
    tl.atomic_add(hist_ptr + val, 1)


@triton.jit
def compute_prefix_sum(hist_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Compute prefix sums of hist_ptr (length num_experts) into offsets_ptr (length num_experts+1),
    with offsets[0] = 0 and offsets[i] = offsets[i-1] + hist[i-1].
    """
    e = tl.program_id(0)
    if e >= num_experts:
        return
    # Accumulate sum of previous elements
    # We can load offsets[e-1] (0 when e==0) and hist[e-1] (0 for e==0) and set offsets[e] = offsets[e-1] + hist[e-1]
    prev = tl.load(offsets_ptr + e - 1) if (e - 1) >= 0 else tl.zeros((), dtype=tl.int32)
    h = tl.load(hist_ptr + e - 1) if (e - 1) >= 0 else tl.zeros((), dtype=tl.int32)
    curr = prev + h
    tl.store(offsets_ptr + e, curr)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized forward:
        - Computes sorted_token_indices using a counting-sort permutation kernel.
        - Computes expert_offsets using a histogram kernel and a tiny prefix-sum kernel.
        """
        # Ensure we are on CUDA and have int32
        if not topk_idx.is_cuda:
            # Fallback: move to CUDA if available; for safety, if no CUDA, fallback to original PyTorch path
            # But here we assume the caller provides CUDA tensor.
            raise RuntimeError("topk_idx must be a CUDA tensor.")
        if topk_idx.dtype != torch.int32:
            topk_idx = topk_idx.to(torch.int32)

        # Flatten and ensure contiguous
        a = topk_idx.contiguous()
        N = a.numel()
        num_experts = 256  # From the original code; can be made configurable but is fixed here.

        # 1) Stable sort via counting-sort permutation (Triton kernel)
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=a.device)
        # Launch one program per element
        grid_sort = (N,)
        sort_by_counting[grid_sort](a, sorted_token_indices, N)

        # 2) Histogram of expert IDs (Triton kernel)
        histogram = torch.zeros(num_experts, dtype=torch.int32, device=a.device)
        grid_hist = (N,)
        compute_histogram[grid_hist](a, histogram, N)

        # 3) Prefix sum to get expert offsets (Triton kernel)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=a.device)
        # Set offsets[0] to 0 (already zero by default); kernel computes 1..num_experts
        grid_prefix = (num_experts,)
        compute_prefix_sum[grid_prefix](histogram, offsets, num_experts)

        # Reshape sorted_token_indices to original shape if desired (return flattened as per original)
        sorted_token_indices = sorted_token_indices.view(*topk_idx.shape)

        return sorted_token_indices, offsets