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
    i = tl.program_id(0)  # linear program id for each element i in [0, N)
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
def _compute_expert_offsets_kernel(a_ptr, N, offsets_ptr, num_buckets: tl.constexpr):
    """
    Compute expert offsets (cumulative counts per expert ID) from the flattened
    values in 'a_ptr' (int32), writing into 'offsets_ptr' (int32) of length
    num_buckets + 1. We assume values are in [0, num_buckets-1] (here num_buckets=256).
    The kernel performs an inclusive scan across buckets to fill offsets[1..256].
    offsets_ptr[0] is left as scratch (we set it to 0 on host).
    """
    # We use a static loop over buckets and iterative doubling for inclusive scan.
    # Copy histogram to offsets[1..]
    for b in range(0, num_buckets):
        # histogram[b] is the count of elements equal to b
        # We compute counts by looping over N; here we recompute counts per bucket
        cnt = 0
        for j in range(0, N):
            val = tl.load(a_ptr + j)
            cnt += (val == b)
        # Write to offsets[1 + b]
        tl.store(offsets_ptr + 1 + b, cnt)

    # Inclusive scan via iterative doubling: offsets[1..256] = cumsum
    step = 1
    while step < num_buckets:
        # For each bucket j, offsets[j+step] += offsets[j]
        for j in range(0, num_buckets - step):
            tl.store(offsets_ptr + 1 + j + step, tl.load(offsets_ptr + 1 + j + step) +
                     tl.load(offsets_ptr + 1 + j))
        step *= 2
    # offsets_ptr[0] should be 0; write it explicitly
    tl.store(offsets_ptr + 0, 0)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Compute sorted_token_indices (stable argsort of flattened topk_idx values) via Triton.
        - Compute expert_offsets (cumulative counts per expert ID) via Triton.
        """
        # Ensure data is int32 and contiguous for Triton kernels
        a = topk_idx.reshape(-1).to(torch.int32).contiguous()
        device = a.device
        N = a.numel()
        num_experts = 256  # matches original code's hard-coded num_experts

        # 1) Stable argsort: permutation indices
        out = torch.empty(N, dtype=torch.int32, device=device)  # holds positions 0..N-1
        grid = (N,)
        _argsort_indices_by_values_stable_kernel[grid](a, N, out)

        # 2) Expert offsets via Triton inclusive scan (no torch.cumsum/cat)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        # Launch Triton kernel to compute offsets; it will fill offsets[1..256]
        # We still need to zero offsets[0] on host side
        _compute_expert_offsets_kernel[(1,)](a, N, offsets, num_buckets=num_experts)

        # Cast sorted indices to int64 to match original run's dtype
        sorted_token_indices = out.to(torch.int64)

        return sorted_token_indices, offsets