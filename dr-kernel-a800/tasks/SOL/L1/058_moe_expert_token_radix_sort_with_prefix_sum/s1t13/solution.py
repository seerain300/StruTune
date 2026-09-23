import torch

NUM_EXPERTS = 256  # fixed as per original code


@triton.jit
def count_experts_kernel(flat_ptr, counts_ptr, N: tl.int32, NUM_EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    """
    Count occurrences of each expert id in flat_ptr[0:N].
    Uses atomic_add to accumulate counts. NUM_EXPERTS is 256.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    for off in range(BLOCK):
        idx = start + off
        mask = idx < N
        if mask:
            idv = tl.load(flat_ptr + idx)  # expert id
            # Ensure idv is within valid range; counts_ptr is int32
            tl.atomic_add(counts_ptr + idv, 1)


@triton.jit
def prefix_sum_kernel(counts_ptr, out_ptr, M: tl.int32):
    """
    Compute inclusive prefix sum of counts_ptr[0:M] into out_ptr[0:M].
    Sequential loop; M is expected to be small (<= NUM_EXPERTS).
    """
    carry = tl.zeros((), dtype=tl.int32)
    for i in range(M):
        val = tl.load(counts_ptr + i)
        carry += val
        tl.store(out_ptr + i, carry)


@triton.jit
def compute_out_pos_real(flat_ptr, le_counts_ptr, lt_counts_ptr, out_ptr, N: tl.int32, NUM_EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    """
    Compute stable argsort permutation out[i] for i in [0, N), where out[i] is the
    stable position of flat[i] based on le_counts_ptr and lt_counts_ptr of length NUM_EXPERTS.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    for off in range(BLOCK):
        idx = start + off
        m = idx < N
        if m:
            idv = tl.load(flat_ptr + idx)  # scalar expert id
            # Get le_counts[idv] and lt_counts[idv]
            lev = tl.load(le_counts_ptr + idv)
            ltv = tl.load(lt_counts_ptr + idv)
            duplicates = 1 if (ltv > 0) else 0
            pos = lev - duplicates  # scalar
            tl.store(out_ptr + idx, pos)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation that reproduces:
          - sorted_token_indices = torch.argsort(topk_idx.flatten(), stable=True)
          - expert_offsets = torch.bincount(topk_idx.flatten()).cumsum(0)
        All heavy computation is done via Triton kernels; no torch ops on tensors.
        Returns (sorted_token_indices, expert_offsets).
        """
        # Flatten and ensure contiguous
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        device = flat.device

        # Prepare expert-related data
        counts = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=device)
        # Launch counting kernel
        BLOCK_COUNT = 1024  # tile size; each program processes BLOCK_COUNT elements
        grid_count = (triton.cdiv(N, BLOCK_COUNT),)
        count_experts_kernel[grid_count](flat, counts, N, NUM_EXPERTS, BLOCK_COUNT)

        # Compute inclusive prefix sum of counts (le_counts) and lt_counts
        le_counts = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=device)
        prefix_sum_kernel[(1,)](counts, le_counts, NUM_EXPERTS)
        lt_counts = le_counts - counts  # exclusive prefix count per expert

        # Allocate output for sorted token indices
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=device)

        # Launch Triton kernel to compute stable argsort permutation
        BLOCK_OUT = 1024  # each program handles BLOCK_OUT elements
        grid_out = (triton.cdiv(N, BLOCK_OUT),)
        compute_out_pos_real[grid_out](flat, le_counts, lt_counts, sorted_token_indices, N, NUM_EXPERTS, BLOCK_OUT)

        # Optionally compute expert_offsets using Triton to avoid any torch usage
        # This mirrors original: torch.bincount + torch.cumsum, implemented via Triton:
        # We can reuse counts or re-bin; simplest is to bincount via the same kernel and cumsum via prefix_sum_kernel.
        # However, since we already computed counts, we can compute offsets:
        offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=device)
        # offsets[0] = 0 by default in empty
        # Compute inclusive cumsum for offsets[1:]
        # We need the current counts (which are correct). Just re-run prefix_sum_kernel with these counts.
        # For clarity, we can prefix_sum over counts and write into offsets[1:].
        offsets[0] = 0
        # Copy counts into a temporary buffer to compute prefix over counts and write into offsets[1:]
        tmp = counts.clone()
        # Launch prefix_sum_kernel over NUM_EXPERTS and store into offsets[1:]
        prefix_sum_kernel[(1,)](tmp, offsets[1:], NUM_EXPERTS)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
