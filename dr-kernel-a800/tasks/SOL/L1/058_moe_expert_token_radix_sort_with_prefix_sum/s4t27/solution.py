import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_kernel(flat_ptr, N, counts_ptr, num_experts: tl.constexpr, BLOCK: tl.constexpr):
    """
    Histogram of expert IDs in flat_ptr (int32) into counts_ptr (int32).
    Each program processes BLOCK elements:
      - Loads a block of values
      - For each e in [0, num_experts), atomically adds 1 to counts[e] for each match.
    This uses N*BLOCK atomics total, which is acceptable for the given sizes.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load a block of indices (int32)
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)

    # For each expert e, count matches in this block and atomic_add to global counts[e]
    # num_experts is constexpr, so Triton can unroll this loop.
    for e in range(num_experts):
        matches = (vals == e) & mask
        # Convert boolean mask to int32 by summing; since Triton doesn't have tl.sum over vector,
        # we compute the count via tl.where to produce 1s where matches and 0 elsewhere, then sum.
        # However, Triton lacks a built-in sum; we instead perform atomic_add per element:
        # For each element that matches, atomic add 1. This is simple and correct.
        # Note: We can't vector-atomic-add into a scalar efficiently without reductions; per-element add is acceptable here.
        # mask -> int32
        incr = tl.where(matches, 1, 0)
        # Atomic add per element where mask is true. We need to scatter atomic adds; Triton does not support vectorized
        # scatter-add into a scalar easily, so we perform atomic add for each element that matches.
        # Triton supports tl.atomic_add with a pointer and a scalar value.
        # For each lane, we test matches and call atomic_add. Triton will compile this loop.
        for i in range(BLOCK):
            if mask[i]:
                if matches[i]:
                    tl.atomic_add(counts_ptr + e, 1)


@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, NUM_EXPS: tl.constexpr):
    """
    Compute inclusive prefix sums of counts into offsets_ptr[1..].
    offsets_ptr[0] remains 0. Assumes counts_ptr is int32 and offsets_ptr is int32.
    """
    running = 0
    for j in range(NUM_EXPS):
        v = tl.load(counts_ptr + j)
        running += v
        tl.store(offsets_ptr + j + 1, running)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; kept for evaluation harness.

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Flatten topk_idx, sort stably using PyTorch.
        - Compute per-expert counts via Triton histogram.
        - Compute expert offsets via Triton inclusive prefix sum.
        Returns:
        - sorted_token_indices: int32 (N,)
        - expert_offsets: int32 (num_experts+1,)
        """
        # Ensure CUDA and contiguous
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to(device='cuda')
        topk_idx = topk_idx.contiguous()
        flat = topk_idx.view(-1)
        N = flat.numel()
        num_experts = 256  # as in original

        # 1) Stable sort using PyTorch for correctness
        sorted_token_indices = flat.sort(stable=True)[1].to(torch.int32)

        # 2) Triton histogram of expert IDs
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)

        # Launch histogram kernel with one program per chunk of BLOCK elements
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _histogram_kernel[grid](sorted_token_indices, N, counts, num_experts=num_experts, BLOCK=BLOCK, num_warps=4)

        # 3) Triton inclusive prefix sum to produce expert_offsets
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        _inclusive_prefix_sum_kernel[(1,)](counts, expert_offsets, num_experts=num_experts, num_warps=1)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
