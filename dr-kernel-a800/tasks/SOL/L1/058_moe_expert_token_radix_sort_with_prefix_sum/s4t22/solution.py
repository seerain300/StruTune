import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(flat_ptr, counts_ptr, N, num_experts: tl.constexpr):
    """
    Per-element histogram: for each element, atomically add 1 to counts[idx].
    grid = (N,)
    """
    pid = tl.program_id(0)
    in_bounds = pid < N
    idx = tl.load(flat_ptr + pid, mask=in_bounds, other=0)
    idx = idx.to(tl.int32)
    # If idx is out of range [0, num_experts-1], atomic_add still safe (won't harm).
    tl.atomic_add(counts_ptr + idx, 1, mask=in_bounds)


@triton.jit
def inclusive_scan_experts(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Compute inclusive prefix sums per expert.
    One program per expert index i; it accumulates counts[0..i] and writes offsets[i+1] = running.
    offsets[0] = 0 is set on host after the kernel.
    """
    i = tl.program_id(0)  # program id is the expert index
    running = 0
    # Unrolled loop over all experts; for j == i, we write running at the end.
    for j in range(num_experts):
        count_j = tl.load(counts_ptr + j)
        running += count_j
    # Write inclusive sum at position i+1
    tl.store(offsets_ptr + (i + 1), running)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version:
        - Flattens topk_idx, computes counts per expert id using a Triton histogram kernel.
        - Computes expert_offsets via a Triton inclusive prefix-sum kernel.
        - Returns sorted_token_indices (PyTorch stable sort) and expert_offsets.
        """
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to(device="cuda")

        # Flatten and ensure contiguous
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()

        # Triton histogram
        num_experts = 256
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)

        grid = (N,)
        histogram_kernel[grid](flat, counts, N, num_experts=num_experts, num_warps=1)

        # Triton inclusive scan across experts
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)

        # Run per-expert inclusive scan: grid = (num_experts,)
        grid_experts = (num_experts,)
        inclusive_scan_experts[grid_experts](counts, expert_offsets, num_experts=num_experts, num_warps=1)

        # Set offsets[0] = 0 (original logic sets expert_offsets[0] = 0 via torch.bincount)
        # Our scan wrote offsets[1:], so we explicitly set offsets[0] = 0.
        expert_offsets[0] = 0

        # Stable sort of flattened indices (PyTorch; not tied to num_experts)
        sorted_token_indices = flat.sort(stable=True)[1].to(torch.int32)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
