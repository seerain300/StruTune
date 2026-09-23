# task: SOL/L1/058_moe_expert_token_radix_sort_with_prefix_sum
# batch: stts3turn
# pass_at_1: 0.12
# final_geomean_speedup(A800, official re-eval): 0.005
import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_kernel(inp_ptr, N, out_ptr):
    """
    Triton histogram: one program per element.
    - inp_ptr points to a 1D array of indices (host provides int32).
    - For each element idx_val, atomically increment out_ptr[idx_val].
    """
    pid = tl.program_id(axis=0)
    mask = pid < N
    # Load index as int32 (mask ensures no OOB)
    idx_val = tl.load(inp_ptr + pid, mask=mask, other=0).to(tl.int32)
    # Atomically increment the bin for this index
    tl.atomic_add(out_ptr + idx_val, 1, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version:
        - Flattens topk_idx, computes histogram in Triton, then creates expert_offsets via torch.cumsum.
        - Keeps the stable sort in PyTorch as in the original code.
        Returns:
          - sorted_token_indices: int32 tensor of shape (N,)
          - expert_offsets: int32 tensor of shape (num_experts + 1,)
        """
        # Ensure tensor is on CUDA and contiguous
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to(device="cuda")
        topk_idx = topk_idx.contiguous()

        # Flatten
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # Triton counts per expert id (num_experts = 256 as in the original run)
        num_experts = 256
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)

        # Launch one program per element
        grid = (N,)
        _histogram_kernel[grid](flat, N, counts, num_warps=1)

        # Compute expert offsets: inclusive prefix sum
        expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
        running = 0
        for i in range(num_experts):
            running += counts[i]
            expert_offsets[i + 1] = running

        # Stable sort of flattened indices (same as original)
        sorted_token_indices = flat.sort(stable=True)[1].to(torch.int32)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
