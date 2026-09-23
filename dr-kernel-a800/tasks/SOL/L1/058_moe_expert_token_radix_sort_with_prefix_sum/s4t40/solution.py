import torch
import triton
import triton.language as tl


@triton.jit
def _inclusive_prefix_sum_kernel(
    counts_ptr,        # *const int32, shape (num_experts,)
    offsets_ptr,       # *int32, shape (num_experts+1,)
    num_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # set BLOCK_SIZE = num_experts for single-program scan
):
    # Compute inclusive prefix sum of counts into offsets.
    # offsets[0] = 0; offsets[i+1] = sum_{j=0..i} counts[j].
    # We use a single program (grid=(1,)) and BLOCK_SIZE=num_experts
    # to scan all counts and update running sum.

    # Initialize offsets[0] = 0
    tl.store(offsets_ptr + 0, 0)

    running = 0
    # Loop over all expert ids 0..num_experts-1
    for i in range(0, num_experts):
        # Load counts[i] (scalar)
        val_i = tl.load(counts_ptr + i)  # scalar load
        running = running + val_i
        # Store inclusive prefix sum at offsets[i+1]
        tl.store(offsets_ptr + (i + 1), running)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized forward:
        - Use torch.sort for token sorting (correct and efficient).
        - Compute per-expert counts via torch.bincount (fast on GPU).
        - Compute expert_offsets via Triton inclusive prefix sum.
        """
        # Ensure tensor is on CUDA and contiguous
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels."
        topk_idx = topk_idx.contiguous()

        # Flatten the tensor
        flat = topk_idx.reshape(-1)  # shape: (N,)
        N = flat.numel()
        num_experts = 256  # matches the original run function

        # Use torch.sort for correctness
        _, sorted_token_indices = flat.sort(stable=True)

        # Compute per-expert counts using torch.bincount (fast on GPU)
        counts = torch.bincount(flat.long(), minlength=num_experts)

        # Triton inclusive prefix sum kernel to produce expert_offsets
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)

        _inclusive_prefix_sum_kernel[(1,)](
            counts, expert_offsets, num_experts=num_experts, BLOCK_SIZE=num_experts
        )

        return sorted_token_indices.to(torch.int32), expert_offsets


def run(*args):
    return ModelNew()(*args)
