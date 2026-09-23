import torch
import triton
import triton.language as tl


@triton.jit
def _hist_kernel(indices_ptr, counts_ptr, N, NUM_EXPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel to compute per-expert counts from a flattened int32 indices array.
    Each program instance processes BLOCK_SIZE elements, and for each element i in [0, N),
    it atomically increments counts[i] (assuming i in [0, NUM_EXPS)).
    This reduces atomic contention compared to per-element atomics.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load a block of indices (int32), masked for out-of-range
    idx = tl.load(indices_ptr + offs, mask=mask, other=0)

    # For each valid element in the block, atomic add 1 to counts[idx].
    # We iterate over the BLOCK_SIZE lanes and only act on valid lanes.
    for k in range(BLOCK_SIZE):
        i = idx[k]
        valid = mask[k] & (i >= 0) & (i < NUM_EXPS)
        # Atomic add 1 to counts[i] if valid
        tl.atomic_add(counts_ptr + i, 1, mask=valid)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version:
        - Flattens topk_idx
        - Computes per-expert counts via Triton histogram kernel
        - Computes expert_offsets as inclusive prefix sum using torch.cumsum
        Returns: sorted_token_indices (from PyTorch), expert_offsets (int32, length=num_experts+1)
        """
        # Ensure on CUDA
        assert topk_idx.is_cuda, "Input must be on CUDA device for Triton kernels."
        # Flatten and ensure contiguous
        flat = topk_idx.reshape(-1).contiguous()

        # Number of experts is fixed in the reference as 256
        NUM_EXPERTS = 256

        # Allocate counts (int32) initialized to zeros
        counts = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=flat.device)

        # Launch histogram kernel: grid size is ceil_div(N, BLOCK_SIZE)
        N = flat.numel()
        BLOCK_SIZE = 1024  # large block to reduce atomic operations
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        _hist_kernel[grid](flat, counts, N, NUM_EXPERTS, BLOCK_SIZE)

        # Compute expert_offsets: inclusive prefix sum with initial 0
        expert_offsets = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32, device=flat.device)
        expert_offsets[1:] = torch.cumsum(counts, dim=0)

        # sorted_token_indices is computed using PyTorch's stable sort (already efficient)
        sorted_token_indices = flat.sort()[1]

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
