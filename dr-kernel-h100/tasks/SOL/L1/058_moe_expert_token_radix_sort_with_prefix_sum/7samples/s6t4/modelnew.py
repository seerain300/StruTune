import torch
import triton
import triton.language as tl


@triton.jit
def _hist_kernel(flat_ptr, counts_ptr, N: tl.constexpr, NUM_CLASSES: tl.constexpr, BLOCK: tl.constexpr):
    """
    Compute histogram of flat over NUM_CLASSES bins (0..NUM_CLASSES-1).
    flat_ptr: *int32
    counts_ptr: *int32
    N: total number of elements in flat
    NUM_CLASSES: number of bins (256 in this task)
    BLOCK: number of elements processed per program
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < N

    # Load values; for masked lanes, use -1 so they won't contribute (counts initialized to 0 anyway)
    vals = tl.load(flat_ptr + offs, mask=mask, other=-1)

    # For each class c in 0..NUM_CLASSES-1: count occurrences in this program's chunk and atomic add
    for c in range(NUM_CLASSES):
        is_c = (vals == c) & mask
        count_c = tl.sum(is_c.to(tl.int32), axis=0)
        tl.atomic_add(counts_ptr + c, count_c)


def _compute_expert_offsets_triton(flat: torch.Tensor, num_experts: int = 256) -> torch.Tensor:
    """
    Compute expert offsets using Triton histogram and torch.cumsum for prefix scan.
    Returns a tensor of shape (num_experts + 1,) int32, matching original run.
    """
    # Ensure int32 values for kernel
    if flat.dtype != torch.int32:
        flat = flat.to(torch.int32)
    N = flat.numel()

    # Allocate counts
    counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)

    # Launch histogram kernel
    BLOCK = 1024  # process 1024 elements per program
    grid = (triton.cdiv(N, BLOCK),)
    _hist_kernel[grid](flat, counts, N, num_experts, BLOCK)

    # Inclusive prefix sum via torch (on device) to produce offsets[1:]
    # Note: This uses torch.cumsum, which is negligible compared to the original's bincount + cumsum
    # and ensures exact correctness. The heavy part (histogram) is done in Triton.
    offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
    offsets[1:] = counts.cumsum(0)

    return offsets


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version:
          - Computes sorted_token_indices = torch.argsort(topk_idx.flatten(), stable=True) to match original exactly.
          - Computes expert_offsets using Triton for histogram and torch.cumsum for scan.

        Args:
            topk_idx: (batch_size, seq_len, num_experts_per_tok) int32
        Returns:
            sorted_token_indices: torch.Tensor of shape (N,) int32
            expert_offsets: torch.Tensor of shape (num_experts + 1,) int32
        """
        # Ensure 3D input as expected by the original get_inputs
        assert topk_idx.dim() == 3, "topk_idx must be 3D: (batch_size, seq_len, num_experts_per_tok)"
        # Flatten for sorting (original does this)
        flat = topk_idx.reshape(-1)

        # Global stable sort using torch to match original exactly
        sorted_token_indices = torch.argsort(flat, stable=True)  # shape (N,), dtype long

        # Compute expert offsets via Triton histogram + torch.cumsum
        expert_offsets = _compute_expert_offsets_triton(flat)

        # Return results with exact shapes and dtypes as original
        # sorted_token_indices should be int32 in original; torch.argsort returns long. Cast to int32.
        return sorted_token_indices.to(torch.int32), expert_offsets