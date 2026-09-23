import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_kernel(flat_ptr, N, counts_ptr, BLOCK: tl.constexpr):
    """
    Histogram of 32-bit integer ids in flat_ptr into counts_ptr[0..255].
    Each program handles BLOCK elements; for each id in 0..255:
      - build mask (flat == id) & (offs < N), sum it, atomic_add to counts[id].
    This minimizes atomic operations per program.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Load flat values (int32)
    x = tl.load(flat_ptr + offs, mask=mask, other=0)

    # Process each expert id 0..255
    for i in range(256):
        eq = (x == i) & mask
        # Reduce across the block to get count for this id
        cnt = tl.sum(eq.to(tl.int32), axis=0)
        # Atomic add count to global counts[i]
        tl.atomic_add(counts_ptr + i, cnt)


@triton.jit
def _compute_offset_for_bin(counts_ptr, offsets_ptr, bin_idx, carry_ptr):
    """
    Helper to compute offsets[bin_idx+1] given counts[bin_idx] and current carry.
    We update offsets[bin_idx+1] in-place. This is a single-bin scan with carry.
    """
    # Load current counts[bin_idx]
    curr = tl.load(counts_ptr + bin_idx)
    # Load previous carry (sum of counts[0..bin_idx-1])
    prev_carry = tl.load(carry_ptr)
    # left = counts - carry
    left_i = curr - prev_carry
    # Update offsets[bin_idx+1] = prev_offset + left
    prev_offset = tl.load(offsets_ptr + bin_idx)
    new_offset = prev_offset + left_i
    tl.store(offsets_ptr + (bin_idx + 1), new_offset)
    # Update carry for next bins: sum of processed bins so far
    new_carry = prev_carry + left_i
    tl.store(carry_ptr, new_carry)


class ModelNew(torch.nn.Module):
    """
    Triton-optimized version:
    - Uses a Triton histogram kernel to count per-expert tokens.
    - Computes per-expert inclusive offsets with Triton via a single-pass scan per bin.
    - Keeps stable sorting in PyTorch to match original behavior precisely.
    Handles CPU input by moving to CUDA inside forward since Triton runs on GPU.
    """

    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # Move to CUDA if needed
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to(device="cuda")
        topk_idx = topk_idx.contiguous()

        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # Ensure dtype int32 for histogram
        flat_i32 = flat.to(torch.int32)

        # Triton histogram per expert id (0..255). counts buffer
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)

        # Kernel launch configuration: process 1024 elements per program
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)

        _histogram_kernel[grid](flat_i32, N, counts, BLOCK=BLOCK, num_warps=4)

        # Compute per-expert offsets using Triton (inclusive prefix sum)
        num_experts = 256
        offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
        # carry buffer (single int32) initialized to 0
        carry = torch.zeros(1, dtype=torch.int32, device=flat.device)

        # Single-pass per-bin scan: compute offsets[i+1] incrementally
        for i in range(num_experts):
            _compute_offset_for_bin(counts, offsets, i, carry)

        # Stable sort of flattened indices (PyTorch), matching original behavior
        sorted_token_indices = flat.sort(stable=True)[1].to(torch.int32)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
