import torch
import triton
import triton.language as tl


@triton.jit
def histogram_atomic_kernel(x_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    """
    Build a histogram of int32 values in x_ptr (length N) into counts_ptr[0..255].
    Each program handles BLOCK elements; masked loads avoid OOB.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load int32 values; for masked lanes, load 0 (won't contribute due to mask)
    vals = tl.load(x_ptr + offsets, mask=mask, other=0)

    # Atomic add to counts[vals]; counts is int32
    # Ensure vals are in [0, 255] (per original code). Mask prevents OOB.
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def inclusive_scan_prefix_sum_kernel(counts_ptr, offsets_ptr, M: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr[0..M-1] into offsets_ptr[0..M].
    offsets_ptr[0] must be initialized to 0 by host. Single program instance (grid=(1,))
    loops over M and updates offsets.
    """
    total = 0
    # Loop over 256 elements: i in [0, M)
    for i in range(0, M):
        # Load count[i]
        count_i = tl.load(counts_ptr + i)
        # Update total (inclusive scan)
        total += count_i
        # Store total at offsets[i+1]
        tl.store(offsets_ptr + i + 1, total)


def _triton_histogram_and_offsets(flat: torch.Tensor):
    """
    Helper that launches Triton kernels for histogram and prefix sum.
    Returns counts (int32, length 256) and offsets (int32, length 257).
    """
    device = flat.device
    N = flat.numel()

    # Histogram counts for indices in [0..255]
    counts = torch.zeros(256, dtype=torch.int32, device=device)

    # Configure and launch histogram kernel
    BLOCK = 4096  # large block; mask handles tail
    grid = (triton.cdiv(N, BLOCK),)
    histogram_atomic_kernel[grid](flat, counts, N, BLOCK=BLOCK, num_warps=8)

    # Compute inclusive prefix sum into offsets
    offsets = torch.empty(257, dtype=torch.int32, device=device)
    offsets[0] = 0
    inclusive_scan_prefix_sum_kernel[(1,)](counts, offsets, M=256, num_warps=1)

    return counts, offsets


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version:
        - Uses torch.argsort(stable=True) for the permutation to ensure correctness.
        - Uses Triton kernels for histogram and prefix sum (offsets).
        """
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels."
        # Flatten and ensure contiguous
        flat = topk_idx.reshape(-1).contiguous()
        # Stable sort permutation via PyTorch (exact behavior)
        sorted_token_indices = torch.argsort(flat.long(), stable=True).to(torch.int32)

        # Triton histogram and offsets
        counts, offsets = _triton_histogram_and_offsets(flat)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
