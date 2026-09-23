import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Compute histogram of flat indices (int32) into counts_ptr[0..255] using atomic adds.
    flat_ptr: pointer to int32 flat array of length N
    counts_ptr: pointer to int32 counts array of length 256
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load values with mask; other=0 for out-of-range lanes
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
    # Ensure integer type and mask invalid lanes to 0
    # Triton will cast vals to int32 for indexing
    # Atomic add only for valid lanes
    # Note: indices are in [0, 255] as per original get_inputs
    for i in range(0, BLOCK):
        if mask[i]:
            idx = vals[i].to(tl.int32)
            # Atomic add into counts[idx]
            tl.atomic_add(counts_ptr + idx, 1)


@triton.jit
def inclusive_scan_prefix_sum(counts_ptr, offsets_ptr, M: tl.int32):
    """
    Compute inclusive prefix sum of counts_ptr[0..M-1] into offsets_ptr[0..M].
    offsets_ptr[0] = 0
    offsets_ptr[i] = offsets_ptr[i-1] + counts_ptr[i-1], for i in [1..M]
    """
    # Single program instance performs the scan sequentially over M.
    running = 0
    offsets_ptr[0] = 0
    for i in range(0, M):
        running += counts_ptr[i]
        offsets_ptr[i + 1] = running


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version:
        - Uses torch.argsort for stable sort permutation (to ensure correctness).
        - Uses Triton kernels to compute histogram and prefix sum for offsets.
        """
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels."
        # Flatten indices (int32)
        flat = topk_idx.reshape(-1).contiguous()  # N can be large
        N = flat.numel()
        device = flat.device

        # 1) Stable sort permutation via PyTorch (ensures correctness)
        # Values are indices in [0, 255]; stable=True preserves order ties.
        sorted_token_indices = torch.argsort(flat.long(), stable=True).to(torch.int32)

        # 2) Histogram via Triton (counts per expert index 0..255)
        counts = torch.zeros(256, dtype=torch.int32, device=device)

        # Launch histogram kernel with a block size tuned for throughput
        BLOCK = 4096  # large block for fewer programs; mask handles tail
        grid = (triton.cdiv(N, BLOCK),)
        histogram_kernel[grid](flat, counts, N=N, BLOCK=BLOCK, num_warps=8)

        # 3) Prefix sum (offsets) via Triton inclusive scan
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        inclusive_scan_prefix_sum[(1,)](counts, offsets, M=256, num_warps=1)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
