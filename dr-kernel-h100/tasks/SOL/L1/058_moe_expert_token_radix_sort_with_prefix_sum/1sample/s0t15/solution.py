import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_atomic_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Build histogram of flat indices (int32) into counts_ptr[0..255].
    For each element flat[i], atomic_add counts[flat[i]] by 1.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # Load values from flat (int32)
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
    # Ensure we only atomic add for valid offsets
    # Triton will handle out-of-range indices by masking; no need to do anything else.
    # Atomic add into counts[vals]
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def _inclusive_scan_prefix_sum(counts_ptr, offsets_ptr, M: tl.int32):
    """
    Compute inclusive prefix sum of counts_ptr[0..M-1] into offsets_ptr[0..M].
    offsets[0] must be set to 0 on host before launching this kernel.
    For j in 1..M: offsets[j] = offsets[j-1] + counts[j-1].
    """
    # Single program instance performs the scan across 256 elements
    j = 0
    while j < M:
        # Load previous offset and current count
        prev = tl.load(offsets_ptr + j)  # scalar
        cnt = tl.load(counts_ptr + j)    # scalar
        new = prev + cnt
        tl.store(offsets_ptr + j, new)
        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version that:
          - Uses torch.argsort for the stable sort permutation (robust and correct).
          - Uses Triton kernels for histogram (counts per expert id) and prefix sum (offsets).
        """
        # Ensure tensor is on CUDA for Triton kernels
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device"
        # Flatten indices
        flat = topk_idx.reshape(-1)  # int32 on CUDA
        N = flat.numel()
        device = flat.device

        # 1) Stable sort permutation using PyTorch (exact original behavior)
        # Inputs are in [0, 255] so cast to int64 for argsort is fine; return int32 indices
        sorted_token_indices = torch.argsort(flat.long(), stable=True).to(torch.int32)

        # 2) Histogram via Triton
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _histogram_atomic_kernel[grid](flat, counts, N=N, BLOCK=BLOCK, num_warps=8)

        # 3) Prefix sum (offsets) via Triton inclusive scan over 256 elements
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        offsets[0] = 0
        _inclusive_scan_prefix_sum[(1,)](counts, offsets, M=256, num_warps=1)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
