import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_atomic_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Compute histogram of flat indices (int32) into counts_ptr[0..255] using atomic adds.
    flat_ptr: *int32, length N
    counts_ptr: *int32, length 256 (we will write into counts_ptr[0..255])
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load a block of flat indices; invalid lanes masked to 0 to avoid OOB
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
    # Only count if 0 <= val <= 255; otherwise ignore
    in_range = (vals >= 0) & (vals <= 255)
    # Ensure lanes out of N are 0
    vals = tl.where(mask, vals, 0)

    # Atomic add for valid in-range values
    # Note: we assume counts_ptr is int32 and zero-initialized.
    # For vals that are not in [0, 255], atomic_add with 0 does nothing.
    # For valid vals, atomic_add increments counts[vals].
    tl.atomic_add(counts_ptr + vals, 1, mask=in_range & mask)


@triton.jit
def _inclusive_scan_prefix_sum(counts_ptr, offsets_ptr, M: tl.int32):
    """
    Compute inclusive prefix sum of counts_ptr[0..M-1] into offsets_ptr[0..M], with offsets_ptr[0] = 0.
    offsets_ptr[M] (if allocated) is not written by this kernel; we pass M+1 and set offsets_ptr[0] = 0 on host.
    """
    # Single-program kernel; i ranges 0..M-1
    total = 0
    # Note: Triton kernels run sequentially for this loop; M is small (256).
    for i in range(0, M):
        # Load counts[i]
        cnt = tl.load(counts_ptr + i)
        total += cnt
        tl.store(offsets_ptr + i + 1, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version:
        - Compute argsort via torch (stable) to ensure correctness/performance.
        - Histogram of indices and offsets via Triton kernels.
        Returns:
          sorted_token_indices: torch.int32, permutation of indices length N
          expert_offsets: torch.int32, length num_experts+1 (257)
        """
        # 1) Flatten and compute stable argsort using torch (robust and fast)
        flat = topk_idx.reshape(-1)
        # Ensure dtype int32 for comparison and counting
        flat = flat.to(torch.int32)
        N = flat.numel()
        # Stable sort permutation
        sorted_token_indices = torch.argsort(flat, stable=True).to(torch.int32)

        # 2) Histogram via Triton
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        BLOCK = 1024
        grid_hist = (triton.cdiv(N, BLOCK),)
        _histogram_atomic_kernel[grid_hist](flat, counts, N=N, BLOCK=BLOCK, num_warps=8)

        # 3) Prefix sum (offsets) via Triton
        offsets = torch.empty(257, dtype=torch.int32, device=flat.device)
        offsets[0] = 0  # offsets[0] = 0 by definition
        _inclusive_scan_prefix_sum[(1,)](counts, offsets, M=256, num_warps=1)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
