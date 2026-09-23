import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_atomic(values_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    """
    Histogram of values in [0..255] using atomic adds.
    Each program processes BLOCK elements.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # Load values; if mask is False, load 0 (not used in stores)
    vals = tl.load(values_ptr + offsets, mask=mask, other=0)
    # Atomic add 1 for each valid lane
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def _inclusive_scan_prefix_sum(counts_ptr, offsets_ptr, M: tl.constexpr):
    """
    Compute inclusive prefix sum over counts[0..M-1] into offsets[1..M],
    with offsets[0] already set to 0 in host code.
    """
    acc = tl.zeros((), dtype=tl.int32)
    # Loop over M entries
    for i in range(0, M):
        ci = tl.load(counts_ptr + i)
        acc += ci
        tl.store(offsets_ptr + i + 1, acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure CUDA tensor
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device"
        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # 1) Stable sort permutation via PyTorch (robust and correct)
        sorted_token_indices = torch.argsort(flat, stable=True).to(torch.int32)

        # 2) Histogram via Triton (counts per expert id in [0..255])
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _histogram_atomic[grid](flat, counts, N, BLOCK=BLOCK, num_warps=8)

        # 3) Prefix sum (offsets) via Triton inclusive scan over counts
        offsets = torch.empty(257, dtype=torch.int32, device=flat.device)
        offsets[0] = 0  # initialize first offset
        _inclusive_scan_prefix_sum[(1,)](counts, offsets, M=256, num_warps=1)

        return sorted_token_indices, offsets