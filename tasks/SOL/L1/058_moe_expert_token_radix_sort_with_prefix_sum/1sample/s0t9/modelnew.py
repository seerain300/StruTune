import torch
import triton
import triton.language as tl


@triton.jit
def histogram_atomic_kernel(
    x_ptr,            # *const int32, flattened indices
    counts_ptr,       # *int32, length 256
    N,                # int32, number of elements
    BLOCK: tl.constexpr,
):
    """
    Compute histogram of values in x_ptr (int32) into counts_ptr (int32[256]).
    Each program handles BLOCK elements via masked loads and atomic adds.
    Assumes values are in [0, 255], consistent with get_inputs().
    """
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load a block of values; other=0 for masked lanes
    vals = tl.load(x_ptr + offsets, mask=mask, other=0)
    # Convert to int32 in case they're not
    vals = vals.to(tl.int32)

    # For each valid lane, atomic add 1 into counts[vals]
    # This produces correct histogram for any N, as long as vals in [0, 255].
    for i in range(BLOCK):
        idx = offsets[i]
        v = vals[i]
        if mask[i]:
            tl.atomic_add(counts_ptr + v, 1)


@triton.jit
def inclusive_scan_prefix_sum(
    counts_ptr,       # *const int32, length 256
    offsets_ptr,      # *int32, length 257
    M: tl.constexpr,  # int32, number of bins = 256
):
    """
    Compute inclusive prefix sum of counts_ptr into offsets_ptr[1..].
    offsets_ptr[0] = 0, and offsets_ptr[i] = sum_{j=0..i-1} counts[j].
    """
    # offsets_ptr is 1-based for prefix sums, 0 reserved for zero.
    total = 0
    # Simple sequential scan; M is small (256), so this is fine.
    for i in range(M):
        total += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + i + 1, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Compute:
          - sorted_token_indices: int32[flattened_size] — stable sort permutation of flattened indices.
          - expert_offsets: int32[num_experts+1] — prefix sum of counts of each expert id (0..255).
        topk_idx: (batch_size, seq_len, num_experts_per_tok), int32 on CUDA.
        """
        assert topk_idx.is_cuda, "topk_idx must be on CUDA for Triton kernels."
        # Flatten indices and make contiguous
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        device = flat.device

        # 1) Stable sort permutation using PyTorch (guarantees correctness)
        sorted_token_indices = torch.argsort(flat.long(), stable=True).to(torch.int32)

        # 2) Histogram via Triton atomic adds over 256 bins
        counts = torch.zeros(256, dtype=torch.int32, device=device)

        # Launch histogram kernel
        BLOCK = 1024  # block size for parallel processing; 1024 elements per program
        grid = (triton.cdiv(N, BLOCK),)
        histogram_atomic_kernel[grid](flat, counts, N, BLOCK=BLOCK, num_warps=8)

        # 3) Prefix sum (offsets) via Triton inclusive scan over 256 counts
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        offsets[0] = 0
        inclusive_scan_prefix_sum[(1,)](counts, offsets, M=256, num_warps=1)

        return sorted_token_indices, offsets