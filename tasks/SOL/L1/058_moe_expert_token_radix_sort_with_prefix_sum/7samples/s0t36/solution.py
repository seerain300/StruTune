import torch
import triton
import triton.language as tl


@triton.jit
def bincount_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    """
    Bincount of flattened int32 values in [0, 255].
    - flat_ptr: *int32, pointer to flattened indices
    - counts_ptr: *int32, pointer to counts[0..255]
    - N: int32, number of elements in flat
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # Load int32 values; masked loads fill with 0 for out-of-range
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)

    # Accumulate atomic adds per bin; only count in-range elements
    for i in range(256):
        # Create a boolean: (vals == i) & mask
        is_i = (vals == i) & mask
        # Convert boolean to int and sum per lane, then atomic add to global counts[i]
        add = is_i.to(tl.int32)
        # Note: Triton atomic_add expects same dtype as counts_ptr (int32 here)
        tl.atomic_add(counts_ptr + i, tl.sum(add, axis=0))


@triton.jit
def inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, L: tl.constexpr):
    """
    Compute inclusive prefix sum of counts (int32) into offsets (int64).
    - counts_ptr: *int32
    - offsets_ptr: *int64, length L (we use L=257)
    """
    # Single-program loop to compute prefix sum
    acc_i32 = tl.zeros((), dtype=tl.int32)
    for k in range(L):
        acc_i32 += tl.load(counts_ptr + k)
        # Write as int64 to ensure offsets are int64
        tl.store(offsets_ptr + k, acc_i32.to(tl.int64))


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version:
        - Triton: bincount of expert ids in [0, 255] and inclusive prefix sum of counts (offsets).
        - PyTorch: stable argsort for flattened indices to produce sorted_token_indices.
        """
        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # 1) Triton bincount: counts[0..255] as int32
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        bincount_kernel[grid](flat, counts, N, BLOCK=BLOCK)

        # 2) Triton inclusive prefix sum to produce offsets (length 257, int64)
        offsets = torch.empty(257, dtype=torch.int64, device=flat.device)
        offsets[0] = 0  # inclusive, starting at 0
        inclusive_prefix_sum_kernel[(1,)](counts, offsets, L=257)

        # 3) Use PyTorch for stable argsort of flattened indices (returns permutation of [0, N-1])
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
