import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(original_ptr, counts_ptr, M, BLOCK: tl.constexpr):
    """
    Histogram of int32 values in original_ptr[0:M] into counts_ptr[0:256].
    For each element, atomic_add 1 to counts[value].
    Grid: (grid_size,)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M
    vals = tl.load(original_ptr + offsets, mask=mask, other=0)  # int32 loads
    for j in range(BLOCK):
        idx = offsets[j]
        if mask[j]:
            val = vals[j]
            tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def exclusive_cumsum_kernel(prefix_ptr, N: tl.constexpr):
    """
    Exclusive prefix sum of prefix_ptr of length N.
    For v in [0..N-2]: prefix[v+1] += prefix[v].
    Launch with grid=(1,).
    """
    for v in range(0, N - 1):
        prev = tl.load(prefix_ptr + v)
        tl.store(prefix_ptr + v + 1, prev + prev)


@triton.jit
def assemble_offsets_kernel(offsets_ptr, counts_ptr, N: tl.constexpr):
    """
    Assemble offsets:
    offsets[0] = 0; offsets[i+1] = offsets[i] + counts[i] for i in [0..N-1]
    Launch with grid=(1,), N is constexpr for loop unrolling.
    """
    tl.store(offsets_ptr + 0, 0)
    for i in range(0, N):
        prev = tl.load(offsets_ptr + i)
        c = tl.load(counts_ptr + i)
        tl.store(offsets_ptr + i + 1, prev + c)


@triton.jit
def stable_permutation_kernel(original_ptr, out_ptr, prefix_ptr, M, BLOCK: tl.constexpr):
    """
    Stable permutation indices:
    For each position i, value v = original_ptr[i], place i at position prefix[v],
    where prefix[v] = number_of_less(v) (exclusive scan).
    Grid: (grid_size,)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M
    vals = tl.load(original_ptr + offsets, mask=mask, other=0)  # int32 values
    for j in range(BLOCK):
        idx = offsets[j]
        if mask[j]:
            v = vals[j]
            number_of_less = tl.load(prefix_ptr + v)  # int32
            tl.store(out_ptr + number_of_less, idx)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only forward:
        - Compute counts via histogram kernel.
        - Compute sorted_token_indices via stable permutation kernel.
        - Compute expert offsets via prefix sum and assembly kernels.
        No torch ops are used for numerical computation.
        """
        # Ensure tensor is on CUDA
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.cuda()

        # Flatten
        original_flat = topk_idx.reshape(-1).contiguous()
        M = original_flat.numel()

        # Counts per value (int32, length 256)
        counts = torch.zeros(256, dtype=torch.int32, device=original_flat.device)

        # Launch histogram kernel
        BLOCK = 1024
        grid_size = triton.cdiv(M, BLOCK)
        histogram_kernel[(grid_size,)](original_flat, counts, M, BLOCK=BLOCK)

        # Allocate sorted_token_indices (int32)
        sorted_token_indices_int32 = torch.empty(M, dtype=torch.int32, device=original_flat.device)

        # Compute exclusive prefix sum of counts
        prefix = torch.zeros(256, dtype=torch.int32, device=original_flat.device)
        exclusive_cumsum_kernel[(1,)](prefix, N=256)

        # Launch stable permutation kernel
        stable_permutation_kernel[(grid_size,)](original_flat, sorted_token_indices_int32, prefix, M, BLOCK=BLOCK)

        # Convert to int64 to match torch.sort(indices) dtype
        sorted_token_indices = sorted_token_indices_int32.to(torch.int64)

        # Compute expert offsets: offsets[0] = 0; offsets[i+1] = sum_{x<=i} counts_x
        offsets = torch.empty(257, dtype=torch.int32, device=original_flat.device)
        assemble_offsets_kernel[(1,)](offsets, counts, N=256)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
