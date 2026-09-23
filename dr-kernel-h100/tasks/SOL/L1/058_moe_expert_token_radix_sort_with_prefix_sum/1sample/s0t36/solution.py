import torch
import triton
import triton.language as tl


@triton.jit
def _odd_even_argsort_and_indices(values_ptr, indices_ptr, N: tl.int32, BLOCK: tl.int32):
    """
    Triton kernel implementing stable odd-even transposition sort on values_ptr[0..N-1]
    and maintaining the permutation in indices_ptr[0..N-1]. Assumes values_ptr is int32.
    Each iteration does pairs (0,1), (2,3), ... for even phases and (1,2), (3,4), ... for odd phases.
    """
    # We will iterate over phases in a while loop. Triton allows while loops in kernels.
    p = 0
    while p < 2 * N:
        # Determine whether this is an even phase
        even_phase = (p % 2) == 0

        # Compute start index for this phase: 0 for even, 1 for odd
        start = 0 if even_phase else 1

        # Loop over pairs within this phase
        i = start
        while i + 1 < N:
            j = i + 1

            # Load values at i and j
            a = tl.load(values_ptr + i)
            b = tl.load(values_ptr + j)

            # Load original positions (indices) for i and j
            idx_i = tl.load(indices_ptr + i)
            idx_j = tl.load(indices_ptr + j)

            # Compare and decide swap
            # Only swap when a > b. For ties, do not swap to preserve stable order.
            swap = a > b

            # If swap, swap both values and indices
            new_a = tl.where(swap, b, a)
            new_b = tl.where(swap, a, b)
            new_idx_i = tl.where(swap, idx_j, idx_i)
            new_idx_j = tl.where(swap, idx_i, idx_j)

            # Store results back
            tl.store(values_ptr + i, new_a)
            tl.store(values_ptr + j, new_b)
            tl.store(indices_ptr + i, new_idx_i)
            tl.store(indices_ptr + j, new_idx_j)

            # Advance to next pair
            i += 2

        p += 1


@triton.jit
def _histogram_atomic_kernel(values_ptr, counts_ptr, N: tl.int32, BLOCK: tl.int32):
    """
    Triton kernel that builds counts[0..255] of values_ptr[0..N-1] using atomic adds.
    Assumes values_ptr contains int32 indices in [0..255]. Each program instance processes BLOCK elements.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(values_ptr + offsets, mask=mask, other=0)  # int32 loads

    # Atomic add into counts; counts_ptr is int32[256]
    # Ensure masked lanes don't contribute; vals without mask are 0 for out-of-range.
    for k in range(BLOCK):
        idx = offsets[k]
        if mask[k]:
            tl.atomic_add(counts_ptr + vals[k].to(tl.int32), 1)


@triton.jit
def _inclusive_scan_prefix_sum(counts_ptr, offsets_ptr, M: tl.int32):
    """
    Triton kernel to compute inclusive prefix sum of counts_ptr[0..M-1] into offsets_ptr[1..M].
    offsets_ptr[0] must be set to 0 on host.
    We implement a simple sequential loop across M (M=256 is small). The kernel runs with grid=(1,)
    and num_warps=1. Each iteration loads the current prefix and adds counts[i] to it, storing to offsets[i+1].
    """
    running = 0
    i = 0
    while i < M:
        val = tl.load(counts_ptr + i)
        running += val
        tl.store(offsets_ptr + i + 1, running)
        i += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # Ensure CUDA and int32
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device"
        # Flatten
        flat = topk_idx.reshape(-1)
        if flat.dtype != torch.int32:
            flat = flat.to(torch.int32)
        device = flat.device
        N = flat.numel()

        # 1) Stable sort permutation via Triton odd-even sort
        # Prepare working buffers
        values = flat.clone()  # int32, same device
        # Initialize indices buffer: original positions [0..N-1]
        indices = torch.arange(N, dtype=torch.int32, device=device)

        # Triton kernel launch: iterate phases in kernel (BLOCK is ignored in this kernel, but pass a placeholder)
        _odd_even_argsort_and_indices[(1,)](values, indices, N, BLOCK=1)

        # 2) Histogram via Triton
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        # One program instance suffices; BLOCK controls internal vector size, but histogram is simple.
        _histogram_atomic_kernel[(1,)](flat, counts, N, BLOCK=1024)

        # 3) Prefix sum (offsets) via Triton
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        offsets[0] = 0
        _inclusive_scan_prefix_sum[(1,)](counts, offsets, M=256)

        return indices, offsets


def run(*args):
    return ModelNew()(*args)
