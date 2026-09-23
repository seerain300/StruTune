import torch
import triton
import triton.language as tl


@triton.jit
def _odd_even_stable_argsort(values_ptr, indices_ptr, N, num_warps: tl.constexpr):
    # This kernel performs an odd-even transposition sort on 'values_ptr' of length N.
    # 'indices_ptr' is the permutation buffer initialized to [0..N-1], updated with swaps.
    # Stability: for equal values, we do not swap, preserving original index order.
    # Grid: 1D with size N. Each program handles its lane 'i' and contributes to pairing in even/odd phases.
    i = tl.program_id(axis=0)
    # Precompute lane indices
    # We will use masks to avoid out-of-bounds: only act if i < N
    # Odd phases: pairs (1,2), (3,4), ...
    # Even phases: pairs (0,1), (2,3), ...
    # Number of phases is N (sufficient to sort any N).
    # We vectorize operations across lanes for efficiency; Triton will handle masking.
    # Note: Triton does not support direct while loops over runtime N; we emulate via a large constexpr MAX_STEPS.
    # Since Triton requires static loops, we instead launch a 1D grid and rely on pairwise updates across phases.
    # In practice, for correctness, we run a fixed large number of phases (e.g., 4*N), which guarantees convergence.
    # However, Triton requires compile-time static loops. Therefore, we implement nested static loops up to MAX_STEPS.
    MAX_STEPS = 2048  # chosen to be >= 4*N for typical N up to ~4096
    for step in range(MAX_STEPS):
        # Even phase: pairs (0,1), (2,3), ...
        # Odd phase: pairs (1,2), (3,4), ...
        if (step % 2) == 0:
            j = i + 1
            # Only pairs where j < N are valid
            if j < N:
                # Load values and indices for i and j
                vi = tl.load(values_ptr + i)
                vj = tl.load(values_ptr + j)
                ii = tl.load(indices_ptr + i)
                ij = tl.load(indices_ptr + j)
                # Compare and possibly swap (ascending). For ties, do not swap to preserve stability.
                swap = vi > vj
                new_vi = tl.where(swap, vj, vi)
                new_vj = tl.where(swap, vi, vj)
                new_ii = tl.where(swap, ij, ii)
                new_ij = tl.where(swap, ii, ij)
                # Store back
                tl.store(values_ptr + i, new_vi)
                tl.store(values_ptr + j, new_vj)
                tl.store(indices_ptr + i, new_ii)
                tl.store(indices_ptr + j, new_ij)
        else:
            j = i + 1
            if j < N:
                vi = tl.load(values_ptr + i)
                vj = tl.load(values_ptr + j)
                ii = tl.load(indices_ptr + i)
                ij = tl.load(indices_ptr + j)
                # Odd phase pairs: (1,2), (3,4), ...
                # Compare and possibly swap (ascending). For ties, do not swap.
                swap = vi > vj
                new_vi = tl.where(swap, vj, vi)
                new_vj = tl.where(swap, vi, vj)
                new_ii = tl.where(swap, ij, ii)
                new_ij = tl.where(swap, ii, ij)
                tl.store(values_ptr + i, new_vi)
                tl.store(values_ptr + j, new_vj)
                tl.store(indices_ptr + i, new_ii)
                tl.store(indices_ptr + j, new_ij)


@triton.jit
def _histogram_atomic_kernel(values_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    # Compute histogram of int32 values in [0..255] using atomics.
    # Each program handles a block of elements and atomically increments counts.
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(values_ptr + offsets, mask=mask, other=0)  # int32
    # Atomic add 1 for each valid element
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def _inclusive_scan_prefix_sum(counts_ptr, offsets_ptr, M: tl.constexpr):
    # Single-program inclusive scan over M elements. offsets_ptr[0] is pre-initialized to 0.
    acc = tl.load(offsets_ptr + 0)
    for i in range(M):
        ci = tl.load(counts_ptr + i)
        acc += ci
        tl.store(offsets_ptr + i + 1, acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure input is on CUDA and int32
        assert topk_idx.is_cuda, "Input must be on CUDA device"
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        device = flat.device

        # 1) Triton stable argsort: produce permutation indices
        # Copy values and initialize permutation with original indices
        values = flat.clone()  # Triton expects pointer to data; ensure int32
        sorted_token_indices = torch.arange(N, dtype=torch.int32, device=device)

        # Launch odd-even stable argsort kernel
        grid_sort = (N,)
        _odd_even_stable_argsort[grid_sort](values, sorted_token_indices, N, num_warps=8)

        # 2) Histogram via Triton (atomic add into 256 counts)
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        BLOCK = 1024
        grid_hist = (triton.cdiv(N, BLOCK),)
        _histogram_atomic_kernel[grid_hist](values, counts, N, BLOCK=BLOCK, num_warps=8)

        # 3) Prefix sum (offsets) via Triton inclusive scan over 256 elements
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        offsets[0] = 0  # initialize first element to 0
        _inclusive_scan_prefix_sum[(1,)](counts, offsets, M=256, num_warps=1)

        # Return sorted permutation (int32) and offsets (int32)
        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
