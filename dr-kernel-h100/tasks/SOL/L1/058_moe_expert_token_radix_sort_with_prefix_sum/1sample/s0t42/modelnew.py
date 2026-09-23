import torch
import triton
import triton.language as tl


@triton.jit
def _odd_even_stable_argsort(values_ptr, indices_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Triton kernel: perform stable odd-even transposition sort on 'values' and track indices permutation.
    - values_ptr: int32 *, flattened values to sort
    - indices_ptr: int32 *, output permutation of length N
    - N: total number of elements
    - BLOCK: lane count; we run a single program with BLOCK lanes and mask lanes >= N
    """
    # This kernel uses a single program instance with BLOCK lanes to emulate the odd-even passes.
    # We operate only on lanes < N.
    lanes = tl.arange(0, BLOCK)
    mask = lanes < N

    # Copy values into working buffer and initialize indices
    # We'll update 'values_ptr' and 'indices_ptr' in-place.
    # First, read current values and set indices = lanes for lanes < N.
    # We'll perform N phases; each phase does even and odd passes.
    # The kernel uses vectorized updates for all pairs at once.

    # We'll do the sorting in-place by repeatedly comparing adjacent pairs.
    # To implement odd-even, we need to perform even-phase and odd-phase alternately.
    # For simplicity and correctness, we emulate both passes by pairwise updates.
    # Note: Using a loop of N iterations; this is acceptable for benchmark sizes.

    # Initialize indices to [0..BLOCK-1]; only lanes < N are valid.
    tl.store(indices_ptr + lanes, lanes, mask=mask)

    # Prepare a working copy of values into a temporary vector 'v' for updates.
    # Since Triton operates on pointers, we use tl.load/tl.store on values_ptr to read and write.
    v = tl.load(values_ptr + lanes, mask=mask, other=0)  # vector of length BLOCK, int32

    # Perform N phases of odd-even sorting
    # Even phase: compare (0,1), (2,3), ...
    # Odd phase: compare (1,2), (3,4), ...
    # We update both values and indices in each phase.
    for t in range(0, N):  # N phases
        # Even pass: pairs (i, i+1) where i is even
        # Odd pass: pairs (i, i+1) where i is odd
        # Implement by computing partner indices and conditional swaps.
        # Note: Triton does not support dynamic if on scalar t for control flow per lane, so we use uniform phases.
        # We'll do both passes in each iteration: the loop will alternate behavior via even/odd t.

        # Even pass: pairs (0,1), (2,3), ...
        # For i even: i = 2*k
        for k in range(0, BLOCK // 2):
            i = 2 * k
            j = i + 1
            # guard lanes within N
            mask_i = i < N
            mask_j = j < N
            if mask_i and mask_j:
                vi = tl.load(values_ptr + i)
                vj = tl.load(values_ptr + j)
                idx_i = tl.load(indices_ptr + i)
                idx_j = tl.load(indices_ptr + j)
                swap = vi > vj  # stable: no swap on equality
                new_i = tl.where(swap, vj, vi)
                new_j = tl.where(swap, vi, vj)
                new_idx_i = tl.where(swap, idx_j, idx_i)
                new_idx_j = tl.where(swap, idx_i, idx_j)
                tl.store(values_ptr + i, new_i)
                tl.store(values_ptr + j, new_j)
                tl.store(indices_ptr + i, new_idx_i)
                tl.store(indices_ptr + j, new_idx_j)

        # Odd pass: pairs (1,2), (3,4), ...
        for k in range(0, BLOCK // 2):
            i = 2 * k + 1
            j = i + 1
            mask_i = i < N
            mask_j = j < N
            if mask_i and mask_j:
                vi = tl.load(values_ptr + i)
                vj = tl.load(values_ptr + j)
                idx_i = tl.load(indices_ptr + i)
                idx_j = tl.load(indices_ptr + j)
                swap = vi > vj  # stable: no swap on equality
                new_i = tl.where(swap, vj, vi)
                new_j = tl.where(swap, vi, vj)
                new_idx_i = tl.where(swap, idx_j, idx_i)
                new_idx_j = tl.where(swap, idx_i, idx_j)
                tl.store(values_ptr + i, new_i)
                tl.store(values_ptr + j, new_j)
                tl.store(indices_ptr + i, new_idx_i)
                tl.store(indices_ptr + j, new_idx_j)

    # After N phases, 'indices_ptr' holds the permutation. Return it as the sorted_token_indices.


@triton.jit
def _histogram_atomic(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Triton kernel: compute histogram of values in flat_ptr (int32) into counts_ptr (int32[256]).
    Each element of flat is processed in blocks; atomic_add increments counts[value].
    """
    lanes = tl.arange(0, BLOCK)
    # Iterate over flat in chunks
    for start in range(0, N, BLOCK):
        idx = start + lanes
        mask = idx < N
        # Load a block of values
        vals = tl.load(flat_ptr + idx, mask=mask, other=0)  # int32
        # For each value in the block, atomically add to counts[vals]
        # Note: vals are in [0..255], so within bounds for counts_ptr
        # Use masked atomic_add for lanes < N
        # Triton supports atomic_add with int32 pointers
        for k in range(0, BLOCK):
            # Each lane performs its atomic add if valid
            if (start + k) < N:
                val = vals[k]
                # Atomic add into counts[val]
                tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def _inclusive_scan_prefix_sum_single(counts_ptr, offsets_ptr, M: tl.int32):
    """
    Triton kernel: single-program inclusive scan over counts_ptr[M] into offsets_ptr[M+1].
    offsets_ptr[0] must be set to 0 on host before launch.
    """
    # M = 256 (num_experts), scan produces offsets[0..256] with offsets[0]=0
    acc = 0
    for i in range(0, M):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + i + 1, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
          - Computes stable argsort permutation of flattened topk_idx via Triton.
          - Builds histogram of values and prefix offsets via Triton.
        Returns:
          sorted_token_indices: int32 tensor of shape (N,)
          expert_offsets: int32 tensor of shape (num_experts+1,) where num_experts=256
        """
        # Ensure device and dtype
        assert topk_idx.is_cuda, "Input must be on CUDA device for Triton kernels"
        topk_idx = topk_idx.contiguous()
        flat = topk_idx.view(-1)
        device = flat.device
        N = flat.numel()
        num_experts = 256

        # 1) Stable argsort via Triton odd-even transposition sort
        # Allocate working values buffer (copy of flat) and output indices
        values = flat.clone()  # int32 on device
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=device)

        # Choose BLOCK large enough to cover typical N; here we use 8192 lanes
        BLOCK = 8192
        _odd_even_stable_argsort[(1,)](values, sorted_token_indices, N, BLOCK=BLOCK, num_warps=8)

        # 2) Histogram via Triton atomics
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        BLOCK_H = 1024
        grid_hist = (triton.cdiv(N, BLOCK_H),)
        _histogram_atomic[grid_hist](flat, counts, N=N, BLOCK=BLOCK_H, num_warps=8)

        # 3) Prefix sum (offsets) via Triton inclusive scan over 256 elements
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0
        _inclusive_scan_prefix_sum_single[(1,)](counts, offsets, M=num_experts, num_warps=1)

        return sorted_token_indices, offsets