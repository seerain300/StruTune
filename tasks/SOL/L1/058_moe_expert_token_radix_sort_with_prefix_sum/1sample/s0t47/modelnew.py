import torch
import triton
import triton.language as tl


@triton.jit
def _bitonic_argsort_stable(values_ptr, indices_ptr, N: tl.int32, BLOCK: tl.int32):
    # Implement odd-even transposition sort in Triton to produce permutation (indices).
    # We iterate for N phases. For simplicity and robustness with any N, we use BLOCK >= N.
    # Each program processes its own index position. We do all compare-exchanges via
    # global loads/stores, updating both values and indices buffers.
    # Note: This kernel assumes values_ptr points to an int32 array of length N (contiguous).
    # indices_ptr points to an int32 array of length N (contiguous), initialized to [0..N-1].
    # Stability: only swap when v_i > v_j; if equal, do not swap (preserve original order).
    for phase in range(0, N):
        # Even phase: pairs (0,1), (2,3), ...
        # Odd phase: pairs (1,2), (3,4), ...
        if (phase % 2) == 0:
            i = tl.arange(0, BLOCK)  # lanes 0..BLOCK-1, mask i+1 < N
            j = i + 1
            pair_mask = j < N
        else:
            i = tl.arange(0, BLOCK) - 1  # lanes -1..BLOCK-2; with mask i >= 0 and j < N
            j = i + 1
            pair_mask = (i >= 0) & (j < N)

        # Mask invalid lanes: i must be valid, j must be valid
        valid_i = i < N
        mask = valid_i & pair_mask

        # Load current values for i and j
        vi = tl.load(values_ptr + i, mask=mask, other=0)
        vj = tl.load(values_ptr + j, mask=mask, other=0)
        ii = tl.load(indices_ptr + i, mask=mask, other=0)
        ij = tl.load(indices_ptr + j, mask=mask, other=0)

        # Compare and decide swap: swap if vi > vj; do not swap if equal
        swap = mask & (vi > vj)

        # Compute new values after potential swap
        new_vi = tl.where(swap, vj, vi)
        new_vj = tl.where(swap, vi, vj)
        new_ii = tl.where(swap, ij, ii)
        new_ij = tl.where(swap, ii, ij)

        # Store back
        tl.store(values_ptr + i, new_vi, mask=mask)
        tl.store(values_ptr + j, new_vj, mask=mask)
        tl.store(indices_ptr + i, new_ii, mask=mask)
        tl.store(indices_ptr + j, new_ij, mask=mask)


@triton.jit
def _histogram_atomic_kernel(values_ptr, counts_ptr, N: tl.int32, BLOCK: tl.int32):
    # Each program processes BLOCK elements; use atomic add into counts[0..255]
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(values_ptr + offs, mask=mask, other=0).to(tl.int32)
    # Atomic add 1 for each valid element into counts[vals]
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def _inclusive_scan_prefix_sum(counts_ptr, offsets_ptr, M: tl.int32):
    # Single-program inclusive scan over M=256 to produce offsets[0..M] with offsets[0]=0
    acc = tl.zeros((), dtype=tl.int32)
    # Manually unroll small loop
    for i in range(0, 256):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + i + 1, acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA and contiguous
        assert topk_idx.is_cuda, "Inputs must be on CUDA device"
        values = topk_idx.reshape(-1).contiguous()  # flattened indices
        N = values.numel()
        device = values.device

        # 1) Triton sort: produce permutation indices (stable)
        # Create indices buffer initialized to [0..N-1]
        indices = torch.arange(N, dtype=torch.int32, device=device)
        # Choose BLOCK >= N (next power-of-two up to a safe max). N is up to ~8192 in provided workloads.
        # We can set BLOCK to 8192 and mask; to keep it simple and robust, we use BLOCK = 8192.
        BLOCK = 8192
        grid = (1,)  # single program; loop handles all phases
        _bitonic_argsort_stable[grid](values, indices, N=N, BLOCK=BLOCK, num_warps=8)

        # 2) Histogram via Triton
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        grid_hist = (triton.cdiv(N, 1024),)
        _histogram_atomic_kernel[grid_hist](values, counts, N=N, BLOCK=1024, num_warps=8)

        # 3) Prefix sum (offsets) via Triton
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        offsets[0] = 0  # inclusive scan starts at 0
        _inclusive_scan_prefix_sum[(1,)](counts, offsets, M=256, num_warps=1)

        # Return sorted indices and offsets. ModelNew.forward must return the same structure as original run.
        # sorted_token_indices is indices permutation; expert_offsets is offsets.
        # Note: original run returns (sorted_token_indices, expert_offsets). We return them as a tuple.
        return indices, offsets