import torch
import triton
import triton.language as tl


@triton.jit
def odd_even_stable_argsort(work_ptr,        # *int32, flattened values buffer (copy of flat)
                             perm_ptr,       # *int32, permutation buffer (length N)
                             N,              # int32, total number of elements
                             T):             # int32, total number of phases = 2*N
    pid = tl.program_id(axis=0)
    # Each program handles a single index position i = pid
    i = pid
    # Loop over phases
    for t in range(0, T):
        # Determine if this is an even or odd phase
        is_even_phase = (t % 2) == 0
        # Compute partner j = i + 1
        j = i + 1
        # Only proceed if i is the "self" of a pair and within bounds
        if (i % 2 == (0 if is_even_phase else 1)) and (i < N) and (j < N):
            # Load values
            val_i = tl.load(work_ptr + i)
            val_j = tl.load(work_ptr + j)
            idx_i = tl.load(perm_ptr + i)
            idx_j = tl.load(perm_ptr + j)
            # Decide swap based on values
            swap_val = val_j < val_i
            swap_val |= (val_j > val_i) & (~swap_val)  # else only if strictly greater
            # For equal values, swap based on indices to ensure stability (smaller index first)
            swap_val |= (val_j == val_i) & (idx_j < idx_i)
            # Write back swaps using atomic ops to avoid races
            # Note: only one of these two branches will be true per pair.
            # If swap:
            new_val_i = val_j
            new_val_j = val_i
            new_idx_i = idx_j
            new_idx_j = idx_i
            # Non-swap:
            # We don't explicitly store; letting the other lane do the swap maintains consistency.
            # Atomic updates:
            tl.atomic_max(work_ptr + i, new_val_i, mask=swap_val)
            tl.atomic_min(work_ptr + i, val_i, mask=~swap_val)  # revert to original if not swapping
            tl.atomic_min(work_ptr + j, new_val_j, mask=swap_val)
            tl.atomic_max(work_ptr + j, val_j, mask=~swap_val)

            # Update permutation accordingly
            tl.atomic_max(perm_ptr + i, new_idx_i, mask=swap_val)
            tl.atomic_min(perm_ptr + i, idx_i, mask=~swap_val)
            tl.atomic_max(perm_ptr + j, new_idx_j, mask=swap_val)
            tl.atomic_min(perm_ptr + j, idx_j, mask=~swap_val)


@triton.jit
def histogram_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32
    # Atomic add into counts
    # counts is length 256, assuming vals in [0..255]
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def inclusive_scan_prefix_sum(counts_ptr, offsets_ptr, M: tl.constexpr):
    # Compute inclusive scan of counts_ptr[0..M-1] into offsets_ptr[1..M]
    # offsets_ptr[0] must be set to 0 on host
    acc = 0
    for k in range(0, M):
        acc += tl.load(counts_ptr + k)
        tl.store(offsets_ptr + k + 1, acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version:
        - Sorts the flattened expert indices using a Triton odd-even transposition sort (stable),
          and returns the permutation (sorted_token_indices).
        - Builds expert_offsets via Triton histogram and prefix sum.
        """
        assert topk_idx.is_cuda, "topk_idx must be on CUDA for Triton kernels."
        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()
        device = flat.device

        # 1) Stable argsort via Triton odd-even sort
        # Work buffer (copy of flat) and permutation buffer
        work = flat.clone().to(torch.int32)
        sorted_token_indices = torch.arange(N, dtype=torch.int32, device=device)
        # Total phases = 2*N
        T = 2 * N
        # Launch 1D grid over N elements; each program handles its own index and pairs it appropriately
        grid = (N,)
        odd_even_stable_argsort[grid](work, sorted_token_indices, N, T, num_warps=4)

        # 2) Histogram via Triton
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        BLOCK = 1024
        grid_hist = (triton.cdiv(N, BLOCK),)
        histogram_kernel[grid_hist](flat, counts, N, BLOCK=BLOCK, num_warps=8)

        # 3) Prefix sum (offsets) via Triton inclusive scan over 256 elements
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        offsets[0] = 0
        inclusive_scan_prefix_sum[(1,)](counts, offsets, M=256, num_warps=1)

        return sorted_token_indices, offsets