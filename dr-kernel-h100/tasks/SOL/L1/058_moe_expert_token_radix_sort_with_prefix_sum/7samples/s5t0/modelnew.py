import torch
import triton
import triton.language as tl


# Kernel 1: Initialize keys arrays for bitonic sort (stable)
# out_idx: [N] will hold the final sorted indices (we write to it during sort)
# keys_id: [N] holds the expert IDs (same as input vector)
# keys_idx: [N] holds the original position (i)
@triton.jit
def init_keys_kernel(flat_ptr, out_idx_ptr, keys_id_ptr, keys_idx_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # Load IDs
    ids = tl.load(flat_ptr + offsets, mask=mask, other=0)
    # Store into keys
    tl.store(keys_id_ptr + offsets, ids, mask=mask)
    # Store original index
    tl.store(keys_idx_ptr + offsets, offsets, mask=mask)
    # Initialize out_idx as identity
    tl.store(out_idx_ptr + offsets, offsets, mask=mask)


# Kernel 2: Bitonic sort of (keys_id, keys_idx) in ascending order, stable tie-break by idx
# We sort out_idx based on (keys_id, keys_idx). No torch ops on host; only kernel.
# We implement one-pass compare-and-swap using pairs computed with XOR partner indices.
@triton.jit
def bitonic_sort_stable_kernel(keys_id_ptr, keys_idx_ptr, out_idx_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N

    # Bitonic sorting network
    # We run a fixed number of iterations up to N; Triton supports loops and masks.
    # For simplicity, we use while loops; N is constexpr so Triton can unroll or manage well.
    # Note: We only process pairs once to avoid double writes; partner index j = i ^ stride.
    # We do not write to out_idx for i where i >= j to ensure each pair is handled exactly once.
    # But since we initialize out_idx to identity, we can safely write both sides for all i.
    # To be safe, we can use a partner mask and only write when i < j.

    # We need to define loops. Triton prefers for-loops with tl.arange; here we emulate with while.
    size = 2
    while size <= N:
        stride = size // 2
        while stride > 0:
            j = i ^ stride
            partner_mask = j > i  # ensure each pair is processed once
            id_i = tl.load(keys_id_ptr + i, mask=mask & partner_mask, other=0)
            id_j = tl.load(keys_id_ptr + j, mask=mask & partner_mask, other=0)
            idx_i = tl.load(keys_idx_ptr + i, mask=mask & partner_mask, other=0)
            idx_j = tl.load(keys_idx_ptr + j, mask=mask & partner_mask, other=0)

            # Ascending direction for the current block
            asc = ( (i & size) == 0 )

            # Decide swap: swap if (id_i > id_j) or (id_i == id_j and idx_i > idx_j) when ascending,
            # or (id_i < id_j) or (id_i == id_j and idx_i < idx_j) when descending.
            swap_asc = (id_i > id_j) | ((id_i == id_j) & (idx_i > idx_j))
            swap_desc = (id_i < id_j) | ((id_i == id_j) & (idx_i < idx_j))
            swap = tl.where(asc, swap_asc, swap_desc) & partner_mask

            # If swap, exchange out_idx[i] and out_idx[j]
            idx_i_old = tl.load(out_idx_ptr + i, mask=mask & partner_mask, other=0)
            idx_j_old = tl.load(out_idx_ptr + j, mask=mask & partner_mask, other=0)

            new_i = tl.where(swap, idx_j_old, idx_i_old)
            new_j = tl.where(swap, idx_i_old, idx_j_old)

            tl.store(out_idx_ptr + i, new_i, mask=mask & partner_mask)
            tl.store(out_idx_ptr + j, new_j, mask=mask & partner_mask)

            stride = stride // 2
        size = size * 2


# Kernel 3: Count per-expert occurrences of IDs; out_counts: [num_experts]
@triton.jit
def count_per_expert_kernel(flat_ptr, out_counts_ptr, num_experts: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    e = tl.program_id(0)  # one program per expert
    # Accumulate count in a scalar
    cnt = tl.zeros((), dtype=tl.int32)
    start = 0
    while start < N:
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < N
        ids = tl.load(flat_ptr + offsets, mask=mask, other=0)
        # Mask where ids == e
        eq = ids == e
        # Sum matches in this block; we need a reduction. Triton supports tl.sum over vectors.
        cnt += tl.sum(tl.where(mask & eq, 1, 0), axis=0)
        start += BLOCK
    # Write out
    tl.store(out_counts_ptr + e, cnt)


# Kernel 4: Prefix sum (exclusive) of counts to produce expert_offsets: [num_experts+1]
@triton.jit
def prefix_sum_offsets_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    # Single program performs scan sequentially. offsets_ptr is length num_experts+1.
    # offsets[0] = 0; offsets[1] = counts[0]; offsets[2] = offsets[1] + counts[1]; ...
    tl.store(offsets_ptr + 0, 0)
    total = 0
    for e in range(0, num_experts):
        total += tl.load(counts_ptr + e)
        tl.store(offsets_ptr + e + 1, total)


class ModelNew(torch.nn.Module):
    def forward(self, flat: torch.Tensor):
        """
        flat: 1D int32 tensor of expert indices on CUDA device, length N
        Returns:
          sorted_token_indices: int32 (N,)
          expert_offsets: int32 (num_experts + 1,)
        """
        assert flat.is_cuda, "ModelNew.forward requires a CUDA tensor"
        assert flat.dtype == torch.int32, "flat must be int32"
        N = flat.numel()
        device = flat.device
        num_experts = 256  # as in the original code

        # Output buffers
        # 1) sorted_token_indices as out_idx
        out_idx = torch.empty(N, dtype=torch.int32, device=device)

        # 2) keys arrays for stable sort
        keys_id = torch.empty(N, dtype=torch.int32, device=device)
        keys_idx = torch.empty(N, dtype=torch.int32, device=device)

        # 3) counts per expert
        counts = torch.empty(num_experts, dtype=torch.int32, device=device)

        # 4) offsets
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)

        # Launch init keys kernel
        BLOCK = 1024
        grid_init = (triton.cdiv(N, BLOCK),)
        init_keys_kernel[grid_init](flat, out_idx, keys_id, keys_idx, N, BLOCK)

        # Launch bitonic sort kernel
        bitonic_sort_stable_kernel[grid_init](keys_id, keys_idx, out_idx, N, BLOCK)

        # Launch count per expert kernel
        grid_counts = (num_experts,)
        count_per_expert_kernel[grid_counts](flat, counts, num_experts, N, BLOCK)

        # Launch prefix sum kernel
        prefix_sum_offsets_kernel[grid_counts](counts, offsets, num_experts)

        return out_idx, offsets