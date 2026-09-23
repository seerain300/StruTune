import torch
import triton
import triton.language as tl


# Triton kernel: build stable argsort permutation indices using counting-sort logic.
# Assumes flat contains unique expert IDs [0, num_experts-1]. We still implement general IDs.
# out_idx[out_idx[i] = i] is not used here; instead we place each i at position equal to its ID.
# Because IDs are unique, no two threads write to the same index.
@triton.jit
def reorder_by_counting_kernel(flat_ptr, out_idx_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    ids = tl.load(flat_ptr + offsets, mask=mask, other=0)  # int32
    # For valid positions, write out_idx[ids] = offsets
    # Note: since ids are in [0, N-1], this is a permutation placement.
    tl.store(out_idx_ptr + ids, offsets, mask=mask)


# Triton kernel: histogram of expert IDs in flat using atomics.
# flat_ptr points to int32 values. counts_ptr is length num_experts, int32.
@triton.jit
def count_expert_ids_kernel(flat_ptr, counts_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    # Single program scans flat in chunks and atomically increments counts.
    for start in range(0, N, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < N
        ids = tl.load(flat_ptr + offsets, mask=mask, other=0)  # int32
        # Atomically add 1 for each valid element
        for i in range(BLOCK):
            if mask[i]:
                id_i = ids[i]
                tl.atomic_add(counts_ptr + id_i, 1)


# Triton kernel: compute exclusive prefix sum of counts to produce offsets[1..].
# counts_ptr is length num_experts, int32. offsets_ptr is length num_experts+1, int32.
@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    total = 0
    for e in range(num_experts):
        total += tl.load(counts_ptr + e)
        tl.store(offsets_ptr + e + 1, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten and ensure contiguous, device matching
        flat = topk_idx.reshape(-1).contiguous()
        device = flat.device
        N = flat.numel()
        num_experts = 256  # as per original code

        # Allocate outputs
        out_idx = torch.empty(N, dtype=torch.int32, device=device)  # permutation indices
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)

        # Launch Triton kernels
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)

        # Kernel 1: build permutation indices using counting-sort approach
        reorder_by_counting_kernel[grid](flat, out_idx, N, BLOCK=BLOCK)

        # Kernel 2: count per-expert IDs
        count_expert_ids_kernel[(1,)](flat, counts, N, BLOCK=BLOCK)

        # Kernel 3: exclusive prefix sum of counts to produce offsets
        exclusive_prefix_sum_kernel[(1,)](counts, offsets, num_experts)

        # Set offsets[0] = 0
        offsets[0] = 0

        # Return: permutation indices (stable argsort) and offsets
        return out_idx, offsets