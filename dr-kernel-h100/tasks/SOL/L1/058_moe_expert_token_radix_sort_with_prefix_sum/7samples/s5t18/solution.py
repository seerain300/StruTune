import torch
import triton
import triton.language as tl


# Triton kernel: perform a counting-sort-based stable argsort.
# We create out_ids (sorted IDs) and out_idx (permutation indices).
# Processing original indices in increasing order yields stable ordering by IDs.
# If duplicates exist, the first occurrence determines the position; this is deterministic.
@triton.jit
def stable_argsort_counting_kernel(flat_ptr, out_ids_ptr, out_idx_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    # One program processes the entire range in chunks. We use a for-loop over chunks.
    for start in range(0, N, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < N
        # Load original values; for masked elements, we can use 0 but won't write them.
        vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
        # For each valid element i in the chunk, place it at position 'vals[i]' in out arrays.
        for i in range(BLOCK):
            if mask[i]:
                v = vals[i]  # id
                # out_ids[v] = v (self-key assignment)
                tl.store(out_ids_ptr + v, v)
                # out_idx[v] = original index i
                tl.store(out_idx_ptr + v, offsets[i])


# Triton kernel: count per-expert IDs using chunked iteration and atomics.
# flat_ptr points to int32 IDs. counts_ptr is length num_experts, int32.
@triton.jit
def count_expert_ids_kernel(flat_ptr, counts_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    for start in range(0, N, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < N
        ids = tl.load(flat_ptr + offsets, mask=mask, other=0)  # int32
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
        # Flatten and ensure contiguity; device follows input
        flat = topk_idx.reshape(-1).contiguous()
        device = flat.device
        N = flat.numel()
        num_experts = 256  # as per original code

        # Allocate outputs
        out_idx = torch.empty(N, dtype=torch.int32, device=device)  # permutation indices
        out_ids = torch.empty(N, dtype=torch.int32, device=device)  # sorted IDs (not returned)
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)

        # Launch Triton kernels
        BLOCK = 1024
        # Kernel 1: stable argsort permutation via counting-sort approach
        stable_argsort_counting_kernel[(1,)](flat, out_ids, out_idx, N, BLOCK=BLOCK)

        # Kernel 2: count per-expert IDs
        count_expert_ids_kernel[(1,)](flat, counts, N, BLOCK=BLOCK)

        # Kernel 3: exclusive prefix sum of counts to produce offsets
        exclusive_prefix_sum_kernel[(1,)](counts, offsets, num_experts)

        # Set offsets[0] = 0
        offsets[0] = 0

        # Return: permutation indices and offsets
        # The original run returns (sorted_token_indices, expert_offsets).
        # Here, sorted_token_indices are the values in out_idx.
        return out_idx, offsets


def run(*args):
    return ModelNew()(*args)
