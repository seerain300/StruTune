import torch
import triton
import triton.language as tl


# Triton kernel: stable argsort permutation using counting-sort approach.
# Assumes input IDs are unique (e.g., random in [0, num_experts-1]).
# flat_ptr: *int32, length N
# out_idx_ptr: *int32, length N (output permutation indices)
# N: int (runtime)
@triton.jit
def stable_argsort_counting_kernel(flat_ptr, out_idx_ptr, N: tl.int32, BLOCK: tl.constexpr):
    for start in range(0, N, BLOCK):
        idx = start + tl.arange(0, BLOCK)            # [BLOCK] vector of indices
        mask = idx < N                               # mask for valid indices
        vals = tl.load(flat_ptr + idx, mask=mask, other=0)  # load IDs
        # Write out_idx[vals[i]] = idx[i] for valid i
        tl.store(out_idx_ptr + vals, idx, mask=mask)         # out_idx_ptr is int32


# Triton kernel: count per-expert IDs via chunked iteration with atomics.
# flat_ptr: *int32, length N
# counts_ptr: *int32, length num_experts
# N: int (runtime)
@triton.jit
def count_expert_ids_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    for start in range(0, N, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < N
        ids = tl.load(flat_ptr + offsets, mask=mask, other=0)  # int32
        # Atomically add 1 for each valid element
        for i in range(BLOCK):
            if mask[i]:
                id_i = ids[i]
                tl.atomic_add(counts_ptr + id_i, 1)


# Triton kernel: exclusive prefix sum of counts to produce offsets[1..].
# counts_ptr: *int32, length num_experts
# offsets_ptr: *int32, length num_experts+1
# num_experts: int (compile-time for loop bound)
@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    total = tl.zeros((), dtype=tl.int32)
    for e in range(num_experts):
        val = tl.load(counts_ptr + e)
        total += val
        tl.store(offsets_ptr + e + 1, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten and ensure contiguous
        flat = topk_idx.reshape(-1).contiguous()
        device = flat.device
        N = flat.numel()
        num_experts = 256  # as in the original code

        # Allocate outputs
        out_idx = torch.empty(N, dtype=torch.int32, device=device)  # permutation indices (sorted_token_indices)
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)

        # Triton kernel launches
        BLOCK = 1024  # chunk size for vectorized loads/stores

        # 1) Stable argsort permutation via Triton (counting-sort approach)
        stable_argsort_counting_kernel[(1,)](flat, out_idx, N, BLOCK=BLOCK)

        # 2) Count per-expert IDs
        count_expert_ids_kernel[(1,)](flat, counts, N, BLOCK=BLOCK)

        # 3) Exclusive prefix sum of counts to produce offsets[1..]
        exclusive_prefix_sum_kernel[(1,)](counts, offsets, num_experts)

        # Set offsets[0] = 0 (the original implicitly sets offsets[0] = 0)
        offsets[0] = 0

        # Return: permutation indices and offsets
        return out_idx, offsets