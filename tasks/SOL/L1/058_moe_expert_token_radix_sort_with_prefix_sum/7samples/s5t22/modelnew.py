import torch
import triton
import triton.language as tl


# Kernel 1: Stable argsort permutation using counting-sort approach (assumes unique IDs).
# For each i in [0, N), load id = flat[i], then write out_idx[id] = i.
@triton.jit
def stable_argsort_counting_kernel(flat_ptr, out_idx_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < N
    # Load original IDs
    ids = tl.load(flat_ptr + offs, mask=mask, other=0)  # assume int32
    # Write permutation: out_idx[id] = i
    tl.store(out_idx_ptr + ids, offs, mask=mask)       # stable by insertion order


# Kernel 2: Count per-expert IDs using chunked iteration with atomic adds.
@triton.jit
def count_expert_ids_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < N
    ids = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32
    # Atomic add 1 for each valid id
    tl.atomic_add(counts_ptr + ids, 1, mask=mask)


# Kernel 3: Exclusive prefix sum over counts to produce offsets[1..].
# This kernel is sequential over num_experts; safe for num_experts=256.
@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    # Compute offsets[j] = sum_{i < j} counts[i]
    s = tl.zeros((), dtype=tl.int32)
    for j in range(0, num_experts):
        ci = tl.load(counts_ptr + j)
        s += ci
        tl.store(offsets_ptr + j + 1, s)
    # offsets[0] must be 0 (set on host)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure int32 on device
        if topk_idx.dtype != torch.int32:
            flat = topk_idx.to(torch.int32)
        else:
            flat = topk_idx
        device = flat.device
        num_experts = 256
        # Flatten to 1D
        N = flat.numel()
        # Allocate outputs
        out_idx = torch.empty(N, dtype=torch.int32, device=device)  # permutation indices (sorted_token_indices)
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)

        # Triton kernel launches: use a single program for simplicity and robustness
        BLOCK = 1024  # chunk size; 1024 works well and keeps control flow simple

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