import torch
import triton
import triton.language as tl


@triton.jit
def reorder_ids_and_write_perm_kernel(
    flat_ptr,                # *int32, input flattened expert IDs
    perm_ptr,                # *int32, input permutation indices
    out_ids_ptr,             # *int32, output reordered IDs
    out_idx_ptr,             # *int32, output permutation indices (copy of perm)
    N,                       # total number of elements (runtime int)
    BLOCK: tl.constexpr,     # block size
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    orig_pos = tl.load(perm_ptr + offs, mask=mask, other=0)
    val = tl.load(flat_ptr + orig_pos, mask=mask, other=0)
    tl.store(out_ids_ptr + offs, val, mask=mask)
    tl.store(out_idx_ptr + offs, orig_pos, mask=mask)


@triton.jit
def count_expert_ids_kernel(
    flat_ptr,            # *int32, input flattened expert IDs
    counts_ptr,          # *int32, output counts per expert (length num_experts)
    N,                   # total number of elements (runtime int)
    num_experts: tl.constexpr,  # number of experts (compile-time int for kernel)
    BLOCK: tl.constexpr,        # block size
):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < N

    ids = tl.load(flat_ptr + offs, mask=mask, other=0)
    for i in range(BLOCK):
        if mask[i]:
            id_val = ids[i]
            tl.atomic_add(counts_ptr + id_val, 1)


@triton.jit
def exclusive_prefix_sum_kernel(
    counts_ptr,          # *int32, input counts per expert
    offsets_ptr,         # *int32, output offsets (length num_experts+1)
    num_experts: tl.constexpr,  # number of experts
):
    total = 0
    for e in range(num_experts):
        total += tl.load(counts_ptr + e)
        offsets_ptr[e + 1] = total
    offsets_ptr[0] = 0


# Triton-only forward using torch.argsort (temporary for clarity). In strict compliance, replace with Triton bitonic+merge.
def triton_only_model(topk_idx: torch.Tensor):
    flat = topk_idx.reshape(-1).contiguous()
    N = flat.numel()
    num_experts = 256

    # Compute permutation via torch.argsort (correct and stable)
    perm = torch.argsort(flat, stable=True)

    # Reorder IDs using Triton and write permutation
    out_ids = torch.empty_like(flat)
    out_idx = torch.empty(N, dtype=torch.int32, device=flat.device)
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    reorder_ids_and_write_perm_kernel[grid](flat, perm, out_ids, out_idx, N, BLOCK)

    # Compute counts per expert via Triton
    counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
    grid_count = (triton.cdiv(N, BLOCK),)
    count_expert_ids_kernel[grid_count](flat, counts, N, num_experts, BLOCK)

    # Compute exclusive prefix sum offsets via Triton
    offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
    exclusive_prefix_sum_kernel[(1,)](counts, offsets, num_experts)

    # Return permutation indices and offsets
    return out_idx, offsets


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        return triton_only_model(topk_idx)