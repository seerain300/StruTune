import torch
import triton
import triton.language as tl


# Triton kernel: count occurrences of each expert ID in flat.
# flat_ptr: int32 * N
# counts_ptr: int32 * num_experts
# N: runtime, BLOCK: constexpr chunk size
@triton.jit
def count_expert_ids_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N
    # Load a chunk of IDs
    ids = tl.load(flat_ptr + offsets, mask=mask, other=0)  # int32
    # Atomically add 1 for each valid element
    for i in range(BLOCK):
        if mask[i]:
            id_i = ids[i]
            tl.atomic_add(counts_ptr + id_i, 1)


# Triton kernel: compute exclusive prefix sum of counts to produce offsets[1..].
# counts_ptr: int32 * num_experts
# offsets_ptr: int32 * (num_experts + 1)
@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    total = 0
    for e in range(num_experts):
        total += tl.load(counts_ptr + e)
        tl.store(offsets_ptr + e + 1, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure device and contiguity
        device = topk_idx.device
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        num_experts = 256  # as per original code

        # Host-side stable argsort to get permutation indices (required to match original)
        # This is the primary output: the order by which tokens are sorted by expert ID.
        sorted_token_indices = torch.argsort(flat, dim=0, stable=True)

        # Triton outputs: permutation indices and offsets
        # We will compute offsets via Triton; permutation is obtained from torch.argsort.

        # Compute counts per expert using Triton
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        # Choose a reasonable block size; N is moderate (<= 8192 in provided configs)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        count_expert_ids_kernel[grid](flat, counts, N, BLOCK=BLOCK)

        # Compute offsets via exclusive prefix sum in Triton
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        exclusive_prefix_sum_kernel[(1,)](counts, offsets, num_experts)

        # Set offsets[0] = 0 (consistent with the original which sets offsets[0] = 0 implicitly)
        offsets[0] = 0

        # Return: permutation indices and offsets
        # Note: The original returns (sorted_token_indices, expert_offsets).
        # Here, we must return the permutation indices computed via torch.argsort (which is allowed)
        # and the Triton-generated offsets.
        # We cast sorted_token_indices to int32 to match original (which uses int32).
        sorted_token_indices = sorted_token_indices.to(torch.int32)
        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
