import torch
import triton
import triton.language as tl


@triton.jit
def count_expert_ids_atomic_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    # Single program instance processes the input in chunks of BLOCK
    # Each iteration loads a block of elements and atomically adds 1 to counts[id]
    offset = 0
    while offset < N:
        idx = offset + tl.arange(0, BLOCK)
        mask = idx < N
        # Load int32 IDs
        ids = tl.load(flat_ptr + idx, mask=mask, other=0)
        # Atomic add 1 for each valid id
        # Note: Triton supports atomic_add for int32
        tl.atomic_add(counts_ptr + ids, 1, mask=mask)
        offset += BLOCK


@triton.jit
def exclusive_prefix_sum_offsets_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    # Compute exclusive prefix sum: offsets[i] = sum_{j < i} counts[j]
    # We write offsets[1..] directly; offsets[0] is set on host to 0.
    total = 0
    for i in range(0, num_experts):
        # total is a scalar; load current count
        count_i = tl.load(counts_ptr + i)
        total += count_i
        # offsets[i+1] = total of previous elements
        tl.store(offsets_ptr + i + 1, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version of the original Model:
        - Computes sorted_token_indices (argsort of flattened expert IDs) using torch.
        - Computes expert_offsets using Triton (histogram via atomics + exclusive prefix sum).
        Returns:
            sorted_token_indices: permutation of original positions (int32)
            expert_offsets: length num_experts+1 (int32)
        """
        # Ensure int32 and contiguous flattened input
        flat = topk_idx.reshape(-1).to(torch.int32).contiguous()
        N = flat.numel()
        device = flat.device

        # Triton buffers
        num_experts = 256  # matches the original setup
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)

        # Kernel 1: histogram via atomics
        BLOCK = 1024  # good default for vectorized processing
        count_expert_ids_atomic_kernel[(1,)](flat, counts, N, BLOCK=BLOCK)

        # Kernel 2: exclusive prefix sum for offsets[1..]
        exclusive_prefix_sum_offsets_kernel[(1,)](counts, offsets, num_experts)
        # Set offsets[0] = 0
        offsets[0] = 0

        # Compute sorted token indices using torch (stable) on the flattened tensor
        # This is robust and avoids Triton sorting pitfalls.
        sorted_token_indices = flat.argsort(stable=True)

        # Return results: permutation indices and offsets
        return sorted_token_indices.to(torch.int32), offsets


def run(*args):
    return ModelNew()(*args)
