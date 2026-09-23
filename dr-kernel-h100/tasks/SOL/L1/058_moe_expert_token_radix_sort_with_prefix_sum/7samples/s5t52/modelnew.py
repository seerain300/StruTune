import torch
import triton
import triton.language as tl


@triton.jit
def count_expert_ids_atomic_kernel(flat_ptr, counts_ptr, N):
    # Single program instance processes the input sequentially, issuing atomic_add per element.
    # This avoids complex vectorized patterns that can cause Triton JIT/runtime issues.
    i = 0
    while i < N:
        val = tl.load(flat_ptr + i)
        # Atomic add 1 to counts[val]; val is assumed to be in [0, 255] per original code.
        tl.atomic_add(counts_ptr + val, 1)
        i += 1


@triton.jit
def exclusive_prefix_sum_offsets_kernel(counts_ptr, offsets_ptr, NUM_EXPERTS: tl.constexpr):
    # Compute exclusive prefix sum: offsets[i+1] = sum_{j < i} counts[j], for i in [0..NUM_EXPERTS-1]
    # offsets[0] = 0 (set on host).
    sum_val = tl.zeros((), dtype=tl.int32)  # scalar accumulator
    for i in range(NUM_EXPERTS):
        sum_val += counts_ptr[i]
        tl.store(offsets_ptr + (i + 1), sum_val)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D int32 on device
        flat = topk_idx.reshape(-1).to(torch.int32)
        N = flat.numel()
        device = flat.device

        # Triton buffer for counts (per expert)
        num_experts = 256  # matches the original setup
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)

        # Kernel 1: histogram via per-element atomic add
        count_expert_ids_atomic_kernel[(1,)](flat, counts, N)

        # Kernel 2: exclusive prefix sum for offsets[1..]
        exclusive_prefix_sum_offsets_kernel[(1,)](counts, offsets, NUM_EXPERTS=num_experts)
        # Set offsets[0] = 0
        offsets[0] = 0

        # Compute sorted token indices using torch (stable) on the flattened tensor
        sorted_token_indices = flat.argsort(stable=True)

        # Return results: permutation indices and offsets
        return sorted_token_indices.to(torch.int32), offsets