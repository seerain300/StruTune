import torch
import triton
import triton.language as tl


@triton.jit
def count_expert_ids_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    # Each element in the flat array performs an atomic add to counts[id]
    offset = 0
    while offset < N:
        idx = offset + tl.arange(0, BLOCK)
        mask = idx < N
        vals = tl.load(flat_ptr + idx, mask=mask, other=0)
        # Atomic add 1 for each valid element
        tl.atomic_add(counts_ptr + vals, 1, mask=mask)
        offset += BLOCK


@triton.jit
def inclusive_cumsum_offsets_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    # Compute inclusive prefix sum: offsets[i] = sum(counts[:i])
    running = tl.zeros((), dtype=tl.int32)
    for i in range(num_experts):
        running += counts_ptr[i]
        tl.store(offsets_ptr + i, running)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D int32 on device
        flat = topk_idx.reshape(-1).to(torch.int32)

        # Triton buffers
        num_experts = 256  # matches the original setup
        device = flat.device

        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)

        # Kernel 1: histogram via per-element atomics (robust and previously correct)
        BLOCK = 1024  # safe default; avoids large vector issues
        count_expert_ids_kernel[(1,)](flat, counts, flat.numel(), BLOCK=BLOCK)

        # Kernel 2: inclusive prefix sum -> offsets[1..], offsets[0] = 0
        inclusive_cumsum_offsets_kernel[(1,)](counts, offsets, num_experts)
        offsets[0] = 0

        # Compute sorted token indices using torch (stable) on the flattened tensor
        sorted_token_indices = flat.argsort(stable=True)

        # Return results: permutation indices and offsets
        return sorted_token_indices.to(torch.int32), offsets