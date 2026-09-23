import torch
import triton
import triton.language as tl


@triton.jit
def count_expert_ids_atomic_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    # Single program instance processes the input in chunks of BLOCK.
    # For each element i in [0, N), loads flat[i], computes id, and atomically increments counts[id].
    offset = 0
    while offset < N:
        idx = offset + tl.arange(0, BLOCK)
        mask = idx < N
        # Load flat values; masked elements load 0 to avoid undefined behavior
        vals = tl.load(flat_ptr + idx, mask=mask, other=0)
        # Atomically add 1 to counts[vals] for valid elements
        tl.atomic_add(counts_ptr + vals, 1, mask=mask)
        offset += BLOCK


@triton.jit
def cumsum_inclusive_kernel(counts_ptr, inclusive_ptr, num_items: tl.constexpr):
    # Compute inclusive prefix sum of counts[0:num_items] into inclusive_ptr[0:num_items]
    sum_val = 0
    for i in range(num_items):
        sum_val += tl.load(counts_ptr + i)
        tl.store(inclusive_ptr + i, sum_val)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D int32 on device
        flat = topk_idx.reshape(-1).to(torch.int32)
        N = flat.numel()
        device = flat.device

        num_experts = 256  # matches the original setup

        # Triton buffers
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        inclusive = torch.empty(num_experts, dtype=torch.int32, device=device)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)

        # Kernel 1: histogram via atomics
        BLOCK = 2048  # tuned for throughput; also try 1024 if needed
        count_expert_ids_atomic_kernel[(1,)](flat, counts, N, BLOCK=BLOCK)

        # Kernel 2: inclusive prefix sum of counts using Triton
        cumsum_inclusive_kernel[(1,)](counts, inclusive, num_experts)

        # Compute offsets: offsets[0] = 0; offsets[1:] = inclusive
        offsets[0] = 0
        offsets[1:] = inclusive

        # Compute sorted token indices using torch (stable) on the flattened tensor
        sorted_token_indices = flat.argsort(stable=True)

        # Return results: permutation indices and offsets
        return sorted_token_indices.to(torch.int32), offsets


def run(*args):
    return ModelNew()(*args)
