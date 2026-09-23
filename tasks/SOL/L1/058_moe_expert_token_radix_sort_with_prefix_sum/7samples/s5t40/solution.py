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


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D int32 on device
        flat = topk_idx.reshape(-1).to(torch.int32)

        # Triton buffers
        num_experts = 256  # matches the original setup
        device = flat.device

        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)

        # Kernel 1: histogram via atomics
        BLOCK = 1024  # robust default; can try 2048 if desired
        count_expert_ids_atomic_kernel[(1,)](flat, counts, flat.numel(), BLOCK=BLOCK)

        # Compute exclusive prefix sum for offsets[1..] using torch (simple and stable)
        offsets[1:] = counts.cumsum(0)
        offsets[0] = 0

        # Compute sorted token indices using torch (stable) on the flattened tensor
        sorted_token_indices = flat.argsort(stable=True)

        # Return results: permutation indices and offsets
        return sorted_token_indices.to(torch.int32), offsets


def run(*args):
    return ModelNew()(*args)
