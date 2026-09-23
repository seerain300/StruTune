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
        # We will compute offsets via torch.cumsum to avoid Triton dynamic loops

        # Kernel 1: histogram via atomics
        BLOCK = 2048  # tuned for throughput; also try 1024 if needed
        count_expert_ids_atomic_kernel[(1,)](flat, counts, flat.numel(), BLOCK=BLOCK)

        # Compute expert offsets using torch.cumsum (robust and fast)
        # offsets[0] = 0, offsets[1:] = inclusive cumsum of counts
        inclusive = torch.cumsum(counts, dim=0)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0
        offsets[1:] = inclusive

        # Compute sorted token indices using torch (stable) on the flattened tensor
        sorted_token_indices = flat.argsort(stable=True)

        # Return results: permutation indices and offsets
        return sorted_token_indices.to(torch.int32), offsets