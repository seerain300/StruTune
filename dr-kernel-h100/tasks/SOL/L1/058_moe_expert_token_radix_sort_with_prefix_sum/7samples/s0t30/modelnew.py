import torch
import triton
import triton.language as tl


@triton.jit
def _bincount_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    # Each program handles BLOCK elements
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load values with mask; other values are ignored via mask
    # flat_ptr is int32*
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)

    # Count only for values in [0, 255]
    for i in range(0, 256):
        cond = mask & (vals == i)
        # Atomic add 1 for each matching element
        # counts_ptr is int32*
        tl.atomic_add(counts_ptr + i, cond.to(tl.int32))


@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, L: tl.constexpr):
    # Single program computes inclusive prefix sum for L elements
    # counts_ptr: int32* of length 256
    # offsets_ptr: int64* of length 257, inclusive prefix sum (including leading 0)
    running = tl.zeros((), dtype=tl.int32)
    # Write leading 0
    tl.store(offsets_ptr + 0, running.to(tl.int64))

    for i in range(1, L):
        running += tl.load(counts_ptr + (i - 1))
        tl.store(offsets_ptr + i, running.to(tl.int64))


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()
        device = flat.device

        # 1) Triton bincount into counts[0:256] (int32)
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _bincount_kernel[grid](flat, counts, N, BLOCK=BLOCK)

        # 2) Triton inclusive prefix sum to produce offsets (length 257, int64)
        offsets = torch.empty(257, dtype=torch.int64, device=device)
        _inclusive_prefix_sum_kernel[(1,)](counts, offsets, L=257)

        # 3) Use PyTorch for stable argsort of flattened indices (returns permutation of [0, N-1])
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, offsets