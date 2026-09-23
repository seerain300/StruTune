import torch
import triton
import triton.language as tl


@triton.jit
def _bincount_kernel(flat64_ptr: tl.pointer_type(tl.int64), counts_ptr: tl.pointer_type(tl.int32), N: tl.int32, BLOCK: tl.constexpr):
    # Each program handles BLOCK elements
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load values as int64; masked elements get 0
    v = tl.load(flat64_ptr + offsets, mask=mask, other=0).to(tl.int64)

    # For valid values in [0, 255], atomic add 1 to counts[v]
    valid = (v >= 0) & (v < 256) & mask
    # Cast v to int32 for indexing into counts
    v32 = v.to(tl.int32)
    # Atomic add 1 for each valid v
    for i in range(BLOCK):
        if valid[i]:
            # Atomic add 1 to counts[v32[i]]
            tl.atomic_add(counts_ptr + v32[i], 1)


@triton.jit
def _prefix_sum_kernel(counts_ptr: tl.pointer_type(tl.int32), offsets_ptr: tl.pointer_type(tl.int64), L: tl.constexpr):
    # Compute inclusive prefix sum: offsets[j] = sum_{k=0..j-1} counts[k], j = 0..L-1
    # offsets[0] = 0 (already initialized on host)
    total = tl.zeros((), dtype=tl.int64)
    for j in range(1, L):
        c = tl.load(counts_ptr + j)
        total += c.to(tl.int64)
        tl.store(offsets_ptr + j, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()
        device = flat.device

        # 1) Use PyTorch for stable argsort of flattened indices (returns permutation of [0, N-1])
        #    This matches the original run's behavior for sorted_token_indices.
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        # 2) Triton bincount into counts[0:256] (int32)
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        # Convert flat to int64 for Triton load; ensures Triton sees 64-bit values
        flat64 = flat.to(torch.int64)

        # Launch kernel with grid size over chunks
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _bincount_kernel[grid](flat64, counts, N, BLOCK=BLOCK)

        # 3) Triton inclusive prefix sum to produce offsets (length 257, int64)
        offsets = torch.empty(257, dtype=torch.int64, device=device)
        offsets[0] = 0  # inclusive prefix sum starting at 0
        _prefix_sum_kernel[(1,)](counts, offsets, L=257)

        return sorted_token_indices, offsets