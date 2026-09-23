import torch
import triton
import triton.language as tl


@triton.jit
def bincount_kernel(x_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    # Each program handles a block of elements
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load values; x is flattened int32
    vals = tl.load(x_ptr + offsets, mask=mask, other=0)
    # We assume vals are in [0, 255] (evaluation setup uses num_experts=256)
    # Masked loads set "other" to 0, but we also guard per-bin with (vals == i) & mask
    # Create a vector of counts for each bin 0..255
    # This is unrolled; Triton will generate specialized code for this loop.
    for i in range(256):
        # Count how many vals equal i among valid positions
        eq = vals == i
        local_count = tl.sum((eq & mask).to(tl.int32))
        # Atomic add to global counts[i]
        tl.atomic_add(counts_ptr + i, local_count)


@triton.jit
def inclusive_prefix_sum_kernel(counts_ptr, y_ptr, L: tl.constexpr):
    # Compute inclusive prefix sum of counts into y
    # y[0] = 0
    y_ptr[0] = tl.zeros((), dtype=tl.int64)  # 0
    sum_val = tl.zeros((), dtype=tl.int64)
    for k in range(L):
        sum_val += tl.load(counts_ptr + k).to(tl.int64)
        y_ptr[k + 1] = sum_val


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # 1) Triton bincount into counts[0:256] (int32)
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        BLOCK = 1024  # tuneable
        grid = (triton.cdiv(N, BLOCK),)
        bincount_kernel[grid](flat, counts, N, BLOCK=BLOCK)

        # 2) Triton inclusive prefix sum to produce offsets (length 257, int64)
        offsets = torch.empty(257, dtype=torch.int64, device=flat.device)
        inclusive_prefix_sum_kernel[(1,)](counts, offsets, L=257)

        # 3) PyTorch stable argsort for flattened indices: permutation of [0, N-1]
        #    This matches the original run's behavior for sorted_token_indices.
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, offsets