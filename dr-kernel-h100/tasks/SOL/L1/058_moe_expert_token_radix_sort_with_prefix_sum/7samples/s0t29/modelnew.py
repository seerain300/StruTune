import torch
import triton
import triton.language as tl


# Triton kernel: bincount of flattened expert IDs into counts[0:256] (int32)
@triton.jit
def bincount_kernel_flat256(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Load values; default other=0 is fine since we mask with 'mask'
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32

    # For each possible expert id j in [0, 255], atomically add to counts[j] if mask is true
    # This avoids constructing a 2D index grid in Triton.
    for j in range(256):
        eq_mask = (vals == j) & mask
        # Atomic add 1 for each True in eq_mask
        # counts_ptr is int32; tl.atomic_add supports int32
        tl.atomic_add(counts_ptr + j, tl.where(eq_mask, 1, 0).to(tl.int32))


# Triton kernel: inclusive prefix sum of 'x' of length L, write to 'y'
# Assumes L is small (257). We implement a simple loop in a single program.
@triton.jit
def inclusive_prefix_sum_kernel(x_ptr, y_ptr, L: tl.int32):
    total = 0
    for i in range(L):
        val = tl.load(x_ptr + i)  # int32
        total += val
        tl.store(y_ptr + i, total.to(tl.int64))  # write as int64


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # 1) Triton bincount into counts[0:256] (int32)
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        bincount_kernel_flat256[grid](flat, counts, N, BLOCK=BLOCK)

        # 2) Triton inclusive prefix sum to produce offsets (length 257, int64)
        offsets = torch.empty(257, dtype=torch.int64, device=flat.device)
        # Initialize y[0] = 0
        offsets[0] = 0
        inclusive_prefix_sum_kernel[(1,)](counts, offsets, L=257)

        # 3) Use PyTorch for stable argsort of flattened indices: permutation of [0, N-1]
        #    This matches the original run's behavior for sorted_token_indices.
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, offsets