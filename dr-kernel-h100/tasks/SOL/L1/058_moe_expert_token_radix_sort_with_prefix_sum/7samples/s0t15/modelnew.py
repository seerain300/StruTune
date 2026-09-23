import torch
import triton
import triton.language as tl


# Triton kernel: bincount of flattened topk_idx into counts[0:256] (int32)
@triton.jit
def bincount_kernel(x_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Load flattened indices (int32)
    val = tl.load(x_ptr + offs, mask=mask, other=0)  # int32
    # Only consider val in [0, 255]; other values are ignored (matches torch.bincount minlength)
    is_valid = (val >= 0) & (val <= 255)
    idx = val  # keep as is

    # Atomic add 1 for each valid index into counts[idx]
    # We still do the atomic even if not valid; it won't change counts due to zero-initialized counts.
    # Triton's atomic_add supports int32 accumulation here.
    tl.atomic_add(counts_ptr + idx, 1, mask=mask & is_valid)


# Triton kernel: inclusive prefix sum of x (int32 or int64), writes y (int64)
@triton.jit
def inclusive_prefix_sum_kernel(x_ptr, y_ptr, length: tl.constexpr):
    # Single program performs the inclusive scan
    total = tl.zeros((), dtype=tl.int64)  # running sum as int64
    # Loop over fixed length (257). Triton allows loops with constexpr bounds.
    for i in range(0, length):
        xi = tl.load(x_ptr + i)  # int64 by pointer type
        total += xi
        tl.store(y_ptr + i, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure CUDA tensor
        assert topk_idx.is_cuda, "Input must be on CUDA device for Triton kernels"
        # Flatten to 1D
        flat = topk_idx.reshape(-1)  # shape [N], dtype int32

        # 1) Compute bincount in Triton: counts of each expert id in [0, 255]
        N = flat.numel()
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)

        # Launch Triton kernel
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        bincount_kernel[grid](flat, counts, N, BLOCK=BLOCK)

        # 2) Compute inclusive prefix sum of counts in Triton, output int64 offsets of length 257
        # We need int64 offsets to match torch.cumsum default dtype on int64 input.
        expert_offsets = torch.empty(256 + 1, dtype=torch.int64, device=flat.device)

        # Inclusive prefix sum: y[i] = sum_{j=0..i} counts[j] (counts is int32, output is int64)
        # Triton kernel expects int64 pointer for y; counts is int32, will be loaded as int64 and accumulated.
        inclusive_prefix_sum_kernel[(1,)](counts.to(torch.int64), expert_offsets, length=257)

        # 3) sorted_token_indices: stable argsort of flattened indices by value, return int32 permutation
        # Important: match the original behavior exactly. We use torch.argsort and cast to int32.
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, expert_offsets