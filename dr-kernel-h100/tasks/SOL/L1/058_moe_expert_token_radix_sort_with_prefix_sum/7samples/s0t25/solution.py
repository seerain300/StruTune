import torch
import triton
import triton.language as tl


@triton.jit
def bincount_kernel(x_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    # Each program handles BLOCK elements
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load flattened indices
    # x_ptr points to int32 data; we can load as int32 directly.
    # For masked-out lanes, 'other' doesn't matter because we guard with mask.
    idx = tl.load(x_ptr + offsets, mask=mask, other=0)

    # Atomically accumulate counts into counts_ptr[0..255]
    # idx should be in [0, 255]; for any out-of-range, mask ensures we don't touch memory (we won't add).
    # We perform up to 256 atomic adds per lane; with mask, only valid lanes will add.
    # Note: Triton atomic_add is supported for int32.
    for i in range(0, 256):
        # Condition: only count if idx == i
        cond = mask & (idx == i)
        # Add 1 for each valid lane with idx == i
        # Use a safe cast to int32 for the value to add.
        tl.atomic_add(counts_ptr + i, cond.to(tl.int32))


@triton.jit
def inclusive_prefix_sum_kernel(x_ptr, y_ptr, length: tl.int32):
    # Single-program inclusive scan over x_ptr[0..length-1], writing to y_ptr[0..length-1].
    # We assume x_ptr and y_ptr are int32 for speed; host can convert to int64 before/after if needed.
    # This kernel is O(length) and handles small length (257) efficiently.
    running = 0
    for k in range(0, length):
        v = tl.load(x_ptr + k)
        running += v
        tl.store(y_ptr + k, running)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure we operate on the GPU tensor
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"

        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # 1) Compute sorted_token_indices = stable argsort of flattened positions
        #    This returns permutation of [0, N-1] sorted by values at those positions.
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        # 2) Triton bincount into counts (int32) for 256 bins
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)

        # Grid size: one program per BLOCK elements
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)

        bincount_kernel[grid](flat, counts, N, BLOCK=BLOCK)

        # 3) Inclusive prefix sum of counts to produce expert_offsets (length 257)
        #    We compute prefix sum in Triton (int32), then cast to int64 on host to match torch.cumsum default behavior.
        prefix_sum = torch.empty(256, dtype=torch.int32, device=flat.device)
        inclusive_prefix_sum_kernel[(1,)](counts, prefix_sum, 256)  # length is a constexpr here

        # Add the final element (sum of all counts)
        total = torch.sum(counts)  # int32 scalar on device
        expert_offsets = torch.empty(257, dtype=torch.int64, device=flat.device)
        # First, write 0..256 inclusive prefix sums
        # y[0] = counts[0], y[1] = counts[0] + counts[1], ..., y[256] = sum(counts)
        # Then, add 0 at the beginning: expert_offsets[0] = 0
        # We can set y[0..256] to prefix_sum[0..256] and write expert_offsets[1..257] as prefix_sum.
        # However, torch.cumsum returns int64 by default, and our prefix_sum is int32. Cast before writing.
        expert_offsets[0] = 0
        expert_offsets[1:] = prefix_sum.to(torch.int64)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
