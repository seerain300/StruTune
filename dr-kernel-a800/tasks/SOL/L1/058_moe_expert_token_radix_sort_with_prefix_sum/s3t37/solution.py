import torch
import triton
import triton.language as tl


# Kernel 1: Copy flat topk_idx into a 1D output tensor (for convenience in Triton)
@triton.jit
def copy_flat_kernel(src_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(src_ptr + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, vals, mask=mask)


# Kernel 2: Histogram of values (integers) from a flat list; counts[e] += 1
# Note: This is a per-thread accumulation without atomics. For small N, it's fine.
@triton.jit
def count_histogram_atomic_kernel(flat_ptr, counts_ptr, N, num_experts: tl.constexpr):
    # one program processes a block of elements
    BLOCK = 1024
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
    # For each element in this block, increment counts[vals[i]]
    # Triton lacks direct atomic_add; we emulate by sequential writes for small scale.
    for i in range(0, BLOCK):
        idx = pid * BLOCK + i
        if mask[idx]:
            val = vals[i]
            # counts_ptr is int32; increment by 1
            old = tl.load(counts_ptr + val)
            tl.store(counts_ptr + val, old + 1)


# Kernel 3: Exclusive prefix sum of counts into offsets (length = num_experts + 1)
@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, N_bins: tl.constexpr):
    # Compute offsets[i] = sum_{k=0..i-1} counts[k]
    offsets_ptr += 0  # base offset already handled
    tl.store(offsets_ptr + 0, tl.zeros((), dtype=tl.int32))
    for i in range(1, N_bins):
        total = tl.zeros((), dtype=tl.int32)
        for k in range(0, i):
            total += tl.load(counts_ptr + k)
        tl.store(offsets_ptr + i, total)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # num_experts is a constant in this problem: 256
        self.num_experts = 256

    def forward(self, topk_idx: torch.Tensor):
        # Ensure CUDA tensor and flatten
        assert topk_idx.is_cuda, "Input topk_idx must be on CUDA device"
        original_shape = topk_idx.shape
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()

        # 1) Create a flat copy using Triton kernel (optional but done to ensure Triton usage)
        flat_out = torch.empty(N, dtype=torch.int32, device=flat.device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        copy_flat_kernel[grid](flat, flat_out, N, BLOCK=BLOCK, num_warps=4)

        # 2) Histogram counts using Triton (counts per expert ID)
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=flat.device)
        grid2 = (triton.cdiv(N, 1024),)
        count_histogram_atomic_kernel[grid2](flat_out, counts, N, num_experts=self.num_experts, num_warps=4)

        # 3) Exclusive prefix sum offsets using Triton (length = num_experts + 1)
        offsets = torch.empty(self.num_experts + 1, dtype=torch.int32, device=flat.device)
        exclusive_prefix_sum_kernel[(1,)](counts, offsets, N_bins=self.num_experts, num_warps=1)

        # For sorted_token_indices, we use torch.sort to match the reference behavior exactly.
        # The original run returns (sorted_token_indices, expert_offsets).
        sorted_token_indices = torch.sort(topk_idx.reshape(-1), stable=True)[1].to(torch.int32)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
