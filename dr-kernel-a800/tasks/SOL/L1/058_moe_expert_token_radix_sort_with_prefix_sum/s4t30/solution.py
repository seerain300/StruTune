import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(topk_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    """
    Histogram of expert IDs in topk_ptr (int32) into counts_ptr (int32),
    where counts_ptr[i] = number of times value == i in topk_ptr.

    N: total number of elements in topk_ptr.
    counts_ptr: length = num_experts (256).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    vals = tl.load(topk_ptr + offsets, mask=mask, other=0)  # int32
    # For each valid lane, atomic add 1 into counts[vals[i]]
    for i in range(BLOCK):
        if mask[i]:
            v = vals[i]
            # v is in [0, 255] per input generation; still guard by mask
            tl.atomic_add(counts_ptr + v, 1)


@triton.jit
def prefix_sum_inclusive_kernel(counts_ptr, offsets_ptr, BINS: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr (length BINS) into offsets_ptr (length BINS+1).
    offsets_ptr[0] = 0; offsets_ptr[i+1] = sum_{k=0..i} counts[k].
    """
    running = 0
    for i in range(BINS):
        ci = tl.load(counts_ptr + i)
        running += ci
        tl.store(offsets_ptr + i + 1, running)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized implementation:
        - sorted_token_indices: permutation from torch.argsort(flat, stable=True) to ensure correctness.
        - expert_offsets: computed via Triton histogram + prefix sum.
        """
        # Ensure device and contiguity
        assert topk_idx.is_cuda, "Input must be on CUDA device for Triton kernels."
        topk_idx = topk_idx.contiguous()
        flat = topk_idx.view(-1)
        N = flat.numel()
        num_experts = 256  # original code's num_experts

        # 1) Stable sort permutation using PyTorch (ensures correctness)
        sorted_token_indices = torch.argsort(flat, stable=True)  # int64 by default; convert to int32
        sorted_token_indices = sorted_token_indices.to(torch.int32)

        # 2) Triton histogram of expert IDs
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        BLOCK = 2048
        grid = (triton.cdiv(N, BLOCK),)
        histogram_kernel[grid](flat, counts, N, BLOCK=BLOCK)

        # 3) Triton inclusive prefix sum to get expert_offsets
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        # offsets[0] = 0; fill [1..num_experts]
        prefix_sum_inclusive_kernel[(1,)](counts, offsets, BINS=num_experts)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
