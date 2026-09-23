import torch
import triton
import triton.language as tl


@triton.jit
def histogram_atomic_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    """
    Histogram of values in flat_ptr (int32), one pass with atomic adds.
    flat_ptr: *int32, shape [N]
    counts_ptr: *int32, shape [num_experts]
    N: int, total number of elements
    BLOCK: int, number of elements processed per program
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    ids = tl.load(flat_ptr + offsets, mask=mask, other=0)
    for i in range(BLOCK):
        id_val = ids[i]
        valid = mask[i]
        if valid:
            tl.atomic_add(counts_ptr + id_val, 1)


@triton.jit
def prefix_scan_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr, BLOCK: tl.constexpr):
    """
    Compute prefix sum of counts_ptr (length num_experts) into offsets_ptr (length num_experts),
    then set offsets_ptr[0] = 0 on host. This kernel assumes we process the entire array; BLOCK
    must be >= num_experts. Given num_experts=256, we set BLOCK=256.

    counts_ptr: *int32, length num_experts
    offsets_ptr: *int32, length num_experts (we'll allocate length num_experts + 1 on host,
                  and write into positions 1..num_experts in this kernel, host sets 0).
    num_experts: compile-time constant
    BLOCK: compile-time constant >= num_experts
    """
    running = 0
    for i in range(0, num_experts):
        ci = tl.load(counts_ptr + i)
        running += ci
        tl.store(offsets_ptr + i + 1, running)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure CUDA tensor for Triton
        assert topk_idx.is_cuda, "topk_idx must be a CUDA tensor for Triton kernels."
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32."

        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # Compute sorted_token_indices using PyTorch (fast and stable). The original code
        # uses .sort(stable=True), which returns values and indices. Here we need indices,
        # so we use argsort with stable=True. This matches the permutation produced by sort.
        sorted_token_indices = torch.argsort(flat, stable=True).to(torch.int32)

        # num_experts is fixed at 256 per the original code
        num_experts = 256

        # Allocate counts for histogram
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)

        # Launch histogram kernel
        # Choose BLOCK size for histogram; 1024 works well across typical N
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        histogram_atomic_kernel[grid](flat, counts, N, BLOCK)

        # Compute expert_offsets (prefix sum of counts), including offset[0] = 0
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        expert_offsets[0] = 0  # host sets zeroth offset

        # Run prefix scan kernel: process entire 256 experts; BLOCK_EXPERTS must be >= num_experts
        BLOCK_EXPERTS = 256
        prefix_scan_kernel[(1,)](counts, expert_offsets, num_experts, BLOCK_EXPERTS)

        # Return sorted_token_indices (num_tokens,) and expert_offsets (num_experts+1,)
        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
