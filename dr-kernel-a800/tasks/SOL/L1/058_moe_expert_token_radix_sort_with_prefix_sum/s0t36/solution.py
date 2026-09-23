import torch
import triton
import triton.language as tl


@triton.jit
def count_histogram_kernel(flat_ptr, counts_ptr, N, num_experts: tl.constexpr):
    """
    Compute histogram of flat values (int32) into counts_ptr (int32), length num_experts.
    Each program processes BLOCK elements and atomically adds 1 for each occurrence.
    """
    pid = tl.program_id(axis=0)
    BLOCK = 1024
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32 values in [0, num_experts-1]
    # Atomic add 1 for each occurrence into counts[val]
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr (int64, length num_experts) into offsets_ptr (int64).
    offsets_ptr[0] should be set to 0 by the caller. We write offsets[1..].
    """
    pid = tl.program_id(axis=0)
    # Single program computes the prefix sum sequentially
    acc = tl.zeros((), dtype=tl.int64)
    for i in range(0, num_experts):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + (i + 1), acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten the input indices to 1D
        flat = topk_idx.reshape(-1).contiguous()

        # Use PyTorch for stable sort to ensure correct int64 indices
        sorted_token_indices, _ = torch.sort(flat, stable=True)  # indices are int64

        # Compute histogram in Triton
        num_experts = 256
        N = flat.numel()
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        grid_count = (triton.cdiv(N, 1024),)
        count_histogram_kernel[grid_count](flat, counts, N, num_experts)

        # Compute prefix sum of counts in Triton and form expert_offsets (int32)
        # Allocate offsets as int64 in the kernel for clarity; cast to int32 after.
        offsets = torch.empty(num_experts + 1, dtype=torch.int64, device=flat.device)
        grid_ps = (1,)
        prefix_sum_kernel[grid_ps](counts, offsets, num_experts)
        expert_offsets = offsets.to(torch.int32)

        # sorted_token_indices must be int64
        return sorted_token_indices.to(torch.int64), expert_offsets


def run(*args):
    return ModelNew()(*args)
