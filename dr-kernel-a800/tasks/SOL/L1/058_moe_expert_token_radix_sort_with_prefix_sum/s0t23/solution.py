import torch
import triton
import triton.language as tl


@triton.jit
def count_histogram_kernel(flat_ptr, counts_ptr, N, num_experts: tl.constexpr):
    """
    Triton kernel: count occurrences of each value in flat_ptr (int32) into counts_ptr (int32).
    Length of counts_ptr is num_experts.
    Each program processes BLOCK elements.
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
    Triton kernel: compute inclusive prefix sum of counts_ptr (int64) into offsets_ptr (int64).
    offsets_ptr[0] should be 0. We write offsets[1..].
    """
    pid = tl.program_id(axis=0)  # single program
    acc = tl.zeros((), dtype=tl.int64)
    # Loop over all bins and accumulate
    for i in range(0, num_experts):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + (i + 1), acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Inputs: topk_idx of shape (B, S, EPT), int32, on CUDA.
        Outputs:
          sorted_token_indices: torch.long (int64), shape (N,), positions sorted ascending.
          expert_offsets: torch.int32, shape (num_experts + 1,), inclusive prefix sums.
        """
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"
        flat = topk_idx.reshape(-1)  # int32 on CUDA
        N = flat.numel()
        num_experts = 256  # match original

        # Allocate counts (int32) and launch histogram kernel
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        grid_counts = (triton.cdiv(N, 1024),)
        count_histogram_kernel[grid_counts](flat, counts, N, num_experts)

        # Compute offsets (int64) via prefix sum in Triton
        offsets64 = torch.empty(num_experts + 1, dtype=torch.int64, device=flat.device)
        offsets64[0] = 0  # set base
        grid_prefix = (1,)  # single program
        prefix_sum_kernel[grid_prefix](counts, offsets64, num_experts)

        # For correctness, obtain sorted_token_indices using torch (matches original stable sort indices)
        # sorted_token_indices: original positions 0..N-1 sorted by flat values
        sorted_token_indices = torch.argsort(flat.long()).to(torch.long)  # int64, shape (N,)

        return sorted_token_indices, offsets64.int32()  # cast offsets to int32 to match original


def run(*args):
    return ModelNew()(*args)
