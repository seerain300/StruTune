import torch
import triton
import triton.language as tl


@triton.jit
def count_histogram_kernel(flat_ptr, counts_ptr, N, num_experts: tl.constexpr, BLOCK: tl.constexpr):
    """
    Triton kernel to compute histogram of flat values (int32) into counts_ptr (int32), length num_experts.
    Each program processes BLOCK elements and atomically adds 1 for each occurrence.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32 values in [0, num_experts-1]
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Triton kernel to compute inclusive prefix sum of counts_ptr (int64, length num_experts) into offsets_ptr (int64).
    offsets_ptr[0] should be set to 0 by the caller. We write offsets[1..].
    """
    acc = tl.zeros((), dtype=tl.int64)
    for i in range(0, num_experts):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + (i + 1), acc)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version of the original run function:
        - sorted_token_indices = torch.sort(flat, stable=True).indices (int64).
        - expert_offsets computed via Triton histogram + prefix sum (int32).
        """
        # Flatten and ensure contiguity; keep int32
        flat = topk_idx.reshape(-1).contiguous().to(torch.int32)
        N = flat.numel()

        # sorted_token_indices: use torch for correctness and stable=True
        sorted_token_indices = torch.sort(flat, stable=True).indices  # dtype: int64, shape: (N,)

        # expert_offsets via Triton histogram + prefix sum
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=flat.device)

        # Launch histogram kernel
        BLOCK = 2048
        grid = (triton.cdiv(N, BLOCK),)
        count_histogram_kernel[grid](flat, counts, N, self.num_experts, BLOCK=BLOCK, num_warps=4)

        # Compute inclusive prefix sum of counts (int64) and create offsets (int32)
        offsets_int64 = torch.zeros(self.num_experts + 1, dtype=torch.int64, device=flat.device)
        # prefix sum kernel writes offsets[1..]
        prefix_sum_kernel[(self.num_experts,)](counts, offsets_int64[1:], self.num_experts, num_warps=1)

        expert_offsets = offsets_int64.to(torch.int32)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
