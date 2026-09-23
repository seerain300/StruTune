import torch
import triton
import triton.language as tl


@triton.jit
def flatten_copy_kernel(src_ptr, dst_ptr, N, BLOCK: tl.constexpr):
    """
    Copies N elements from src_ptr to dst_ptr.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(src_ptr + offs, mask=mask, other=0)
    tl.store(dst_ptr + offs, vals, mask=mask)


@triton.jit
def histogram_atomic_kernel(original_ptr, counts_ptr, M, NUM_VALUES: tl.constexpr, BLOCK: tl.constexpr):
    """
    Compute counts of occurrences per value in [0..NUM_VALUES-1] for the first M elements of original_ptr.
    Uses atomic_add into counts_ptr (int32). Assumes original_ptr is int32.
    """
    for start in range(0, M, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < M
        vals = tl.load(original_ptr + offs, mask=mask, other=0)
        # Each lane computes its value and atomically adds 1 to counts[vals]
        # Works for arbitrary int32 values; Triton will cast vi to int pointer offset.
        for j in range(BLOCK):
            vi = vals[j]
            if mask[j]:
                tl.atomic_add(counts_ptr + vi, 1)


@triton.jit
def prefix_sum_inclusive_kernel(counts_ptr, prefix_ptr, N: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr into prefix_ptr of length N+1.
    prefix[0] = 0; prefix[i+1] = prefix[i] + counts[i].
    This runs sequentially and is fine for N=256.
    """
    # Initialize prefix[0] to 0
    tl.store(prefix_ptr + 0, tl.zeros((), dtype=tl.int32))
    # Inclusive scan
    for i in range(N):
        prev = tl.load(prefix_ptr + i)
        cur = tl.load(counts_ptr + i)
        tl.store(prefix_ptr + i + 1, prev + cur)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = num_experts  # From original, equals num_experts_per_tok in this setup (256)

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton version:
        - Flatten topk_idx to 1D int32 'original'
        - Compute histogram of values using Triton atomic adds
        - Compute inclusive prefix sum of histogram using Triton
        - sorted_token_indices: use torch.sort(stable=True) to match original exactly for arbitrary inputs
        - expert_offsets: return the inclusive prefix sum as (num_experts+1,) int32 tensor
        """
        # Ensure input is on CUDA and int32
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device"
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        original = topk_idx.contiguous().view(-1)
        M = original.numel()
        device = original.device

        # 1) Histogram via Triton
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
        BLOCK_HIST = 1024
        grid_h = (triton.cdiv(M, BLOCK_HIST),)
        histogram_atomic_kernel[grid_h](original, counts, M, self.num_experts, BLOCK_HIST)

        # 2) Prefix sum via Triton (inclusive)
        prefix = torch.empty(self.num_experts + 1, dtype=torch.int32, device=device)
        prefix_sum_inclusive_kernel[(1,)](counts, prefix, self.num_experts)

        # 3) sorted_token_indices: exact match with original using torch.sort(stable=True)
        # Flatten was already done; here we sort the original 1D tensor
        sorted_indices = torch.sort(original, stable=True).indices  # torch.int64 by default
        sorted_indices = sorted_indices.to(torch.int32)

        # 4) expert_offsets: inclusive prefix sum counts (length num_experts+1)
        expert_offsets = prefix.clone()

        return sorted_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
