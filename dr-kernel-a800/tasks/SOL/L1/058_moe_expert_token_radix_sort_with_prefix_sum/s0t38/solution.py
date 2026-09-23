import torch
import triton
import triton.language as tl


@triton.jit
def count_histogram_kernel(flat_ptr, counts_ptr, N, NUM_EXPS: tl.constexpr):
    """
    Count occurrences of each value in flat_ptr (int32) into counts_ptr (int32) of length NUM_EXPS.
    Use atomic adds to avoid race conditions. One program processes a block of elements.
    """
    pid = tl.program_id(axis=0)
    BLOCK = 1024
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32 values in [0, NUM_EXPS-1]
    # Atomic add 1 for each occurrence into counts[val]
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def prefix_sum_kernel(counts_ptr, offsets_ptr, NUM_EXPS: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr (int64, length NUM_EXPS) into offsets_ptr (int64, length NUM_EXPS).
    We write to offsets[1..], caller sets offsets[0] = 0.
    """
    acc = tl.zeros((), dtype=tl.int64)
    for i in range(0, NUM_EXPS):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + (i + 1), acc)


@triton.jit
def counting_sort_kernel(flat_ptr, out_idx_ptr, counts_ptr, offsets_ptr, N, NUM_EXPS: tl.constexpr):
    """
    Stable counting sort: out_idx_ptr holds int64 original indices 0..N-1 in sorted order by flat_ptr values.
    counts_ptr (int32) holds per-value counts (length NUM_EXPS).
    offsets_ptr (int64) holds cumulative sums for each value (length NUM_EXPS), offsets[0] should be 0.
    Each program processes a block of elements. For each element i, if flat_ptr[i] == v, write i to out_idx at offsets[v].
    """
    pid = tl.program_id(axis=0)
    BLOCK = 1024
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32 values in [0, NUM_EXPS-1]
    # For each element in this block, if valid, compute its offset based on counts and write original index
    for k in range(0, BLOCK):
        i = offs[k]
        valid = i < N
        v = vals[k]
        # Load current cumulative offset for value v; offsets[0] is 0 by caller
        idx = tl.load(offsets_ptr + v)  # int64
        # Store original index i (as int64) into sorted output at position idx
        tl.store(out_idx_ptr + idx, i.to(tl.int64), mask=valid)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
            topk_idx: (B, S, EPT) int32 on CUDA
        Outputs:
            sorted_token_indices: (N,) int64, original positions sorted by expert IDs
            expert_offsets: (num_experts+1,) int32, prefix counts per expert
        """
        assert topk_idx.is_cuda, "Input must be on CUDA device for Triton kernels"
        # Flatten and ensure contiguous
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()

        # Prepare output and buffers
        out_idx = torch.empty(N, dtype=torch.int64, device=flat.device)
        counts = torch.zeros(self.num_experts + 1, dtype=torch.int32, device=flat.device)  # counts for 0..num_experts

        # Launch histogram kernel: counts[1..] will be filled; counts[0] left 0
        BLOCK_HIST = 1024
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        count_histogram_kernel[grid_hist](flat, counts, N, self.num_experts)

        # Compute prefix sums for offsets (int64)
        offsets64 = torch.empty(self.num_experts + 1, dtype=torch.int64, device=flat.device)
        # Set offsets[0] = 0 explicitly
        offsets64[0] = 0
        # Launch prefix sum kernel on counts[1..]
        grid_ps = (1,)  # single program computes prefix sum
        prefix_sum_kernel[grid_ps](counts[1:], offsets64[1:], self.num_experts)

        # Now perform stable counting sort: out_idx will hold original indices sorted by flat values
        BLOCK_SORT = 1024
        grid_sort = (triton.cdiv(N, BLOCK_SORT),)
        counting_sort_kernel[grid_sort](flat, out_idx, counts[1:], offsets64, N, self.num_experts)

        # expert_offsets is the cumulative counts per expert; cast to int32 to match original
        expert_offsets = counts[1:].to(torch.int32)
        return out_idx, expert_offsets


def run(*args):
    return ModelNew()(*args)
