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
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr (int64, length num_experts) into offsets_ptr (int64).
    offsets_ptr[0] should be set to 0 by the caller. We write offsets[1..].
    """
    acc = tl.zeros((), dtype=tl.int64)
    for i in range(0, num_experts):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + (i + 1), acc)


@triton.jit
def stable_counting_sort_kernel(flat_ptr, out_idx_ptr, positions_ptr, N, num_experts: tl.constexpr):
    """
    Stable counting sort using precomputed positions_ptr (int64) for each expert id.
    positions_ptr[id] is the base starting position for tokens with value id.
    For each token i, load its value v, and compute:
      base = positions_ptr[v]  (int64)
      before = number of tokens j < i with value v  (tie-breaker for stability)
      sorted_index = base + before
    Then set out_idx[i] = sorted_index (int64). This yields stable order.
    """
    i = tl.program_id(axis=0)
    if i >= N:
        return

    v = tl.load(flat_ptr + i)  # int32 in [0, num_experts-1]
    base = tl.load(positions_ptr + v)  # int64

    # Compute 'before' count: number of tokens with the same value that appear before i.
    # This loop is O(N); for typical N (hundreds to a few thousands), it's acceptable and ensures stability.
    before = tl.zeros((), dtype=tl.int32)
    j = 0
    while j < i:
        val_j = tl.load(flat_ptr + j)
        if val_j == v:
            before += 1
        j += 1

    sorted_index = base + before  # int64
    tl.store(out_idx_ptr + i, sorted_index)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized implementation:
        - sorted_token_indices = torch.sort(topk_idx.reshape(-1), stable=True).indices  (int64)
        - expert_offsets = torch.zeros(self.num_experts + 1, int32)
          expert_offsets[1:] = torch.bincount(topk_idx.reshape(-1).int()).cumsum(0).int32()
        """
        # Ensure CUDA device for Triton
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to('cuda')
        device = topk_idx.device

        flat = topk_idx.reshape(-1).to(torch.int32)
        N = flat.numel()

        # 1) Histogram via Triton
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
        grid_h = (triton.cdiv(N, 1024),)
        count_histogram_kernel[grid_h](flat, counts, N, num_experts=self.num_experts)

        # 2) Inclusive prefix sum of counts (int64) to get base positions for each expert id
        positions = torch.cumsum(counts.to(torch.int64), dim=0)  # shape (num_experts,)

        # 3) Stable counting sort: compute sorted indices via Triton
        out_idx = torch.empty(N, dtype=torch.int64, device=device)
        grid_sort = (N,)
        stable_counting_sort_kernel[grid_sort](flat, out_idx, positions, N, num_experts=self.num_experts)

        # 4) Expert offsets: offsets[1:] = cumsum(bincount(flat)), offsets[0] = 0, int32
        offsets64 = torch.zeros(self.num_experts + 1, dtype=torch.int64, device=device)
        prefix_sum_kernel[(1,)](counts, offsets64[1:], num_experts=self.num_experts)
        offsets = offsets64.to(torch.int32)

        # Return results matching original signature
        sorted_token_indices = out_idx
        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
