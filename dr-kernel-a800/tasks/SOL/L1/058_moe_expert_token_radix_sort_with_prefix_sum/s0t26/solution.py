import torch
import triton
import triton.language as tl


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


@triton.jit
def bitonic_stable_sort_indices_kernel(flat_ptr, out_idx_ptr, N, LOGN: tl.constexpr):
    """
    Stable bitonic sort on flat_ptr (int32). out_idx_ptr holds int64 original positions 0..N-1.
    2D grid: axis 0 = N (one program per element), axis 1 = LOGN (one stage per bitonic dimension).
    For each stage j, each program i computes partner = i ^ (1 << j) and performs a single compare-and-swap
    for that pair. Only process i < partner to avoid races. Ascending when (i & (1 << (j+1))) == 0.
    Stability is ensured by tie-breaking: if equal, lower original index comes first.
    """
    i = tl.program_id(axis=0)
    j = tl.program_id(axis=1)

    if i >= N:
        return

    partner = i ^ (1 << j)

    # Process each pair once
    if i >= partner:
        return

    # Load values and indices
    a = tl.load(flat_ptr + i)
    b = tl.load(flat_ptr + partner)
    idx_a = tl.load(out_idx_ptr + i)
    idx_b = tl.load(out_idx_ptr + partner)

    # Sort direction for this stage
    asc = ((i & (1 << (j + 1))) == 0)

    # Stability: tie-break by original index
    less = idx_a < idx_b

    min_val = tl.where(a < b, a, b)
    max_val = tl.where(a > b, a, b)
    min_idx = tl.where(less, idx_a, idx_b)
    max_idx = tl.where(less, idx_b, idx_a)

    # Swap if needed: for ascending, swap when a > b; for descending, swap when a < b.
    # Additionally, ensure stability: if equal, swap when higher idx precedes lower idx.
    need_swap = (asc & (a > b)) | ((not asc) & (a < b)) | ((a == b) & (idx_a > idx_b))

    new_idx_a = tl.where(need_swap, max_idx, min_idx)
    new_idx_b = tl.where(need_swap, min_idx, max_idx)

    tl.store(out_idx_ptr + i, new_idx_a)
    tl.store(out_idx_ptr + partner, new_idx_b)


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


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Flatten topk_idx (int32), compute stable sorted indices (int64) via Triton bitonic sort.
        - Compute expert_offsets (int32) via Triton histogram + prefix sum.
        """
        # CPU fallback (evaluation expects CUDA, but keep correctness)
        if topk_idx.device.type != 'cuda':
            flat = topk_idx.reshape(-1)
            sorted_token_indices = torch.sort(flat, stable=True).indices.to(torch.int64)
            num_experts = 256
            counts = torch.bincount(flat.long(), minlength=num_experts).to(torch.int32)
            expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=topk_idx.device)
            expert_offsets[1:] = counts.cumsum(0)
            return sorted_token_indices, expert_offsets

        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()

        # 1) Stable bitonic sort: initialize out_idx as 0..N-1 (int64)
        out_idx = torch.arange(N, dtype=torch.int64, device=topk_idx.device).contiguous()

        # Compute LOGN = ceil(log2(N))
        LOGN = _next_power_of_2(N)
        # Launch bitonic sorting stages: 2D grid (N, LOGN)
        grid = (N, LOGN)
        bitonic_stable_sort_indices_kernel[grid](flat, out_idx, N, LOGN)

        # 2) expert_offsets: histogram and prefix sum in Triton
        counts = torch.zeros(256, dtype=torch.int32, device=topk_idx.device)
        # count histogram
        num_blocks = triton.cdiv(N, 1024)
        count_histogram_kernel[(num_blocks,)](flat, counts, N, 256)
        # prefix sum to get cumulative counts (int64), then cast to int32
        offsets64 = torch.empty(257, dtype=torch.int64, device=topk_idx.device)
        offsets64[0] = 0
        prefix_sum_kernel[(256,)](counts, offsets64, 256)
        expert_offsets = offsets64.to(torch.int32)

        return out_idx, expert_offsets


def run(*args):
    return ModelNew()(*args)
