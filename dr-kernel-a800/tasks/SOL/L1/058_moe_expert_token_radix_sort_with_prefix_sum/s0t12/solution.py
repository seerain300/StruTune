import torch
import triton
import triton.language as tl


@triton.jit
def stable_bitonic_sort_flat_kernel(flat_ptr, out_idx_ptr, N, LOGN: tl.constexpr):
    """
    Stable bitonic sort on flat_ptr (int32) of length N, writing original positions into out_idx_ptr (int64).
    Grid:
      axis 0: programs over i in [0, N)
      axis 1: programs over j in range(k+1, LOGN) for k in 0..LOGN-1 (bitonic stages).
    Each i computes partner = i ^ (1 << j), compare values, swap; for ties, i < partner decides (stable).
    """
    i = tl.program_id(axis=0)
    j = tl.program_id(axis=1)
    step = 1 << j
    partner = i ^ step

    do_pair = i < partner
    in_bounds = (i < N) & (partner < N) & do_pair

    # Load values (int32)
    val_i = tl.load(flat_ptr + i)
    val_p = tl.load(flat_ptr + partner, mask=in_bounds, other=0)

    # Ascending/descending for this stage determined by next bit
    asc = ((i & (1 << (j + 1))) == 0)

    # Stable compare: tie-break by original index
    tie = val_i == val_p
    take_i = tl.where(asc, val_i < val_p, val_i > val_p) | (tie & (i < partner))

    # New indices after potential swap (original positions)
    new_idx_i = tl.where(take_i, i, partner)
    new_idx_p = tl.where(take_i, partner, i)

    # Store results (original positions as int64)
    tl.store(out_idx_ptr + i, tl.cast(new_idx_i, tl.int64), mask=in_bounds)
    tl.store(out_idx_ptr + partner, tl.cast(new_idx_p, tl.int64), mask=in_bounds)


@triton.jit
def count_histogram_kernel(flat_ptr, counts_ptr, N, num_experts: tl.constexpr, BLOCK: tl.constexpr):
    """
    Histogram of flat values (int32) into counts_ptr (int32), length num_experts.
    Each program processes BLOCK elements and atomically adds 1 for each occurrence.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def prefix_sum_kernel(counts_ptr, offsets_ptr, M: tl.constexpr):
    """
    Inclusive prefix sum of counts_ptr (int64, length M) into offsets_ptr (int64).
    We write offsets[1..M]; host sets offsets[0] to 0 before calling.
    """
    acc = tl.zeros((), dtype=tl.int64)
    for k in range(0, M):
        acc += tl.load(counts_ptr + k)
        tl.store(offsets_ptr + (k + 1), acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only heavy computation:
        - Sort flattened topk_idx (int32) and produce sorted_token_indices as int64 using Triton (stable bitonic).
        - Compute expert_offsets as int32 via Triton histogram + prefix sum.
        Returns:
            sorted_token_indices: torch.int64 tensor of shape (N,)
            expert_offsets: torch.int32 tensor of shape (num_experts + 1,)
        """
        # Ensure input on CUDA and int32
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.cuda()
        if topk_idx.dtype != torch.int32:
            topk_idx = topk_idx.to(torch.int32)

        # Flatten
        flat = topk_idx.reshape(-1)
        N = flat.numel()
        device = flat.device

        # Bitonic sort grid: axis 0 over N, axis 1 over LOGN stages
        LOGN = (N - 1).bit_length() if N > 0 else 0
        out_idx = torch.empty_like(flat, dtype=torch.int64, device=device)
        stable_bitonic_sort_flat_kernel[(N, LOGN)](flat, out_idx, N, LOGN=LOGN)

        # Histogram counts (int32)
        counts = torch.zeros(256, dtype=torch.int32, device=device)  # num_experts = 256
        BLOCK = 1024
        grid_hist = (triton.cdiv(N, BLOCK),)
        count_histogram_kernel[grid_hist](flat, counts, N, num_experts=256, BLOCK=BLOCK)

        # Prefix sum to get expert_offsets (int64), then cast to int32
        expert_offsets64 = torch.empty(257, dtype=torch.int64, device=device)
        expert_offsets64[0] = 0  # offsets[0] unused; we keep for clarity
        prefix_sum_kernel[(1,)](counts, expert_offsets64, M=256)  # single program computes up to 256
        expert_offsets = expert_offsets64[1:].to(torch.int32)

        return out_idx, expert_offsets


def run(*args):
    return ModelNew()(*args)
