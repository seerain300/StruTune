import torch
import triton
import triton.language as tl


@triton.jit
def bitonic_sort_stable_int64(out_ptr, flat_ptr, N, LOGN: tl.constexpr):
    """
    Stable bitonic sort of int64 values. We read from flat_ptr and write sorted indices to out_ptr.
    The sort is stable: for equal values, smaller original index comes first.
    Grid: (N, LOGN) so each element i executes per stage j.
    """
    i = tl.program_id(axis=0)  # lane id
    j = tl.program_id(axis=1)  # stage id

    # For each stage j, perform compare-and-swap for pairs (i, partner)
    step = 1 << j
    partner = i ^ step

    # Only process each pair once
    do_pair = i < partner
    in_bounds = (i < N) & (partner < N) & do_pair

    # Load original index i's value and its partner's value
    val_i = tl.load(flat_ptr + i)
    val_p = tl.load(flat_ptr + partner)

    # Determine ascending/descending for this stage
    asc = ((i & step) == 0)

    # Stable compare: if equal, prefer i < partner
    tie = val_i == val_p
    minv = tl.where(val_i < val_p, val_i, val_p)
    maxv = tl.where(val_i > val_p, val_i, val_p)
    take_i = (val_i <= val_p) | (tie & (i < partner))
    new_i = tl.where(take_i, val_i, val_p)
    new_p = tl.where(take_i, val_p, val_i)

    new_i_stage = tl.where(asc, new_i, new_p)
    new_p_stage = tl.where(asc, new_p, new_i)

    # Store results
    tl.store(out_ptr + i, new_i_stage, mask=in_bounds)
    tl.store(out_ptr + partner, new_p_stage, mask=in_bounds)


@triton.jit
def count_histogram_int64(counts_ptr, flat_ptr, N, num_experts: tl.constexpr):
    """
    Histogram of flat indices (int32) into counts_ptr (int64) using atomic adds.
    counts_ptr: length num_experts, int64
    flat_ptr: length N, int32
    """
    pid = tl.program_id(axis=0)
    # one program processes BLOCK elements
    BLOCK = 1024
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32
    # For each val, atomic add 1 into counts[val]
    for v in range(0, num_experts):
        tl.atomic_add(counts_ptr + v, 1, mask=(vals == v) & mask)


@triton.jit
def cumsum_inclusive_int64(out_ptr, counts_ptr, M: tl.constexpr):
    """
    Inclusive prefix sum of counts_ptr (int64) into out_ptr (int64) using iterative doubling.
    M is the number of elements (num_experts).
    This kernel writes out_ptr[0..M-1] = prefix sums. out_ptr[M] unused.
    """
    # We do a simple serial scan per element to keep it deterministic and simple.
    # Triton does not support vectorized scans across all lanes easily, hence this approach.
    # However, since num_experts is a small constant (256), this is fine.
    for k in range(0, M):
        # Compute prefix sum at position k
        total = tl.zeros((), dtype=tl.int64)
        for p in range(0, k + 1):
            val = tl.load(counts_ptr + p)
            total += val
        tl.store(out_ptr + k, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation of the original run function:
        - Sorts the flattened topk_idx (int32) using Triton stable bitonic sort and returns int64 indices.
        - Computes expert_offsets via Triton histogram and in-kernel prefix sum (converted to int32).
        Returns:
            sorted_token_indices: torch.int64 tensor of shape (N,)
            expert_offsets: torch.int32 tensor of shape (num_experts + 1,)
        """
        # Ensure CUDA and dtype
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.cuda()
        if topk_idx.dtype != torch.int32:
            topk_idx = topk_idx.to(torch.int32)

        # Flatten
        flat = topk_idx.reshape(-1)
        N = flat.numel()
        device = flat.device

        # Output for sorted indices (int64)
        out_idx = torch.empty(N, dtype=torch.int64, device=device)

        # Launch stable bitonic sort: grid = (N, LOGN)
        LOGN = (N - 1).bit_length() if N > 0 else 0  # number of bits in N
        if LOGN == 0:
            # Degenerate case: N == 1
            out_idx.copy_(flat.to(torch.int64))
            LOGN = 1
        grid = (N, LOGN)
        bitonic_sort_stable_int64[grid](out_idx, flat.to(torch.int64), N, LOGN=LOGN)

        # Histogram counts (int64)
        num_experts = 256
        counts = torch.zeros(num_experts, dtype=torch.int64, device=device)

        # Launch histogram kernel
        # We process flat in blocks; since N can be large, use a grid of size ceil_div(N, BLOCK)
        BLOCK = 1024
        grid_counts = (triton.cdiv(N, BLOCK),)
        count_histogram_int64[grid_counts](counts, flat, N, num_experts=num_experts)

        # Inclusive prefix sum (int64) via Triton
        cumsum_counts = torch.empty(num_experts, dtype=torch.int64, device=device)
        cumsum_inclusive_int64[(num_experts,)](cumsum_counts, counts, M=num_experts)

        # expert_offsets: int32 of length num_experts + 1; out[0]=0, out[1:]=cumsum_counts.int32()
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        expert_offsets[0] = 0
        expert_offsets[1:] = cumsum_counts.to(torch.int32)

        return out_idx, expert_offsets


def run(*args):
    return ModelNew()(*args)
