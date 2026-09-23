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
    acc = tl.zeros((), dtype=tl.int64)
    for i in range(0, num_experts):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + (i + 1), acc)


@triton.jit
def stable_bitonic_sort_kernel(flat_ptr, out_idx_ptr, N, LOGN: tl.constexpr):
    """
    Stable bitonic sort on flat_ptr (int32). out_idx_ptr holds int64 original positions 0..N-1.
    We sort flat_ptr values and write the final indices (sorted positions) into out_idx_ptr as int64.
    For each stage j and k, each program i compares with partner = i ^ (1 << j).
    Ascending if (i & (1 << (k+1))) == 0 else descending. Stability: tie-break by i < partner.
    """
    # N is runtime, but LOGN is constexpr (compile-time). We use a 2D grid: (N, LOGN).
    i = tl.program_id(axis=0)  # position index
    j = tl.program_id(axis=1)  # stage index
    # Compute partner via XOR; Triton supports ^ on vectors
    step = 1 << j
    partner = i ^ step
    # Bounds and single-pair processing mask
    do_pair = i < partner
    in_bounds = (i < N) & (partner < N) & do_pair
    # Load current flat values at i and partner
    a = tl.load(flat_ptr + i)
    b = tl.load(flat_ptr + partner, mask=in_bounds, other=0)
    # Load current indices (original positions)
    idx_i = tl.load(out_idx_ptr + i)
    idx_p = tl.load(out_idx_ptr + partner, mask=in_bounds, other=0)
    # Ascending if (i & (1 << (j+1))) == 0; note: j+1 < LOGN by construction
    asc = ((i & (1 << (j + 1))) == 0)
    # Stability tie-breaking: if equal, smaller original index comes first
    tie = a == b
    # Determine min/max ignoring tie, then apply tie-break
    less = a < b
    greater = a > b
    a_le_b = less | tie
    a_ge_b = greater | tie
    # Build new values for positions i and partner
    # If ascending: i gets min; partner gets max. If descending: i gets max; partner gets min.
    new_i_val = tl.where(asc, tl.where(a_le_b, a, b), tl.where(a_ge_b, a, b))
    new_p_val = tl.where(asc, tl.where(a_le_b, b, a), tl.where(a_ge_b, b, a))
    # Stable tie-break: if equal, i should take 'a' and partner 'b'
    new_i_val = tl.where(tie & (i < partner), a, new_i_val)
    new_p_val = tl.where(tie & (i < partner), b, new_p_val)
    # New indices (int64) follow the same ordering of values
    new_i_idx = tl.where(asc, tl.where(a_le_b, idx_i, idx_p), tl.where(a_ge_b, idx_p, idx_i))
    new_p_idx = tl.where(asc, tl.where(a_le_b, idx_p, idx_i), tl.where(a_ge_b, idx_i, idx_p))
    # Store back
    tl.store(out_idx_ptr + i, new_i_idx, mask=in_bounds)
    tl.store(out_idx_ptr + partner, new_p_idx, mask=in_bounds)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Sorts the flattened topk_idx (int32) using a Triton stable bitonic sort and returns int64 indices.
        - Computes expert_offsets via Triton (histogram + prefix sum).
        Returns:
            sorted_token_indices: torch.int64 tensor of shape (N,)
            expert_offsets: torch.int32 tensor of shape (num_experts + 1,)
        """
        # Ensure input is on CUDA and int32
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.cuda()
        if topk_idx.dtype != torch.int32:
            topk_idx = topk_idx.to(torch.int32)

        # Flatten
        flat = topk_idx.reshape(-1)
        N = flat.numel()
        device = flat.device

        # Output buffer for sorted indices (int64)
        out_idx = torch.arange(N, dtype=torch.int64, device=device).contiguous()

        # Compute LOGN as constexpr: number of bits to cover N-1
        # We pass LOGN as a compile-time constant to Triton
        LOGN = (N - 1).bit_length() if N > 0 else 0

        # Launch bitonic sort kernel: 2D grid (N, LOGN)
        grid = (N, LOGN)
        stable_bitonic_sort_kernel[grid](flat, out_idx, N, LOGN=LOGN, num_warps=4)

        # Compute expert_offsets via Triton histogram + prefix sum
        num_experts = 256  # matches original default
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        # Kernel: count histogram
        BLOCK = 1024
        grid_counts = ((N + BLOCK - 1) // BLOCK,)
        count_histogram_kernel[grid_counts](flat, counts, N, num_experts=num_experts, num_warps=4)
        # Kernel: prefix sum (int64)
        offsets = torch.empty(num_experts + 1, dtype=torch.int64, device=device)
        prefix_sum_kernel[(num_experts,)](counts, offsets, num_experts=num_experts, num_warps=4)
        # Cast to int32 as in original: expert_offsets[1:] = cumulative counts
        expert_offsets = offsets.to(torch.int32)

        return out_idx, expert_offsets


def run(*args):
    return ModelNew()(*args)
