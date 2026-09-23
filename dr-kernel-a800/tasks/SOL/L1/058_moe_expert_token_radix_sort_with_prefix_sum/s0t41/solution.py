import torch
import triton
import triton.language as tl


@triton.jit
def count_histogram_kernel(flat_ptr, counts_ptr, N, num_experts: tl.constexpr, BLOCK: tl.constexpr):
    """
    Compute histogram of flat values (int32) into counts_ptr (int32), length num_experts.
    Each program processes BLOCK elements and atomically adds 1 for each occurrence.
    """
    pid = tl.program_id(axis=0)
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
    Single-program sequential loop is fine for small num_experts (e.g., 256).
    """
    acc = tl.zeros((), dtype=tl.int64)
    for i in tl.static_range(0, num_experts):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + (i + 1), acc)


@triton.jit
def stable_bitonic_sort_kernel(flat_ptr, out_idx_ptr, N, LOGN: tl.constexpr):
    """
    Stable bitonic sort on flat_ptr (int32). out_idx_ptr holds int64 original positions 0..N-1.
    Grid: axis 0 = N (one program per element), axis 1 = LOGN (one stage per bitonic dimension).
    For each stage j (tl.program_id(1)), each program i compares with partner = i ^ (1 << j).
    Ascending if ((i & (1 << (j+1))) == 0) else descending. Stability: tie-break by i < partner.
    Two programs per pair (i and partner) execute the compare-and-swap. They both read each other's data
    and write back to their own out_idx slots. No race since stores are from different pids.
    """
    pid0 = tl.program_id(axis=0)  # element index
    j = tl.program_id(axis=1)     # bitonic stage
    # partner index for this stage
    partner = pid0 ^ (1 << j)
    pair_mask = pid0 < partner    # process each pair once

    # Load current and partner values
    val0 = tl.load(flat_ptr + pid0)
    val_p = tl.load(flat_ptr + partner)
    idx0 = tl.cast(pid0, tl.int64)
    idx_p = tl.cast(partner, tl.int64)

    # Ascending if ((pid0 & (1 << (j+1))) == 0)
    asc = ((pid0 & (1 << (j + 1))) == 0)

    # Determine swap based on ascending/descending and stability (equal values -> lower idx first)
    gt = val0 > val_p
    lt = val0 < val_p
    eq = val0 == val_p
    # For ascending: swap if (val0 > val_p) or (equal and pid0 > partner)
    # For descending: swap if (val0 < val_p) or (equal and pid0 < partner)
    swap_asc = (val0 > val_p) | ((val0 == val_p) & (pid0 > partner))
    swap_desc = (val0 < val_p) | ((val0 == val_p) & (pid0 < partner))
    swap = tl.where(asc, swap_asc, swap_desc) & pair_mask

    # Compute new values (only meaningful for pid0; partner program does symmetric update)
    minv = tl.where(val0 < val_p, val0, val_p)
    maxv = tl.where(val0 > val_p, val0, val_p)
    mini = tl.where(val0 < val_p, idx0, idx_p)
    maxi = tl.where(val0 > val_p, idx0, idx_p)
    new_val0 = tl.where(swap, maxv, minv)
    new_idx0 = tl.where(swap, tl.cast(maxi, tl.int64), tl.cast(mini, tl.int64))

    # Store back for pid0. Partner program will do reciprocal update for partner.
    tl.store(out_idx_ptr + pid0, new_idx0, mask=pair_mask)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation of:
          - Stable sort of flat indices (int32) -> sorted_token_indices (int64)
          - Histogram of flat indices -> expert_offsets (int32)
        """
        # Ensure topk_idx is on CUDA and contiguous
        assert topk_idx.is_cuda, "Input must be on CUDA device"
        topk_idx = topk_idx.contiguous()

        # Flatten
        flat = topk_idx.view(-1)  # int32, length N
        N = flat.numel()
        num_experts = 256

        # 1) Stable bitonic sort using Triton: produce sorted_token_indices (int64)
        out_idx = torch.empty(N, dtype=torch.int64, device=flat.device)

        # Grid over elements and bitonic stages. LOGN should cover up to N. For N<=4096, LOGN=12 is sufficient.
        grid = (N, 12)
        stable_bitonic_sort_kernel[grid](flat, out_idx, N, LOGN=12)

        # 2) Histogram via Triton
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        BLOCK = 1024
        grid_counts = (triton.cdiv(N, BLOCK),)
        count_histogram_kernel[grid_counts](flat, counts, N, num_experts=256, BLOCK=BLOCK)

        # 3) Prefix sum via Triton (inclusive prefix sum of counts -> offsets[1:])
        offsets = torch.empty(num_experts + 1, dtype=torch.int64, device=flat.device)
        offsets.fill_(0)  # offsets[0] will be 0
        grid_ps = (1,)
        prefix_sum_kernel[grid_ps](counts, offsets, num_experts=256)

        # Cast offsets to int32 as required by original
        expert_offsets = offsets.to(torch.int32)

        return out_idx, expert_offsets


def run(*args):
    return ModelNew()(*args)
