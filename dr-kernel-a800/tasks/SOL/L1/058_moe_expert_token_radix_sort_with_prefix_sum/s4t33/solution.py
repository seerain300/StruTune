import math
import torch
import triton
import triton.language as tl


@triton.jit
def bitonic_sort_stable_kernel(values_ptr, idxs_ptr, N, LOGN: tl.constexpr):
    """
    Stable bitonic sort over N int32 values in values_ptr using a permutation array idxs_ptr of length N.
    idxs_ptr is initialized to [0, 1, ..., N-1] and becomes the sorted permutation (ascending by value, stable by index).
    LOGN must be the smallest integer such that 2**LOGN >= N. We use 1 << LOGN = upper bound for partner loop.
    """
    pid = tl.program_id(0)
    lanes = tl.num_programs(0) * tl.num_warps(0)  # not used directly; each program handles one lane
    # Bitonic sort network:
    # For k in 1..LOGN-1:
    #   For i in 0..N-1:
    #       partner = i ^ k
    #       dir_asc = ( (i & k) == 0 )
    #       compare-swap and, for ties, keep original order by index.
    # We perform updates for each lane only once per pair (i > partner) to avoid double writes.
    for k in range(1, LOGN):
        for i in range(0, N):
            partner = i ^ k
            # Only process each pair once
            if i > partner:
                continue
            # Load keys and original indices
            ai = tl.load(idxs_ptr + i)
            ap = tl.load(idxs_ptr + partner)
            a = tl.load(values_ptr + ai)
            b = tl.load(values_ptr + ap)
            dir_asc = ( (i & k) == 0 )
            gt = a > b
            lt = a < b
            # Stability: if equal, preserve original index order (smaller index first)
            equal = ~(gt | lt)
            tie = equal & (i > partner)
            swap = tl.where(dir_asc, gt | tie, lt | tie)
            vi = tl.load(idxs_ptr + i)
            vp = tl.load(idxs_ptr + partner)
            new_i = tl.where(swap, vp, vi)
            new_partner = tl.where(swap, vi, vp)
            tl.store(idxs_ptr + i, new_i)
            tl.store(idxs_ptr + partner, new_partner)


@triton.jit
def histogram_kernel(topk_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    """
    Histogram of int32 values in topk_ptr into counts_ptr (length = num_experts=256).
    N is runtime, BLOCK is number of elements each program processes.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(topk_ptr + offsets, mask=mask, other=0)
    for o in range(0, BLOCK):
        if mask[o]:
            val = vals[o]
            tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def prefix_sum_kernel(counts_ptr, offsets_ptr, NUM_BINS: tl.constexpr):
    """
    Compute inclusive prefix sums of counts_ptr into offsets_ptr.
    counts_ptr: 1D int32, length NUM_BINS (num_experts).
    offsets_ptr: 1D int32, length NUM_BINS + 1.
    """
    acc = 0
    tl.store(offsets_ptr + 0, 0)
    for b in range(0, NUM_BINS):
        acc += tl.load(counts_ptr + b)
        tl.store(offsets_ptr + b + 1, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation of run:
        - sorted_token_indices: permutation of 0..N-1 using stable bitonic sort of flattened topk_idx values.
        - expert_offsets: inclusive prefix sum of per-expert counts from flattened topk_idx.
        """
        # Ensure on CUDA
        if not topk_idx.is_cuda:
            raise RuntimeError("topk_idx must be on CUDA device for Triton kernels.")
        # Flatten
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        num_experts = 256

        # 1) Stable sort using Triton bitonic sort
        sorted_perm = torch.empty(N, dtype=torch.int32, device=flat.device)
        # Initialize permutation to [0..N-1]
        sorted_perm = torch.arange(N, dtype=torch.int32, device=flat.device)
        LOGN = math.ceil(math.log2(max(N, 1))) if N > 1 else 1
        # Each program handles one lane (i). We run N programs.
        grid = (N,)
        bitonic_sort_stable_kernel[grid](flat, sorted_perm, N, LOGN)

        # 2) Histogram of expert IDs via Triton (flat is int32)
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        BLOCK = 1024
        grid_hist = (triton.cdiv(N, BLOCK),)
        histogram_kernel[grid_hist](flat, counts, N, BLOCK)

        # 3) Inclusive prefix sums via Triton
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        grid_ps = (1,)
        prefix_sum_kernel[grid_ps](counts, expert_offsets, num_experts)

        # Return sorted permutation and expert offsets
        # sorted_token_indices corresponds to the permutation idxs from the kernel
        return sorted_perm, expert_offsets


def run(*args):
    return ModelNew()(*args)
