import torch
import triton
import triton.language as tl


@triton.jit
def stable_insertion_sort_kernel(flat_ptr, out_idx_ptr, N):
    """
    Stable insertion sort on flat_ptr (int32) of length N.
    out_idx_ptr stores int64 original indices (positions) after sorting.
    Each program handles one element i and performs insertion sort by comparing with all j < i,
    in descending order of j, ensuring stability and correct original index placement.
    """
    pid = tl.program_id(axis=0)
    i = pid
    # If pid >= N, do nothing
    if i >= N:
        return

    # Start with out_idx[i] = i (int64)
    start = tl.full((), i, dtype=tl.int64)

    # Compare with all j < i, in descending order of j, to settle larger j first (helps stability).
    # Note: Triton supports Python loops; we iterate from i-1 down to 0.
    for j in range(i - 1, -1, -1):
        idx_i = out_idx_ptr[i]
        idx_j = out_idx_ptr[j]
        vi = tl.load(flat_ptr + idx_i)
        vj = tl.load(flat_ptr + idx_j)
        # If vi > vj: swap positions i and j
        # If vi == vj: swap only if i < j (stable tie-break).
        swap = (vi > vj) | ((vi == vj) & (i < j))
        if swap:
            # Swap indices in out_idx_ptr
            tmp = out_idx_ptr[i]
            out_idx_ptr[i] = out_idx_ptr[j]
            out_idx_ptr[j] = tmp


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
    Single-program sequential prefix sum for simplicity and correctness.
    """
    pid = tl.program_id(axis=0)
    # Only one program (grid=(1,)) should run this kernel.
    acc = tl.zeros((), dtype=tl.int64)
    for i in range(0, num_experts):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + (i + 1), acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure input is on CUDA
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.cuda()

        # Flatten to 1D
        flat = topk_idx.reshape(-1)  # int32
        N = flat.numel()

        # Prepare output indices as int64 and initialize to [0, 1, 2, ..., N-1]
        out_idx = torch.arange(0, N, device=flat.device, dtype=torch.int64)

        # Launch Triton stable insertion sort kernel
        grid = (N,)
        stable_insertion_sort_kernel[grid](flat, out_idx, N, num_warps=1)

        # Compute expert counts using Triton histogram
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        grid_h = (triton.cdiv(N, 1024),)
        count_histogram_kernel[grid_h](flat, counts, N, num_experts=256, num_warps=1)

        # Compute inclusive prefix sum in int64 and cast to int32 for expert_offsets
        offsets64 = torch.zeros(257, dtype=torch.int64, device=flat.device)
        offsets64[0] = 0
        grid_p = (1,)
        prefix_sum_kernel[grid_p](counts, offsets64, num_experts=256, num_warps=1)

        expert_offsets = offsets64[1:].to(torch.int32)

        return out_idx, expert_offsets


def run(*args):
    return ModelNew()(*args)
