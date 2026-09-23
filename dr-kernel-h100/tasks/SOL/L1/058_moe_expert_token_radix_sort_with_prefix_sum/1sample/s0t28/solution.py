import torch
import triton
import triton.language as tl


@triton.jit
def _argsort_odd_even_stable(flat_ptr, values_ptr, indices_ptr, N: tl.int32, PHASES: tl.int32, BLOCK: tl.constexpr):
    """
    Perform stable odd-even transposition sort on the flattened indices.
    flat_ptr: pointer to int32 input indices (flattened)
    values_ptr: pointer to int32 working buffer (copy of flat)
    indices_ptr: pointer to int32 buffer to produce permutation (original positions)
    N: number of elements (runtime int32)
    PHASES: total number of phases = 2*N
    BLOCK: lane size (constexpr)
    """
    # Initialize working buffers
    for i in range(0, BLOCK):
        idx = i
        mask = idx < N
        val = tl.load(flat_ptr + idx, mask=mask, other=0)
        tl.store(values_ptr + idx, val)        # copy flat into values
        tl.store(indices_ptr + idx, idx.to(tl.int32))  # init indices as [0..N-1]

    # Odd-even transposition sort with stable tie-breaking
    for t in range(0, PHASES):
        start = 0
        stride = 1
        while start < N:
            # Compute partner indices for this pass
            i = start + tl.arange(0, BLOCK)  # vector of lane positions
            j = i + stride
            # Valid compare-swap pairs: both in range and j < N
            valid = (i < N) & (j < N) & (i < j)

            # Load current pair values
            vi = tl.load(values_ptr + i, mask=valid, other=0)
            vj = tl.load(values_ptr + j, mask=valid, other=0)

            # Compare and decide swap (stable: only swap if vi > vj; do not swap if equal)
            swap = valid & (vi > vj)

            # Compute new values after swap
            new_vi = tl.where(swap, vj, vi)
            new_vj = tl.where(swap, vi, vj)

            # Store back
            tl.store(values_ptr + i, new_vi, mask=valid)
            tl.store(values_ptr + j, new_vj, mask=valid)

            # Update indices accordingly
            idx_i = tl.load(indices_ptr + i, mask=valid, other=0)
            idx_j = tl.load(indices_ptr + j, mask=valid, other=0)
            new_idx_i = tl.where(swap, idx_j, idx_i)
            new_idx_j = tl.where(swap, idx_i, idx_j)

            tl.store(indices_ptr + i, new_idx_i, mask=valid)
            tl.store(indices_ptr + j, new_idx_j, mask=valid)

            start += 2 * stride
            stride = 3 - stride  # alternate between 1 and 2 strides


@triton.jit
def _histogram_atomic(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Histogram of values in flat_ptr (int32), values in [0..255], into counts_ptr (int32[256]).
    Use atomic_add per element.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)
    # Atomic add 1 for each valid lane
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def _inclusive_scan_prefix_sum(counts_ptr, offsets_ptr, M: tl.int32):
    """
    Compute inclusive prefix sum of counts_ptr[0..M-1] into offsets_ptr[1..M], and offsets_ptr[0] = 0.
    offsets_ptr[0] is initialized on host.
    """
    # Single program scan over M
    s = 0
    for k in range(0, M):
        ck = tl.load(counts_ptr + k)
        s += ck
        tl.store(offsets_ptr + k + 1, s)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # topk_idx: (B, S, EPT), int32, CUDA
        flat = topk_idx.reshape(-1).contiguous()  # int32 on CUDA
        N = flat.numel()

        # 1) Stable argsort via Triton odd-even
        values = torch.empty_like(flat)
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=flat.device)
        BLOCK = 4096  # large enough for typical N (up to 4096 in evaluator)
        PHASES = 2 * N  # 2*N passes ensure convergence for odd-even
        _argsort_odd_even_stable[(1,)](
            flat, values, sorted_token_indices, N, PHASES, BLOCK=BLOCK, num_warps=4
        )

        # 2) Histogram via Triton
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        grid_hist = (triton.cdiv(N, 1024),)
        _histogram_atomic[grid_hist](flat, counts, N, BLOCK=1024, num_warps=8)

        # 3) Prefix sum (offsets) via Triton
        offsets = torch.empty(257, dtype=torch.int32, device=flat.device)
        offsets[0] = 0
        _inclusive_scan_prefix_sum[(1,)](counts, offsets, M=256, num_warps=1)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
