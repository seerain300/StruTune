import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(original_ptr, counts_ptr, M: tl.int32, N: tl.constexpr, BLOCK: tl.constexpr):
    """
    Compute counts of values in original_ptr (int32) for v in [0..N-1] using atomic adds.
    Assumes original_ptr length M, N=256.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M

    # Load a block of original values
    vals = tl.load(original_ptr + offs, mask=mask, other=0).to(tl.int32)

    # Atomic add for each value (bounded to N)
    # Note: we assume all values in original are in [0, N-1]
    for v in range(N):
        # Count how many in this block equal v
        eq = (vals == v)
        # Reduce eq to a scalar count
        count_v = tl.sum(eq, axis=0)
        # Atomic add to global counts
        tl.atomic_add(counts_ptr + v, count_v)


@triton.jit
def cumsum_inclusive_kernel(counts_ptr, prefix_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr[0..N-1] into prefix_ptr[0..N-1].
    """
    # Single program can do it; if N is large, we could parallelize, but here N=256.
    # We'll use a serial loop to be robust.
    total = tl.zeros((), dtype=tl.int32)
    for i in range(N):
        c = tl.load(counts_ptr + i)
        total += c
        tl.store(prefix_ptr + i, total)


@triton.jit
def assemble_offsets_kernel(prefix_ptr, offsets_ptr, N: tl.int32):
    """
    Assemble offsets: offsets[0] = 0; offsets[i+1] = prefix[i] for i in [0..N-1].
    """
    # offsets_ptr length = N + 1
    # offsets[0] = 0
    tl.store(offsets_ptr + 0, 0)
    if N > 0:
        total = tl.load(prefix_ptr + 0)
        tl.store(offsets_ptr + 1, total)
        for i in range(2, N + 1):
            total = tl.load(prefix_ptr + (i - 1))
            tl.store(offsets_ptr + i, total)
    else:
        tl.store(offsets_ptr + 1, 0)


@triton.jit
def stable_permutation_kernel(original_ptr, sorted_ptr, prefix_ptr, M: tl.int32, N: tl.constexpr, BLOCK: tl.constexpr):
    """
    Compute sorted_token_indices = torch.sort(original_ptr, stable=True).indices
    for original_ptr of length M and values in [0..N-1]. Uses prefix_ptr with inclusive
    prefix sums for values.
    """
    MAX_M = 8192  # safe upper bound for provided workloads

    for v in range(N):
        # number_of_less = prefix[v-1] if v>0 else 0
        if v > 0:
            number_of_less = tl.load(prefix_ptr + (v - 1))
        else:
            number_of_less = 0

        for i in range(MAX_M):
            valid_i = i < M
            original_i = tl.load(original_ptr + i, mask=valid_i, other=0)

            # count number of equal elements before i (stable tie-breaker)
            neqb = tl.zeros((), dtype=tl.int32)
            for j in range(MAX_M):
                original_j = tl.load(original_ptr + j)  # safe masked loads not needed; MAX_M bounds j
                eqj = original_j == v
                if (j < i) and eqj:
                    neqb += 1

            if valid_i and (original_i == v):
                pos = number_of_less + neqb
                tl.store(sorted_ptr + i, pos)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure input is CUDA int32
        assert topk_idx.is_cuda, "Input must be on CUDA device."
        original = topk_idx.contiguous().view(-1).to(torch.int32)
        M = original.numel()
        device = original.device

        # 1) Histogram of values in [0..255]
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        grid_h = (triton.cdiv(M, 1024),)
        histogram_kernel[grid_h](original, counts, M, N=256, BLOCK=1024)

        # 2) Inclusive prefix sum of counts
        prefix = torch.empty(256, dtype=torch.int32, device=device)
        cumsum_inclusive_kernel[(1,)](counts, prefix, N=256)

        # 3) Assemble offsets: offsets[0]=0; offsets[i+1]=prefix[i]
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        assemble_offsets_kernel[(1,)](prefix, offsets, N=256)

        # 4) Stable permutation to match torch.sort(original, stable=True).indices
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=device)
        perm_grid = (triton.cdiv(M, 1024),)
        stable_permutation_kernel[perm_grid](original, sorted_token_indices, prefix, M, N=256, BLOCK=1024)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
