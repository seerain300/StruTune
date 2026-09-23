import math
import torch
import triton
import triton.language as tl


@triton.jit
def bitonic_sort_stable_kernel(topk_ptr, idxs_ptr, N, NUM_STEPS: tl.constexpr):
    """
    Sort the flattened values in topk_ptr (int32) using a bitonic sorting network,
    and write the permutation of indices to idxs_ptr (int32).
    Stable: for equal keys, original index (lane id) determines order (keep smaller index first).

    - topk_ptr: 1D int32 tensor of length N, to be sorted.
    - idxs_ptr: 1D int32 tensor of length N, initialized to [0..N-1], updated with sorted permutation.
    - N: total number of elements (runtime).
    - NUM_STEPS: int = ceil(log2(N)), passed as constexpr at launch.
    """
    # Each program handles one lane i (element in the flattened array).
    i = tl.program_id(0)

    # Initialize idxs[i] = i
    # (Assume idxs_ptr is pre-initialized to [0..N-1] on host side; we only update if needed.)
    # We will read/write idxs_ptr in-place during sorting.

    # Bitonic sort network
    for k in range(0, NUM_STEPS):
        step = 1 << (k + 1)
        for j in range(k, -1, -1):
            size = 1 << j
            partner = i ^ size
            # Only process each pair once
            if partner > i:
                # Load current values/indices
                vi = tl.load(idxs_ptr + i)
                vp = tl.load(idxs_ptr + partner)
                a = tl.load(topk_ptr + vi)
                b = tl.load(topk_ptr + vp)

                # Direction: ascending if (i & step) == 0, else descending
                dir_asc = ( (i & step) == 0 )

                # Compare
                gt = a > b
                lt = a < b
                equal = ~(gt | lt)  # a == b

                # Stability: when equal, prefer smaller original index first
                tie = equal & (i > partner)  # if equal and i > partner, swap to keep i before partner

                if dir_asc:
                    swap = gt | tie
                else:
                    swap = lt | tie

                if swap:
                    # Swap idxs[i] and idxs[partner]
                    tmp = vi
                    vi = vp
                    vp = tmp

                # Write back (only the current lane writes its position)
                tl.store(idxs_ptr + i, vi)
                # partner lane would write vp to its position (handled by other program with i=partner)


@triton.jit
def histogram_kernel(topk_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    """
    Histogram of int32 values in topk_ptr into counts_ptr.
    - topk_ptr: flattened 1D int32 tensor of length N.
    - counts_ptr: 1D int32 tensor of length num_experts (256).
    - N: total elements (runtime).
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
    - counts_ptr: 1D int32, length NUM_BINS (num_experts).
    - offsets_ptr: 1D int32, length NUM_BINS + 1.
    """
    # offsets_ptr[0] = 0
    tl.store(offsets_ptr + 0, 0)
    acc = 0
    for b in range(0, NUM_BINS):
        acc += tl.load(counts_ptr + b)
        tl.store(offsets_ptr + b + 1, acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation of the original run logic:
        - Flattened stable sort (bitonic) using Triton and write permutation indices.
        - Histogram of expert IDs via Triton.
        - Prefix sums for expert offsets via Triton.
        """
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"
        # Flatten
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        num_experts = 256  # fixed per original code

        # 1) Stable sort permutation using Triton bitonic sort
        # Prepare idxs initialized to [0..N-1]
        idxs = torch.arange(N, dtype=torch.int32, device=flat.device)
        # Launch bitonic sort kernel
        # NUM_STEPS = ceil(log2(N))
        if N <= 1:
            # Trivial case
            sorted_token_indices = idxs
        else:
            num_steps = int(math.ceil(math.log2(N)))
            grid = (N,)
            bitonic_sort_stable_kernel[grid](flat, idxs, N, num_steps)

            sorted_token_indices = idxs

        # 2) Triton histogram of flattened topk values
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        BLOCK = 1024
        grid_h = (triton.cdiv(N, BLOCK),)
        histogram_kernel[grid_h](flat, counts, N, BLOCK)

        # 3) Triton prefix sum to get inclusive offsets
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        prefix_sum_kernel[(1,)](counts, offsets, num_experts)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
