import torch
import triton
import triton.language as tl


@triton.jit
def bincount_kernel(x_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    """
    Compute per-element counts of values in [0, 255] into counts_ptr[0:256] using atomics.
    Assumes x_ptr points to int32 values; out-of-range values are ignored.
    Grid: (num_blocks,)
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load values; invalid lanes get 0
    x = tl.load(x_ptr + offsets, mask=mask, other=0)

    # Only count valid lanes
    valid = mask & (x >= 0) & (x <= 255)
    # For valid lanes, add 1 to counts[x]
    for i in range(256):
        # Increment counts[i] for all valid elements where x == i
        tl.atomic_add(counts_ptr + i, valid & (x == i).to(tl.int32))


@triton.jit
def inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, L: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr[0:L-1] into offsets_ptr[0:L],
    where offsets[0] = 0 and offsets[i] = offsets[i-1] + counts[i-1].
    Grid: (1,)
    """
    # This is a simple sequential loop; L is known at compile time (257).
    # offsets_ptr is int64; counts_ptr is int32
    acc = tl.zeros((), dtype=tl.int64)
    # Initialize first element (offsets[0] = 0). We can set it before launch.
    # Loop over remaining L-1 elements
    for i in range(1, L):
        acc += tl.load(counts_ptr + (i - 1)).to(tl.int64)
        tl.store(offsets_ptr + i, acc)


@triton.jit
def stable_argsort_bitonic_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """
    Stable argsort of x_ptr[0:N] (int32 values in [0, 255]) into out_ptr[0:N] (indices).
    Implements bitonic sort network in registers per lane. For each block, we sort a contiguous
    segment of length BLOCK. We assume N is divisible by BLOCK or mask the excess.
    Grid: (num_blocks,)
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load input values; invalid lanes get 0 (neutral for sorting)
    keys = tl.load(x_ptr + offsets, mask=mask, other=0)
    idxs = offsets  # initial indices

    # Bitonic sort network over BLOCK elements
    # We perform a network that sorts the BLOCK-sized vector (keys, idxs) in ascending order,
    # using the standard compare-exchange pattern with stable tie-break by original index.
    # Since BLOCK is constexpr, Triton can unroll this.
    size = 2
    while size <= BLOCK:
        stride = size // 2
        while stride > 0:
            partner = offsets ^ stride
            pkeys = keys
            pidxs = idxs
            qkeys = keys[partner]
            qidxs = idxs[partner]

            # Determine direction: ascending if (offsets & size) == 0
            dir_asc = (offsets & size) == 0

            # Compare and decide: for ascending, take min; for descending, take max.
            # Stable tie-break: prefer the smaller original index when equal.
            less = pkeys < qkeys
            equal = pkeys == qkeys
            pkeep = tl.where(less, True, tl.where(equal, pidxs < qidxs, dir_asc))
            take_p = pkeep

            new_keys = tl.where(take_p, pkeys, qkeys)
            new_idxs = tl.where(take_p, pidxs, qidxs)

            keys = new_keys
            idxs = new_idxs
            stride //= 2
        size *= 2

    # Store sorted indices (stable)
    tl.store(out_ptr + offsets, idxs, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        """
        topk_idx: (B, S, EPT), int32, on device
        Returns:
          sorted_token_indices: int32 permutation of length N
          expert_offsets: int64 tensor of length 257 (cumsum of bincount)
        """
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton"
        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # 1) Triton bincount into counts[0:256] (int32)
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        BLOCK = 1024  # tune as needed
        grid = (triton.cdiv(N, BLOCK),)
        bincount_kernel[grid](flat, counts, N, BLOCK=BLOCK)

        # 2) Triton inclusive prefix sum to produce offsets (length 257, int64)
        offsets = torch.empty(257, dtype=torch.int64, device=flat.device)
        offsets[0] = 0  # inclusive, so starting at 0
        inclusive_prefix_sum_kernel[(1,)](counts, offsets, L=257)

        # 3) Triton stable argsort to produce sorted_token_indices (int32 permutation)
        #    Since values are in [0, 255], bitonic sort by original indices is correct.
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=flat.device)
        # Ensure BLOCK divides N for simpler masking. We can still mask with N.
        BLOCK_SORT = 1024
        grid_sort = (triton.cdiv(N, BLOCK_SORT),)
        stable_argsort_bitonic_kernel[grid_sort](flat, sorted_token_indices, N, BLOCK=BLOCK_SORT)

        return sorted_token_indices, offsets