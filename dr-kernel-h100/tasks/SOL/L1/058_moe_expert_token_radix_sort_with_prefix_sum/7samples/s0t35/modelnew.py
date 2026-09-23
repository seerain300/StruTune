import torch
import triton
import triton.language as tl


@triton.jit
def triton_bincount_kernel(
    flat_ptr,          # *int32, flattened input indices
    counts_ptr,        # *int32, output per-expert counts (length 256)
    N,                 # int32, total number of elements in flat
    BLOCK: tl.constexpr
):
    # Each program handles BLOCK elements
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load flat elements; other=0 for masked-out lanes
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)

    # Atomic add 1 to counts[vals] for valid lanes. Only increment for values in [0, 255].
    # We assume get_inputs produces vals in [0, 255], matching num_experts=256 in the original code.
    # We cast vals to int32 to be explicit.
    vals = vals.to(tl.int32)

    # We need pointer type for counts_ptr; Triton uses element type to infer pointer type.
    # For atomic add, Triton expects counts_ptr to be of type int32*.
    # Mask ensures we only operate on valid lanes.
    # Note: Triton will allow integer indexing into counts_ptr as long as indices are in-range and masked.
    # Since vals are in [0, 255], and we mask based on N, this is safe.
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def triton_inclusive_prefix_sum_kernel(
    counts_ptr,        # *int32, per-expert counts (length 256)
    offsets_ptr,       # *int64, output offsets (length 257)
    L: tl.constexpr    # number of bins to process (257)
):
    # Single-program kernel computes inclusive prefix sum.
    # offsets_ptr has at least length L; we assume L=257 here.
    # Initialize the first element to 0. The caller does this.
    running = tl.load(offsets_ptr + 0)  # int64
    # Loop over bins 1..L-1 and accumulate
    for i in range(1, L):
        # Load count for bin i-1 as int32, then cast to int64 for accumulation
        cnt = tl.load(counts_ptr + (i - 1)).to(tl.int64)
        running = running + cnt
        tl.store(offsets_ptr + i, running)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version:
        - Triton: bincount of expert ids in [0, 255] and inclusive prefix sum of counts.
        - PyTorch: stable argsort for flattened indices to produce sorted_token_indices.
        """
        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # 1) Triton bincount: counts[0..255] as int32
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        triton_bincount_kernel[grid](flat, counts, N, BLOCK=BLOCK)

        # 2) Triton inclusive prefix sum to produce offsets (length 257, int64)
        offsets = torch.empty(257, dtype=torch.int64, device=flat.device)
        offsets[0] = 0  # inclusive, starting at 0
        triton_inclusive_prefix_sum_kernel[(1,)](counts, offsets, L=257)

        # 3) Use PyTorch for stable argsort of flattened indices (returns permutation of [0, N-1])
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, offsets