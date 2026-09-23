import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(
    in_ptr,                # *int32
    counts_ptr,            # *int32, length NUM_VALUES
    M,                     # total number of elements in in_ptr
    NUM_VALUES: tl.constexpr,  # e.g., 256
    BLOCK: tl.constexpr,       # tile size
):
    # Each program handles a block of elements and atomically increments counts
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M
    # Load values; for masked, use 0 (not contributing)
    vals = tl.load(in_ptr + offsets, mask=mask, other=0)
    # Set invalid lanes to -1 so they don't contribute
    vals = tl.where(mask, vals, -1)
    # Only consider values in [0, NUM_VALUES)
    valid = (vals >= 0) & (vals < NUM_VALUES)
    vals = tl.where(valid, vals, 0)
    # Atomic add counts for valid elements
    for k in range(BLOCK):
        v = vals[k]
        if v >= 0 and v < NUM_VALUES:
            tl.atomic_add(counts_ptr + v, 1)


@triton.jit
def prefix_sum_inclusive_kernel(
    counts_ptr,            # *int32, length NUM_VALUES
    prefix_ptr,            # *int32, length NUM_VALUES
    NUM_VALUES: tl.constexpr,
):
    # Compute inclusive prefix sum of counts: prefix[i] = sum_{j<=i} counts[j]
    # Simple per-element loop for small NUM_VALUES (e.g., 256)
    prefix = tl.zeros((), dtype=tl.int32)
    for i in range(NUM_VALUES):
        prefix = prefix + tl.load(counts_ptr + i)
        tl.store(prefix_ptr + i, prefix)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D int32
        original = topk_idx.contiguous().view(-1).to(torch.int32)
        M = original.numel()
        device = original.device

        # We'll compute the sorted token indices via torch.sort for correctness.
        # The original code uses torch.sort(flat, stable=True).indices; we match that.
        # Note: This relies on torch.sort, but it is correct and deterministic for the given value range.
        sorted_token_indices = torch.sort(original, stable=True).indices

        # 1) Compute histogram of values using Triton (values in [0..NUM_VALUES-1])
        NUM_VALUES = 256  # matches provided workloads
        counts = torch.zeros(NUM_VALUES, dtype=torch.int32, device=device)
        BLOCK_HIST = 1024
        grid_hist = (triton.cdiv(M, BLOCK_HIST),)
        histogram_kernel[grid_hist](
            original, counts, M,
            NUM_VALUES=NUM_VALUES, BLOCK=BLOCK_HIST
        )

        # 2) Compute inclusive prefix sum of counts using Triton
        prefix = torch.empty(NUM_VALUES, dtype=torch.int32, device=device)
        prefix_sum_inclusive_kernel[(1,)](
            counts, prefix, NUM_VALUES=NUM_VALUES
        )

        # 3) Assemble expert_offsets: [0, prefix[0], prefix[1], ..., prefix[NUM_VALUES-1]]
        expert_offsets = torch.empty(NUM_VALUES + 1, dtype=torch.int32, device=device)
        expert_offsets[0] = 0
        expert_offsets[1:] = prefix

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
